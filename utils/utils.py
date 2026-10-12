"""
PathFormer
Copyright (C) 2026  Liam Brydon
Contact at: liam.brydon@auckland.ac.nz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import copy
import itertools
import json
import math
import os
import re
import sqlite3
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from statistics import mean
import networkx as nx
import numpy as np
import orjson
import pandas as pd
import torch
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from networkx.utils import edges_equal
from rdkit import RDLogger, Chem
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split, KFold
from torch import nn, Tensor, tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import Dataset
from tqdm import tqdm


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


class TransformerReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, model, mode="max", factor=0.2, patience=2, min_lr=1e-10):
        self.model = model
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)

    def _reduce_lr(self, epoch):
        new_lr = self.model.lr * self.factor
        if new_lr > self.default_min_lr:
            self.model.lr = new_lr


class BigDataset(Dataset):
    def __init__(self, *lists_of_tensors, **kwargs):
        self.pad_id = kwargs["pad_id"]
        assert all([len(lists_of_tensors[0]) == len(t) for t in lists_of_tensors])
        self.lists_of_tensors = lists_of_tensors

    def __getitem__(self, item):
        return [t[item] for t in self.lists_of_tensors]

    def __len__(self):
        return len(self.lists_of_tensors[0])

    def collate_batch(self, batch):
        batch = zip(*batch)
        padded = []
        for i, b in enumerate(batch):
            pad_id = self.pad_id[i] if isinstance(self.pad_id, list) else self.pad_id
            b = pad_sequence(b, batch_first=True, padding_value=pad_id)
            if b.dtype == torch.uint8 or b.dtype == torch.uint16 or b.dtype == torch.uint32:
                b = b.long()
            padded.append(b)
        return padded


class PathFormerPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, positions: Tensor = None) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            positions: Tensor, shape ``[batch_size, seq_len]`` containing absolute position indices
        """
        if positions is None:
            # Default to standard sequential positions (e.g., during training)
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), x.size(1))

        # Gather exact positional embeddings for each token in the batch
        # self.pe is [1, max_len, d_model], so squeeze(0) makes it [max_len, d_model]
        pe = self.pe.squeeze(0)[positions].to(x.dtype)
        x = x + pe
        return self.dropout(x)


def encode_seq(*seqs, char_to_i, max_len):
    seqs = [s.split(" ") if isinstance(s, str) else s for s in seqs]
    d_type = torch.uint8 if len(char_to_i) < 256 else torch.uint16
    enc_seqs = []
    for seq in seqs:
        if len(seq) > max_len:
            enc_seqs.append(None)
        else:
            enc_seqs.append(tensor([char_to_i[token] for token in seq], dtype=d_type))
    return enc_seqs[0] if len(enc_seqs) == 1 else enc_seqs


def seq_metrics_pathformer(pred_is: Tensor, y: Tensor, pad_mask: Tensor) -> tuple:
    char_eq_masked = (pred_is == y) & pad_mask
    char_accuracy = char_eq_masked.sum() / pad_mask.sum()
    seq_eq = torch.all((pred_is == y) | (~pad_mask), dim=1)
    seq_accuracy = torch.count_nonzero(seq_eq) / seq_eq.size(0)
    return seq_accuracy, char_accuracy


def norm_length(seq_lls, mask, length_norm=None):
    """ Normalise log-likelihoods using the length of the constructed sequence
    Equation from:
    Wu, Yonghui, et al.
    "Google's neural machine translation system: Bridging the gap between human and machine translation."
    arXiv preprint arXiv:1609.08144 (2016).

    Args:
        seq_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size] containing log likelihoods for seqs so far
        mask (torch.Tensor): BoolTensor of shape [batch_size, seq_len] containing the padding mask
        length_norm (int): Length norm value

    Returns:
        norm_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size]

    """

    if length_norm is not None:
        seq_lengths = (~mask).sum(dim=1)
        norm = torch.pow(5 + seq_lengths, length_norm) / pow(6, length_norm)
        norm_lls = (seq_lls.T / norm).T
        return norm_lls

    return seq_lls


def beam_step_pathformer(model, decode_func, tokens, lls, idx0_pos, step, kv_cache=None):
    """
    Args:
        tokens: (B, S) current beam tokens
        lls:    (B,)   accumulated log-likelihoods
        idx0_pos: (B,) prefix end index
        step:   int    generation step (0 = first generated token)
        kv_cache: dict containing past Keys and Values
    """
    B = tokens.size(0)
    device = tokens.device
    current_pos = idx0_pos + step  # (B,)

    if step == 0:
        # Prefill Phase: Pass the full sequences up to the current position
        max_pos = int(current_pos.max().item())
        input_tokens = tokens[:, :max_pos + 1]
        positions = torch.arange(max_pos + 1, device=device).unsqueeze(0).expand(B, max_pos + 1)
        output_dist = decode_func(input_tokens, kv_cache=kv_cache, positions=positions)  # (B, S, V)
        # Gather logits at the correct per-sample position
        next_token_lls = output_dist[torch.arange(B, device=device), current_pos]  # (B, V)
    else:
        # Decode Phase: Pass ONLY the most recently generated token
        positions = current_pos.unsqueeze(1)
        input_tokens = tokens[torch.arange(B, device=device), current_pos].unsqueeze(1)
        output_dist = decode_func(input_tokens, kv_cache=kv_cache, positions=positions) # (B, 1, V)
        # S is 1, so the prediction for the next token is just index 0
        next_token_lls = output_dist[:, 0, :] # (B, V)

    # Completion Mask Logic
    V = next_token_lls.size(-1)
    complete_seq_ll = torch.full((1, V), -1e5, device=device)
    complete_seq_ll[:, model.pad_id] = 0.0
    last_tokens = tokens[torch.arange(B, device=device), current_pos]
    is_end_token = last_tokens == getattr(model, "end_id", model.char_to_i.get("eom"))
    is_pad_token = last_tokens == model.pad_id
    ll_mask = (is_end_token | is_pad_token).unsqueeze(1)   # (B,1)
    masked_lls = (ll_mask * complete_seq_ll) + ((~ll_mask) * next_token_lls)

    # Accumulate log-likelihoods
    seq_lls = lls.unsqueeze(1) + masked_lls   # (B, V)
    return seq_lls, masked_lls


def update_beams_pathformer_flat(step, model, decode_func, flat_tokens, flat_pad_mask, flat_lls, flat_token_ll_hist,
                                 flat_idx0_pos, flat_cache, B, num_beams):
    device = flat_tokens.device
    flat_B = B * num_beams
    current_pos = flat_idx0_pos + step

    # Single Vectorized Forward Pass
    input_tokens = flat_tokens[torch.arange(flat_B, device=device), current_pos].unsqueeze(1)
    # Pass the exact current position for the flattened batch
    positions = current_pos.unsqueeze(1)
    output_dist = decode_func(input_tokens, kv_cache=flat_cache, positions=positions)  # (B * num_beams, 1, V)
    next_token_lls = output_dist[:, 0, :]  # (B * num_beams, V)

    # Completion Mask Logic
    V = next_token_lls.size(-1)
    complete_seq_ll = torch.full((1, V), -1e5, device=device)
    complete_seq_ll[:, model.pad_id] = 0.0
    last_tokens = flat_tokens[torch.arange(flat_B, device=device), current_pos]

    end_id = getattr(model, "end_id", model.char_to_i.get("eom"))
    is_end_token = last_tokens == end_id
    is_pad_token = last_tokens == model.pad_id
    ll_mask = (is_end_token | is_pad_token).unsqueeze(1)
    masked_lls = (ll_mask * complete_seq_ll) + ((~ll_mask) * next_token_lls)
    seq_lls = flat_lls.unsqueeze(1) + masked_lls  # (B * num_beams, V)

    # Apply length normalization for ranking
    norm_lls = norm_length(seq_lls, flat_pad_mask, model.model_config["length_norm"])

    # Reshape both normalized and unnormalized tensors to (B, num_beams * V)
    norm_lls_reshaped = norm_lls.view(B, num_beams * V)
    seq_lls_reshaped = seq_lls.view(B, num_beams * V)

    # Use NORMALIZED scores to find the best paths
    _, top_idxs = torch.topk(norm_lls_reshaped, num_beams, dim=1)  # (B, num_beams)

    # Use the indices to gather the UNNORMALIZED scores to carry forward to the next step
    top_lls = torch.gather(seq_lls_reshaped, dim=1, index=top_idxs)

    new_ids = top_idxs % V  # (B, num_beams)
    parent_beams = top_idxs // V  # (B, num_beams)

    # Extract the token_lls for the chosen tokens
    chosen_token_lls = torch.gather(masked_lls.view(B, num_beams * V), dim=1, index=top_idxs).view(-1)

    # Fast Cache Re-alignment
    # Convert relative parent indices (0 to num_beams-1) to absolute flat indices (0 to flat_B-1)
    batch_offsets = (torch.arange(B, device=device) * num_beams).unsqueeze(1)  # (B, 1)
    flat_parent_indices = (batch_offsets + parent_beams).view(-1)  # (B * num_beams)

    # Re-align the cache IN-PLACE safely. No new tensor allocations required.
    for k_key, v_val in flat_cache.items():
        if isinstance(v_val, tuple):
            # Perform in-place memory copy to prevent fragmentation
            v_val[0].copy_(v_val[0][flat_parent_indices])
            v_val[1].copy_(v_val[1][flat_parent_indices])
        elif isinstance(v_val, torch.Tensor):
            v_val.copy_(v_val[flat_parent_indices])
        else:
            flat_cache[k_key] = v_val  # Ints or unchanged variables

    # Re-align tokens and history using the same indices
    flat_tokens.copy_(flat_tokens[flat_parent_indices])
    flat_pad_mask.copy_(flat_pad_mask[flat_parent_indices])
    flat_token_ll_hist.copy_(flat_token_ll_hist[flat_parent_indices])
    flat_lls.copy_(top_lls.view(-1))

    # Insert new generated tokens
    flat_pad_mask_new = (new_ids.view(-1) == end_id) | (new_ids.view(-1) == model.pad_id)
    final_pos = model.max_len - 1

    new_ids_flat = new_ids.view(-1).clone()
    chosen_token_lls = chosen_token_lls.clone()

    if current_pos.max().item() >= final_pos:
        new_ids_flat[~flat_pad_mask_new] = end_id

    new_ids_flat[flat_pad_mask_new] = model.pad_id
    chosen_token_lls[flat_pad_mask_new] = 0.0

    next_pos = current_pos + 1
    flat_tokens[torch.arange(flat_B, device=device), next_pos] = new_ids_flat
    flat_pad_mask[torch.arange(flat_B, device=device), next_pos] = flat_pad_mask_new
    flat_token_ll_hist[torch.arange(flat_B, device=device), next_pos] = chosen_token_lls

    complete = flat_pad_mask_new.all().item()
    return complete


@torch.inference_mode()
def beam_decode_pathformer(model, prefix, idx0_pos, num_beams=5, spectra=None):
    """Prefix-aware beam decoding with KV Caching."""
    if model.__class__.__name__ == "PathFormer":
        decode_func = partial(model.forward, test=True, spectra=spectra)
    elif model.__class__.__name__ == "BinnedPathFormer":
        # The MZ encoder runs exactly once prior to decoding
        z, src_mask = model.encode(spectra)
        decode_func = partial(model.decode, mz_enc_out=z, mz_mask=src_mask, test=True)
    else:
        raise ValueError(f"Unsupported model type: {model.__class__.__name__}")
    # Find shapes for batch size, sequence length and num beams
    B = prefix.size(0)
    device = prefix.device
    S = model.max_len if spectra is None else model.max_len - spectra.shape[1]

    # Create an empty array and empty pad mask and fill with the prefix
    token_ids = torch.full((B, S), model.pad_id, device=device, dtype=torch.long)
    pad_mask = torch.ones((B, S), device=device, dtype=torch.bool)
    for b in range(B):
        token_ids[b, : idx0_pos[b] + 1] = prefix[b, : idx0_pos[b] + 1]
        pad_mask[b, : idx0_pos[b] + 1] = False
    # Initial ts (tokens) is the prefix) initial log likelihoods are zero
    ll = torch.zeros(B, device=device)

    # Initialize the master cache and run the Prefill Step (Step 0) on shape (B, S)
    initial_cache = {}
    first_seq_lls, first_token_lls = beam_step_pathformer(
        model, decode_func, prefix, ll, idx0_pos, step=0, kv_cache=initial_cache
    )

    V = first_seq_lls.size(-1)
    if num_beams > V:
        pad_lls = torch.full((B, num_beams - V), -1e5, device=device)
        first_seq_lls = torch.cat([first_seq_lls, pad_lls], dim=-1)

    # Get topk for the initial branching
    top_lls, top_ids = torch.topk(first_seq_lls, num_beams, dim=-1)  # (B, num_beams)

    if num_beams > V:
        top_ids = torch.where(top_ids >= V, model.pad_id, top_ids)

    # Flatten and expand states to (B * num_beams, ...)
    flat_B = B * num_beams

    # Expand tokens and padding masks
    flat_tokens = token_ids.repeat_interleave(num_beams, dim=0)  # (B * num_beams, S)
    flat_pad_mask = pad_mask.repeat_interleave(num_beams, dim=0)
    flat_token_ll_hist = torch.zeros((flat_B, S), device=device)

    # Expand KV cache safely
    flat_cache = {}
    for k, v in initial_cache.items():
        if isinstance(v, tuple):
            flat_cache[k] = (
                v[0].repeat_interleave(num_beams, dim=0),
                v[1].repeat_interleave(num_beams, dim=0)
            )
        elif isinstance(v, torch.Tensor):
            flat_cache[k] = v.repeat_interleave(num_beams, dim=0)
        else:
            flat_cache[k] = v  # Catch-all for ints, bools, etc.

    # Expand the encoder memory and update decode_func for the flat phase
    if model.__class__.__name__ == "BinnedPathFormer":
        flat_z = z.repeat_interleave(num_beams, dim=0)
        flat_src_mask = src_mask.repeat_interleave(num_beams, dim=0)
        decode_func = partial(model.decode, mz_enc_out=flat_z, mz_mask=flat_src_mask, test=True)
    elif model.__class__.__name__ == "PathFormer":
        flat_spectra = spectra.repeat_interleave(num_beams, dim=0) if spectra is not None else None
        decode_func = partial(model.forward, test=True, spectra=flat_spectra)

    # Insert the initial branched tokens into the flattened tensors
    insert_pos = (idx0_pos + 1).repeat_interleave(num_beams)
    flat_batch_idx = torch.arange(flat_B, device=device)

    flat_tokens[flat_batch_idx, insert_pos] = top_ids.view(-1)
    flat_pad_mask[flat_batch_idx, insert_pos] = False
    flat_token_ll_hist[flat_batch_idx, insert_pos] = top_lls.view(-1)
    flat_lls = top_lls.view(-1)  # (B * num_beams,)
    flat_idx0_pos = idx0_pos.repeat_interleave(num_beams)

    max_gen_steps = S - (idx0_pos.max().item() + 2)

    for step in range(1, max_gen_steps):
        # Pass the flat tensors into the vectorized beam update.
        # Note: The tensors are updated in-place inside this function, so we only need to return the completion status.
        complete = update_beams_pathformer_flat(
            step, model, decode_func,
            flat_tokens, flat_pad_mask, flat_lls, flat_token_ll_hist, flat_idx0_pos, flat_cache,
            B, num_beams
        )
        if complete:
            break

    # Reshape the flat tensors (B * num_beams, ...) back into (B, num_beams, ...) for sorting
    tokens_tensor = flat_tokens.view(B, num_beams, S)
    log_lhs_tensor = flat_lls.view(B, num_beams)
    token_ll_tensor = flat_token_ll_hist.view(B, num_beams, S)

    sorted_mols, sorted_lls, sorted_token_lls = _sort_beams(tokens_tensor, log_lhs_tensor, token_ll_tensor)
    return sorted_mols, sorted_lls, sorted_token_lls


def _sort_beams(mol_strs, log_lhs, all_ll):
    """ Return mols sorted by their log likelihood"""

    assert len(mol_strs) == len(log_lhs)

    sorted_mols = []
    sorted_lls = []
    sorted_all_ll = []

    for mols, lls, all_ls in zip(mol_strs, log_lhs, all_ll):
        mol_lls = sorted(zip(mols, lls, all_ls), reverse=True, key=lambda mol_ll: mol_ll[1])
        mols, lls, all_ls = tuple(zip(*mol_lls))
        sorted_mols.append(torch.stack(mols))
        sorted_lls.append(torch.stack(lls))
        sorted_all_ll.append(torch.stack(all_ls))

    return torch.stack(sorted_mols), torch.stack(sorted_lls), torch.stack(sorted_all_ll)


def get_cannon_func(canon_type):
    if canon_type == "rdkit":
        preprocessor = canon_smile_rdkit
    else:
        raise ValueError(f"{canon_type} is invalid canonical function")
    return preprocessor


def canon_smile_rdkit(smile: str, allow_none=True, remove_stereochemistry=True) -> str | None:
    RDLogger.DisableLog('rdApp.*')
    # Need to split and use a set as sometimes the same mol is repeated just separated by a dot
    parts = []
    parts_set = set()
    for p in smile.split("."):
        mol = Chem.MolFromSmiles(p)
        if allow_none and mol is None:
            return None
        elif mol is None:
            parts_set.add(p)
            parts.append(p)
            continue
        if remove_stereochemistry:
            Chem.RemoveStereochemistry(mol)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DATIVE:
                bond.SetBondType(Chem.BondType.SINGLE)
        new_smile = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not remove_stereochemistry)
        if new_smile not in parts_set:
            parts.append(new_smile)
            parts_set.add(new_smile)
    return ".".join(parts)


def process_seq_test_outputs_pathformer(outputs: list, path: str, i_to_char: dict, length_norm, args) -> dict:
    flat_pred = []
    flat_scores = []
    flat_inputs = []
    flat_true = []
    for inputs, (predict, actual, score) in outputs:
        flat_scores.extend(score.tolist())
        flat_inputs.extend(inputs)
        flat_pred.extend(predict)
        flat_true.extend(actual)
    canon_func = get_cannon_func(args.preprocessor)
    decoded_input = []
    for seq in flat_inputs:
        chars = []
        in_root = False
        for char_idx in seq:
            char = i_to_char[char_idx.item()]
            if char == "[nop]":
                continue
            if char == "<boc>":
                in_root = True
                continue
            if char.startswith("IDX_"):
                break
            if in_root:
                chars.append(char)
        decoded_input.append(canon_func("".join(chars)))
    decoded_true = [" ".join(char for idx in seq if (char := i_to_char[idx.item()]) != "[nop]") for seq in flat_true]
    decoded_pred = [[" ".join(char for idx in seq if (char := i_to_char[idx.item()]) != "[nop]") for seq in beams] for beams in flat_pred]
    decoded_true = [seq_to_pathway_graph(seq, root_smiles=decoded_input[i]) for i, seq in enumerate(decoded_true)]
    decoded_pred = [[seq_to_pathway_graph(seq, seq_probs=flat_scores[i][j][:len(seq)], root_smiles=decoded_input[i]) for j, seq in enumerate(beams)] for i, beams in enumerate(decoded_pred)]

    results_dict = {}
    accuracy = {}
    if flat_pred[0].ndim == 1:
        top_k = [1]
    else:
        top_k = range(1, len(flat_pred[0]) + 1)
    for input_smiles, true_pathway, pred_pathways, scores in zip(decoded_input, decoded_true, decoded_pred, flat_scores):
        true_pathway, _ = true_pathway
        if len(true_pathway.nodes) == 0:
            continue
        if input_smiles not in results_dict:
            results_dict[input_smiles] = {"actual": [], "scores": [],
                                          "invalid_smiles": {f"top_{k}": 0 for k in top_k},
                                          "predict": {f"top_{k}": [] for k in top_k}}
        results_dict[input_smiles]["actual"].append(true_pathway)
        results_dict[input_smiles]["scores"].append(scores)
        for k in top_k:
            results_dict[input_smiles]["predict"][f"top_{k}"].append(pred_pathways[k - 1][0])
            results_dict[input_smiles]["invalid_smiles"][f"top_{k}"] += pred_pathways[k - 1][1]
    for root_smiles in results_dict:
        real_pathways = results_dict[root_smiles]["actual"]
        for k in top_k:
            key = f"top_{k}"
            pred_pathways = results_dict[root_smiles]["predict"][key]
            if key not in accuracy:
                accuracy[key] = []
            for real in real_pathways:
                success = 0
                for pred in pred_pathways:
                    if edges_equal(real.edges, pred.edges, directed=True):
                        success = 1
                        break
                accuracy[key].append(success)
    accuracy = {k: round(sum(v) / len(v), 4) if len(v) else 0.0 for k, v in accuracy.items()}
    for key in results_dict:
        results_dict[key]["actual"] = [nx.node_link_data(pathway) for pathway in results_dict[key]["actual"]]
        results_dict[key].pop("scores")
        for k in results_dict[key]["predict"]:
            results_dict[key]["predict"][k] = [nx.node_link_data(pathway) for pathway in results_dict[key]["predict"][k]]
    save_dict = {"accuracy": accuracy, "predictions": results_dict}
    with open(f"{path}/test_output.json", "w") as out_file:
        json.dump(save_dict, out_file)
    return save_dict


def save_train_metrics(model_metrics: dict[str, dict], path: str, steps_per_epoch: dict):
    metrics = copy.deepcopy(model_metrics)
    for metric in metrics:
        for s_type in metrics[metric]:
            means = [mean(metrics[metric][s_type][i:i + steps_per_epoch[s_type]]) for i in
                     range(0, len(metrics[metric][s_type]), steps_per_epoch[s_type])]
            metrics[metric][s_type] = means

    with open(f"{path}/train_metrics.json", "w") as m_file:
        json.dump(metrics, m_file, indent=4)
    plt.clf()
    fig, axs = plt.subplots(len(metrics), sharex="all")
    fig.set_size_inches(18, 5 * len(metrics))
    plt.suptitle(f"Train Metrics for {path}")
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 26
    axs = axs.flat
    for i, (metric, values) in enumerate(metrics.items()):
        axs[i].set_title(metric, fontsize=BIGGER_SIZE)
        step_num = len(max(values.values(), key=lambda x: len(x)))
        for step_type, value in values.items():
            x_step = step_num // len(value) if len(value) > 0 else 1
            x_coords = list(range(0, step_num, x_step))
            axs[i].plot(x_coords[:len(value)], value, label=step_type)
        axs[i].tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
        axs[i].tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
        axs[i].set_xlabel('Epoch', fontsize=SMALL_SIZE)
        axs[i].legend(list(values.keys()), loc="lower left", fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    plt.savefig(f"{path}/train_metrics.png", dpi=120)
    plt.close()


def pathway_graph_to_seq_bfs(graph, *args, **kwargs):
    use_spectra = kwargs.get("use_spectra", False)
    root_frags = kwargs.get("root_frags", None)
    if root_frags is not None:
        root_frags = orjson.loads(root_frags)
    if isinstance(graph, (str, bytes)):
        graph = orjson.loads(graph)
    if isinstance(graph, dict):
        graph = nx.node_link_graph(graph)
    graph = graph.copy()
    root = [n for n in graph.nodes if graph.nodes[n]["depth"] == 0 or graph.nodes[n].get("root", False)][0]

    def get_spec_type():
        if use_spectra == "energy":
            return defaultdict(int)
        elif use_spectra == "binned_mass":
            return list()
        elif use_spectra == "frag":
            return defaultdict(lambda: {"int": 0, "frags": set()})
        elif use_spectra == "root_frags":
            return set()
        else:
            return None
    spectra_data = {"energy0": get_spec_type(), "energy1": get_spec_type(), "energy2": get_spec_type()}
    energy_keys = sorted(spectra_data.keys(), key=lambda x: int(x[-1]))

    def update_spec_data(n):
        if use_spectra == "energy":
            for energy in energy_keys:
                s_data = graph.nodes[n]['spectra']['energies'][energy]
                for j in range(len(s_data["mz"])):
                    mz = round(s_data["mz"][j], 2)
                    spectra_data[energy][mz] += s_data["int"][j]
        elif use_spectra == "binned_mass":
            s_data = graph.nodes[n]['spectra']['energies']['energy0']
            for j in range(len(s_data["mz"])):
                spectra_data['energy0'].append((s_data["mz"][j], s_data["int"][j]))
        elif use_spectra == "frag":
            s_data = graph.nodes[n]['spectra']['energies']['energy0']
            f_data = graph.nodes[n]['spectra']['structs']
            f_data = {f_id: f for f_id, f in zip(f_data['frag_id'], f_data["frag_struct"])}
            for f_id, intensity, mz in zip(s_data["frag_id"], s_data["int"], s_data["mz"]):
                mz = round(mz, 4)
                spectra_data["energy0"][mz]["frags"].add(f_data[f_id])
                spectra_data["energy0"][mz]["int"] += intensity

    if use_spectra and graph.nodes[root]['spectra']:
        update_spec_data(root)

    node_to_i = {root: 0}
    node_data = ["<sepc>"] + regex_tokenizer(root) + [f"IDX_{node_to_i[root]}"]

    edge_data = []
    for p_node, c_node in nx.bfs_edges(graph, root):
        edge_data.append(['<sepg>', f"IDX_{node_to_i.setdefault(p_node, max(node_to_i.values()) + 1)}",
                          f"IDX_{node_to_i.setdefault(c_node, max(node_to_i.values()) + 1)}"])
        node_data.append("<sepc>")
        node_data.extend(regex_tokenizer(c_node))
        node_data.append(f"IDX_{node_to_i[c_node]}")
        if use_spectra and graph.nodes[c_node]['spectra']:
            update_spec_data(c_node)
    edge_data = sum(edge_data, [])
    node_data[0] = "<boc>"
    edge_data[0] = "<bog>"
    whole_seq = node_data + ["<eoc>"] + edge_data + ["<eog>"]
    if use_spectra:
        if use_spectra == "energy":
            spectra_data = {
                energy: {
                    mz: (intensity / sum(level.values()) if sum(level.values()) > 0 else 0.0)
                    for mz, intensity in level.items()
                }
                for energy, level in spectra_data.items()
            }
            spectra_data = [
                (min(mz / 800.0, 1.0), intensity, int(energy[-1]) / len(energy_keys))
                for energy, level in spectra_data.items()
                for mz, intensity in sorted(level.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            spectra_data = sorted(spectra_data, key=lambda x: x[0], reverse=True)
            return " ".join(whole_seq), spectra_data
        elif use_spectra == "binned_mass":
            spectra_data = sorted(spectra_data['energy0'], key=lambda x: x[1], reverse=True)
            top_spec_data = set()
            for mz, intensity in spectra_data:
                if mz > kwargs.get("max_mz", 800):
                    mz = 800
                mz = int(mz * (10 ** kwargs.get("mz_rounding", 1)))
                top_spec_data.add(mz)
                if len(top_spec_data) >= kwargs.get("num_mz_peaks", 100):
                    break
            return " ".join(whole_seq), top_spec_data
        elif use_spectra == "frag":
            all_frags = spectra_data["energy0"]
            peaks = sorted(all_frags, key=lambda x: all_frags[x]["int"], reverse=True)
            frag_smiles = set()
            for p in peaks:
                frag_smiles.update(all_frags[p]["frags"])
                if len(frag_smiles) >= 10:
                    break
            whole_seq = regex_tokenizer(".".join(frag_smiles)) + whole_seq
            return " ".join(whole_seq)
        elif use_spectra == "root_frags" and root_frags is not None:
            frags = set()
            for i in range(10):
                for energy in energy_keys:
                    try:
                        frags.add(root_frags[energy][i])
                    except (IndexError, KeyError):
                        continue
                if len(frags) >= 6:
                    break
            whole_seq = regex_tokenizer(".".join(frags)) + whole_seq
            return " ".join(whole_seq)
    return " ".join(whole_seq)


def get_workers(args) -> int:
    if args.debug:
        return 1
    return args.workers


def seq_to_pathway_graph(seq, seq_probs=None, root_smiles=None, length_norm=None):
    if isinstance(seq, str):
        seq = seq.split(" ")
    if isinstance(seq_probs, torch.Tensor):
        seq_probs = seq_probs.cpu().tolist()
    elif isinstance(seq_probs, np.ndarray):
        seq_probs = seq_probs.tolist()

    # Formatting tokens to ignore
    fmt_tokens = {"<boc>", "<bog>", "<eog>", "<eoc>", "<sepc>", "<sepg>", "[nop]", "<sec>", "<eec>", "<sepec>"}

    graph = nx.DiGraph()
    invalid_smiles = 0
    node_buffer = []
    ec_buffer = []
    node_prob_indices = []
    nodes = {}
    idx_to_ec = {}
    latest_idx = None

    inside_boc = False
    inside_ec = False
    for i, tok in enumerate(seq):
        if tok == "<boc>":
            inside_boc = True
            continue
        if tok == "<eoc>":
            break
        if not inside_boc:
            continue
        if tok == "<sec>":
            inside_ec = True
            continue
        if tok == "<eec>":
            inside_ec = False
            ec_buffer = "".join(ec_buffer)
            idx_to_ec[latest_idx] = ec_buffer.split("<sepec>")
            ec_buffer = []
            continue
        if tok == "<sepec>":
            ec_buffer.append("<sepec>")
            continue
        if inside_ec:
            ec_buffer.append(tok)
            continue

        # Check if the token is an IDX_x indicating end of this SMILES
        if tok.startswith("IDX_"):
            latest_idx = tok
            node_id = tok  # e.g. "IDX_0"
            smiles_tokens = node_buffer
            prob_indices = node_prob_indices
            smiles = "".join(smiles_tokens)
            smiles = canon_smile_rdkit(smiles, remove_stereochemistry=True)
            if seq_probs is not None:
                probs = sum(seq_probs[i] for i in prob_indices)
                probs /= pow(5 + len(prob_indices), -0.5) / pow(6, -0.5)
            else:
                probs = None
            if smiles is not None:
                nodes[node_id] = (smiles, probs)
            else:
                invalid_smiles += 1
            node_buffer = []
            node_prob_indices = []
            continue
        # skip formatting tokens
        if tok in fmt_tokens:
            continue
        # normal SMILES token
        node_buffer.append(tok)
        node_prob_indices.append(i)

    # Add nodes to graph
    for idx, (smiles, log_p) in nodes.items():
        graph.add_node(smiles, probability=log_p, ec_num=idx_to_ec.get(idx, None))

    inside_bog = False
    current_edge_tokens = []
    for tok in seq:
        if tok == "<bog>":
            inside_bog = True
            continue
        if not inside_bog:
            continue
        if tok == "<sepg>" or tok == "<eog>":   # end of one edge group
            if len(current_edge_tokens) == 2:
                src_idx, dst_idx = current_edge_tokens
                if src_idx in nodes and dst_idx in nodes:
                    src_smiles = nodes[src_idx][0]
                    dst_smiles = nodes[dst_idx][0]
                    graph.add_edge(src_smiles, dst_smiles)
            # reset for next edge pair
            current_edge_tokens = []
            continue
        if tok == "<eog>":
            break
        # Skip formatting tokens
        if tok in fmt_tokens:
            continue
        if tok.startswith("IDX_"):
            current_edge_tokens.append(tok)

    if root_smiles is not None:
        root_smiles = canon_smile_rdkit(root_smiles, remove_stereochemistry=True)
        if root_smiles is None:
            print("Warning root couldn't be canonicalised, depth will not be added")
        else:
            graph.add_node(root_smiles, probability=0.0, root=True)
            depths = nx.shortest_path_length(graph, source=root_smiles)
            for node in list(graph.nodes):
                if node in depths:
                    graph.nodes[node]["depth"] = depths[node]
                else:
                    graph.remove_node(node)

    return graph, invalid_smiles


def get_pathformer_version(args, config):
    start = f"{args.data_name}_{config['num_layers']}_{config['d_model']}_{config['n_heads']}_{args.split_type}_"
    if args.freeze is not None:
        start += f"{args.freeze}_"
    if args.enzymes:
        start += f"ec_{args.enzymes}_"
    if config.get("use_spectra", False):
        start += f"spec_{config['use_spectra']}_"
    end = f"{args.seed}"
    return start + end


def regex_tokenizer(smile: str) -> list:
    """Tokenizes a SMILES string using a regular expression.

    :param smile: The input SMILES string.
    :return: A list of tokens extracted from the SMILES string.
    """
    # pattern = r"(\[|\]|[A-Z][a-z]?|[A-Z]|[a-z]|:[0-9]{3}|:[0-9]{2}|:[0-9]|[0-9]|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@@|@|\?|>>|>|\*|\$|\%)"
    pattern = r"(\[|\]|[A-Z][a-z]?|[A-Z]|[a-z]|:[0-9]{3}|:[0-9]{2}|:[0-9]|[0-9]|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@@|@|\?|>>|>|\*|\$|\%)"

    regex = re.compile(pattern)
    tokens = regex.findall(smile)
    assert smile.strip() == ''.join(tokens), f"Regex SMILE is not the same as original SMILE\nOriginal:{smile}\nRegex:   {''.join(tokens)}"
    return tokens


def get_rows_from_ids(db_conn, split_name, split_id, split_method="compound"):
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split_name: {split_name}")

    split_id = int(split_id)
    if split_id < 0:
        raise ValueError(f"Invalid split_id: {split_id}")

    count = db_conn.execute(
        """
        SELECT COUNT(*)
        FROM pathway_splits s
        WHERE s.split_method = ?
          AND s.split_id = ?
          AND s.split_name = ?
        """,
        (split_method, split_id, split_name),
    ).fetchone()[0]

    rows = db_conn.execute(
        """
        SELECT p.graph_json, p.root_fragments
        FROM pathways p
        JOIN pathway_splits s
          ON s.pathway_id = p.id
        WHERE s.split_method = ?
          AND s.split_id = ?
          AND s.split_name = ?
        ORDER BY p.id
        """,
        (split_method, split_id, split_name),
    )

    return rows, count


def load_pathway_dataset_sql(data_name):
    db_conn = sqlite3.connect(f"data/{data_name}_nx_pathways.sqlite")
    return db_conn


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, size))
        if not batch:
            return
        yield batch


def graph_from_envipath(data):
    graph = nx.DiGraph(name=data.get('name', ""))
    nodes = data["nodes"]
    co2 = {"O=C=O", "C(=O)=O"}
    for link in data["links"]:
        source = nodes[link["source"]]
        target = nodes[link["target"]]
        if source["smiles"] not in graph:
            graph.add_node(source["smiles"], depth=source["depth"])
        else:
            graph.nodes[source["smiles"]]["depth"] = min(source["depth"], graph.nodes[source["smiles"]]["depth"])
        if target["smiles"] not in graph and target["smiles"] not in co2:
            graph.add_node(target["smiles"], depth=target["depth"])
        elif target["smiles"] not in co2:
            graph.nodes[target["smiles"]]["depth"] = min(target["depth"], graph.nodes[target["smiles"]]["depth"])
        if target["smiles"] not in co2 and target["smiles"] != source["smiles"]:
            graph.add_edge(source["smiles"], target["smiles"])
    return graph


def prune_graph(graph, threshold):
    """
    Prunes nodes in the graph based on their precomputed log probabilities.
    Nodes below the threshold are removed.
    """
    remove_graph_cycles(graph)
    root_node = [n for n in graph.nodes if "root" in graph.nodes[n] and graph.nodes[n]["root"]][0]
    for node in list(graph.nodes):
        # Check if the node's log probability meets the threshold
        if graph.nodes[node]["probability"] < threshold and node != root_node:  # / max(graph.nodes[node]["depth"], 1):
            graph.remove_node(node)
    # Remove nodes that now can't be reached from the root
    descendants = nx.descendants(graph, root_node)
    descendants.add(root_node)
    for node in list(graph.nodes):
        if node not in descendants:
            graph.remove_node(node)


def remove_graph_cycles(graph):
    while True:
        try:
            cycle = nx.find_cycle(graph)
            graph.remove_edge(*cycle[-1])  # Remove the last edge in the cycle
        except nx.NetworkXNoCycle:
            break


def get_shortest_path(pathway, in_start_node, in_end_node):
    try:
        pred = nx.shortest_path(pathway, source=in_start_node, target=in_end_node)
    except nx.NetworkXNoPath:
        return []
    pred.remove(in_start_node)
    pred.remove(in_end_node)
    return pred


def set_pathway_eval_weight(pathway):
    # This method intends to replace the weight assignment functionality in compare_pathways
    # So that it can work with merge_pathways
    # This method intends to be applied to raw pathways before they are merged

    node_eval_weights = {}

    for node in pathway.nodes:
        # Scale score according to depth level
        node_eval_weights[node] = 1 / (2 ** pathway.nodes[node]["depth"]) if pathway.nodes[node]["depth"] >= 0 else 0

    return node_eval_weights


def get_depth_adjusted_pathway(pathway, intermediates):
    if len(intermediates) < 1:
        return pathway
    root_nodes = pathway.graph["root_nodes"]

    for node in pathway.nodes:
        if node in root_nodes:
            continue

        if node in intermediates:
            pathway.nodes[node]["depth"] = -99
        else:
            shortest_path_list = []

            for root_node in root_nodes:
                shortest_path_nodes = get_shortest_path(pathway, root_node, node)
                if shortest_path_nodes:
                    shortest_path_list.append(shortest_path_nodes)

            if shortest_path_list:
                shortest_path_nodes = min(shortest_path_list, key=len)
                num_ints = sum(1 for shortest_path_node in shortest_path_nodes if
                               shortest_path_node in intermediates)
                pathway.nodes[node]["depth"] -= num_ints

    return pathway


def assign_next_depth(pathway, current_depth):
    new_assigned_nodes = False
    current_depth_nodes = {n for n in pathway.nodes if pathway.nodes[n]["depth"] == current_depth}
    for node in current_depth_nodes:
        successors = pathway.successors(node)
        for s in successors:
            if pathway.nodes[s]["depth"] < 0:
                pathway.nodes[s]["depth"] = current_depth + 1
                new_assigned_nodes = True
    return new_assigned_nodes


def get_pathway_with_depth(pathway):
    """Recalculates depths in the pathway.
    Can fix incorrect depths from json parse if there were multiple nodes with the same SMILES at
    different depths that got merged."""
    current_depth = 0
    for node in pathway.nodes:
        if node in pathway.graph["root_nodes"]:
            pathway.nodes[node]["depth"] = current_depth
        else:
            pathway.nodes[node]["depth"] = -99

    while assign_next_depth(pathway, current_depth):
        current_depth += 1
    return pathway


def initialise_pathway(pathway):
    pathway.graph["root_nodes"] = {n for n in pathway.nodes if pathway.nodes[n]["depth"] == 0}
    pathway = get_pathway_with_depth(pathway)
    return pathway


def find_intermediates(data_pathway, pred_pathway):
    # data_pathway_nodes = data_pathway.nodes
    common_nodes = get_common_nodes(pred_pathway, data_pathway)
    intermediates = set()
    for node in common_nodes:
        down_stream_nodes = data_pathway.successors(node)
        for down_stream_node in down_stream_nodes:
            if down_stream_node in pred_pathway:
                all_ints = get_shortest_path(pred_pathway, node, down_stream_node)
                # intermediates.update(i for i in all_ints if i not in data_pathway_nodes)
                intermediates.update(all_ints)
    return intermediates


def get_common_nodes(pred_pathway, data_pathway):
    common_nodes = set()

    for node in data_pathway.nodes:
        is_pathway_root_node = node in data_pathway.graph["root_nodes"]
        is_this_root_node = node in pred_pathway.graph["root_nodes"]
        if node in pred_pathway.nodes:
            if is_pathway_root_node is False and is_this_root_node is False:
                common_nodes.add(node)
            elif is_pathway_root_node and is_this_root_node:
                common_nodes.add(node)
    return common_nodes


def compare_pathways(data_pathway, pred_pathway, weighted=True):
    """Compare two pathways for multi-gen evaluation.
    It is assumed the smiles in both pathways have been standardised in the same manner."""
    data_pathway = initialise_pathway(data_pathway)
    pred_pathway = initialise_pathway(pred_pathway)
    start_and_end_nodes = find_intermediates(data_pathway, pred_pathway)
    intermediates = {n for n in start_and_end_nodes}

    if intermediates:
        pred_pathway = get_depth_adjusted_pathway(pred_pathway, intermediates)

    test_pathway_eval_weights = set_pathway_eval_weight(data_pathway) if weighted else {}
    pred_pathway_eval_weights = set_pathway_eval_weight(pred_pathway) if weighted else {}

    common_nodes = get_common_nodes(pred_pathway, data_pathway)

    data_only_nodes = set(n for n in data_pathway.nodes if n not in common_nodes)
    pred_only_nodes = set(n for n in pred_pathway.nodes if n not in common_nodes)

    score_TP = 0.0
    score_FP = 0.0
    score_FN = 0.0
    final_score = 0.0
    precision = 0.0
    recall = 0.0

    # If weighted == False, then the weights dicts will not contain nodes and 1 is always used
    for node in common_nodes:
        if pred_pathway.nodes[node]["depth"] > 0:
            score_TP += test_pathway_eval_weights.get(node, 1)
    for node in data_only_nodes:
        if data_pathway.nodes[node]["depth"] > 0:
            score_FN += test_pathway_eval_weights.get(node, 1)
    for node in pred_only_nodes:
        if pred_pathway.nodes[node]["depth"] > 0:
            score_FP += pred_pathway_eval_weights.get(node, 1)

    if (score_TP + score_FP + score_FN) > 0:
        final_score = score_TP / (score_TP + score_FP + score_FN)
    if (score_TP + score_FP) > 0:
        precision = score_TP / (score_TP + score_FP)
    if (score_TP + score_FN) > 0:
        recall = score_TP / (score_TP + score_FN)
    data_pathway.graph["root_nodes"] = list(data_pathway.graph["root_nodes"])
    pred_pathway.graph["root_nodes"] = list(pred_pathway.graph["root_nodes"])
    return (final_score, precision, recall), intermediates


def pathway_analysis(test_output, threshold):
    scores = []
    for i, pathway in enumerate(test_output["predictions"]):
        true, predicted_graph_dict = pathway[0], pathway[1]
        # Convert the predicted tree to a NetworkX directed graph
        predicted_graph = nx.node_link_graph(predicted_graph_dict)
        try:
            true_graph = nx.node_link_graph(true)
        except KeyError:
            true_graph = graph_from_envipath(true)

        # Prune nodes below the threshold
        prune_graph(predicted_graph, threshold)
        remove_graph_cycles(true_graph)
        p_way_score, intermediates = compare_pathways(true_graph, predicted_graph, weighted=True)
        scores.append(p_way_score)

    return threshold, scores


def process_compare_pathways(args, test_output, thresholds, save_path=None, plot_curve=True):
    if save_path is None:
        save_path = f"results/{args.model_name}/{args.data_name}_{args.preprocessor}"
    # Process the predicted pathway tree with varying thresholds
    for i in range(len(test_output["predictions"])):
        for j, graph in enumerate(test_output["predictions"][i]):
            if isinstance(graph, nx.Graph):
                test_output["predictions"][i][j] = nx.node_link_data(graph)
    func_list = [delayed(pathway_analysis)(test_output, threshold)
                 for threshold in thresholds]
    results = Parallel(n_jobs=min(get_workers(args), len(thresholds)), verbose=2)(func_list)
    for threshold, result in results:
        thresholds[threshold] = result
    precision = {}
    recall = {}
    for threshold in thresholds:
        precision_sum = []
        recall_sum = []
        for result in thresholds[threshold]:
            precision_sum.append(result[1])
            recall_sum.append(result[2])
        precision_sum = sum(precision_sum) / len(precision_sum)
        recall_sum = sum(recall_sum) / len(recall_sum)
        precision[threshold] = precision_sum
        recall[threshold] = recall_sum

    if plot_curve:
        sorted_recall, sorted_precision, area_under_curve = sort_recall_precision(recall, precision)
        plot_pr_curve(sorted_recall, sorted_precision, area_under_curve, f"{args.data_name} MultiGen",
                      save_path, "multigen")
    return recall, precision


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str, help="Valid models include: TransformerModel")
    parser.add_argument("data_name", type=str, help="Which dataset to use, uspto, pubchem, envipath, baeyer")
    parser.add_argument("--tokenizer", type=str, default="regex", help="Style of tokenizer, selfies, regex")
    parser.add_argument("--split-path", type=str, default="", help="Predetermined split file with train, val and test SMILES")
    parser.add_argument("--max-len", type=int, default=1000, help="Maximum encoded length to consider")
    parser.add_argument("--min-len", type=int, default=0, help="Minimum encoded length to consider")
    parser.add_argument("--augment-count", type=int, default=0, help="How much SMILES augmentation to do, <=0 disables")
    parser.add_argument("--debug", action="store_true", default=False, help="Whether to set parameters for debugging")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size to target")
    parser.add_argument("--weights-dataset", type=str, default="", help="Pretrained weights based on what dataset")
    parser.add_argument("--test-mapping", action="store_true",
                        help="Whether to remove atom mapping before calculating accuracy")
    parser.add_argument("--score-all", action="store_true", help="Whether to group same reactants together")
    parser.add_argument("--folds", type=int, default=10, help="How many folds to use for cross validation, default 10")
    parser.add_argument("--preprocessor", type=str, default="rdkit",
                        help="Type of preprocessing to use, envipath or rdkit")
    parser.add_argument("--scenario", action="store_true", help="Whether to also use scenario data")
    parser.add_argument("--enzymes", action="store_true", help="Whether to also use enzyme data")
    parser.add_argument("--pathway-train", action="store_true", help="Whether to train with whole pathway as input")
    parser.add_argument("--seed", default=1, type=int, help="Seed to use for randomness")
    parser.add_argument("--test", default="", type=str, help="Dataset to use for testing instead of training")
    parser.add_argument("--no-train", action="store_true", help="Do not train, intended to be used with --weights-dataset")
    parser.add_argument("-tm", "--train-many", action="store_true", help="Whether to train many models")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expand", action="store_true", help="Whether to use data expanded with different standardisation")
    parser.add_argument("--freeze", type=int, default=None, help="Layers to freeze used with a slice")
    parser.add_argument("-sf", "--spec-filter", type=float, default=None, help="Whether to filter predicted pathways with spectra data")
    parser.add_argument("-st", "--split-type", type=str, default="roots", help="How to split the data")
    arguments = parser.parse_args()
    return arguments


def expand_search_space(search_space_dict):
    keys = list(search_space_dict.keys())
    values = list(search_space_dict.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def get_pathformer_thresholds(value_type=list):
    with open("data/pathformer_thresholds_bin.txt") as t_file:
        thresholds = t_file.readline().split(" ")
    thresholds.append(-1000000)
    return {float(t): value_type() for t in thresholds}


def sort_recall_precision(recall, precision, return_order=False):
    sorted_recall = sorted(recall.items(), key=lambda x: x[1])
    order = [r[0] for r in sorted_recall]
    sorted_recall = np.array([r[1] for r in sorted_recall])
    sorted_precision = np.array([precision[key] for key in order])
    area_under_curve = auc(sorted_recall, sorted_precision)
    if return_order:
        return sorted_recall, sorted_precision, area_under_curve, order
    return sorted_recall, sorted_precision, area_under_curve


def plot_pr_curve(recall, precision, area, data_name, path, file_name=None, extra_pr=None, plot_title=None, **kwargs):
    fig_size = (7, 7)

    # Plot the curve with recall on the x-axis and precision on the y-axis
    plt.figure(figsize=fig_size)
    if kwargs.get("usetex", False):
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = "Computer Modern Roman"
    plt.style.use('seaborn-v0_8-dark-palette')
    if len(recall) > 0:
        plt.plot(recall, precision, label=f'{data_name} (AUC: {area:.4f})')
    colours = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#D55E00", "#0072B2"]
    if extra_pr is not None:
        error = kwargs.get("error", None)
        for i, pr in enumerate(extra_pr):
            plt.plot(pr[0], pr[1], label=pr[2], color=colours[i])
            if error is not None:
                rec_err, prec_err = error[i]

                lower = np.clip(pr[1] - prec_err, 0, 1)
                upper = np.clip(pr[1] + prec_err, 0, 1)

                plt.fill_between(
                    pr[0],
                    lower,
                    upper,
                    alpha=0.25,
                    facecolor=colours[i]
                )
    plt.xlim(kwargs.get("xmin", 0), kwargs.get("xlim", 1))
    plt.ylim(kwargs.get("ymin", 0), kwargs.get("ylim", 1))

    # Add labels and title
    plt.xlabel('Recall', fontsize=24)
    plt.xticks(fontsize=24)
    plt.ylabel('Precision', fontsize=24)
    plt.yticks(fontsize=24)

    # Show the legend
    plt.legend(fontsize=22, loc=kwargs.get("legend_loc", "upper right"), framealpha=0.5)
    plt.tight_layout()

    # Show the plot or save it
    if ".png" in path or ".pdf" in path:
        plt.savefig(path, dpi=120)
    else:
        for i in range(1, 100):
            save_loc = f"{path}/precision_recall_{i if file_name is None else file_name}.png"
            if not os.path.exists(save_loc) or file_name is not None:
                plt.savefig(save_loc, dpi=120)
                break
    plt.clf()
    plt.close()
    return


def _compute_pathway_groups(pathways):
    roots = []
    roots_set = set()
    root_to_pathways = {}
    (total_rows,) = pathways.execute(
        "SELECT COUNT(*) FROM pathways"
    ).fetchone()
    for p_way in tqdm(pathways.execute("SELECT graph_json, id FROM pathways ORDER BY id"), total=total_rows, desc="Loading rows from database"):
        p_way, p_id = p_way
        p_way = orjson.loads(p_way)
        p_rs = [n['id'] for n in p_way['nodes'] if n.get("root", False)]
        if len(p_rs) != 1:
            raise ValueError()
        p_rs = p_rs[0]
        if p_rs not in roots_set:
            roots_set.add(p_rs)
            roots.append(p_rs)
        root_to_pathways.setdefault(p_rs, []).append(p_id)
    splitter = KFold(n_splits=10, shuffle=True, random_state=0)
    roots = pd.Series(roots)
    splits = {}

    def get_pathway_ids(root_ids):
        pathway_ids = []
        for root in roots.iloc[root_ids]:
            pathway_ids.extend(root_to_pathways[root])
        return pathway_ids

    for i, (train_val_ids, test_ids) in enumerate(splitter.split(roots)):
        train_ids, val_ids = train_test_split(train_val_ids, test_size=len(test_ids), random_state=0)
        # Minus one to make the ids zero based, they get converted back to one based later
        train_ids = get_pathway_ids(train_ids - 1)
        val_ids = get_pathway_ids(val_ids - 1)
        test_ids = get_pathway_ids(test_ids - 1)
        splits[i] = {"train": train_ids, "val": val_ids, "test": test_ids}
    return splits


def _write_split_table(db_conn, all_splits, split_method):
    """
    all_splits: dict[int, dict[str, list[int]]]

    split_dict example:
    {
        "train": [0, 2, 5],
        "val":   [1, 4],
        "test":  [3]
    }
    """
    cur = db_conn.cursor()

    cur.execute("BEGIN;")

    split_ids = [split_id for split_id, _ in all_splits.items()]
    cur.executemany(
        """
        DELETE FROM pathway_splits
        WHERE split_method = ?
          AND split_id = ?;
        """,
        [(split_method, sid) for sid in split_ids],
    )

    rows = []

    for split_id, split_dict in all_splits.items():
        for split_name, indices in split_dict.items():
            for idx in indices:
                rows.append((int(idx) + 1, split_method, split_id, split_name))

    cur.executemany(
        """
        INSERT INTO pathway_splits (
            pathway_id,
            split_method,
            split_id,
            split_name
        )
        VALUES (?, ?, ?, ?);
        """,
        rows,
    )

    db_conn.commit()


def pathformer_split_disjoint(args, force_re_split=False):
    """
    Create 10 disjoint splits and store them in the SQLite pathways table.
    Split strategies are stored independently, so compound and roots splits
    can coexist for the same fold ids.
    """
    if hasattr(args, "test") and args.test:
        data_name = args.test
    else:
        data_name = args.data_name

    db_conn = load_pathway_dataset_sql(data_name)

    split_id = int(args.seed)
    split_method = args.split_type

    cur = db_conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pathway_splits (
            pathway_id   INTEGER NOT NULL,
            split_method TEXT NOT NULL,
            split_id     INTEGER NOT NULL,
            split_name   TEXT NOT NULL,
            PRIMARY KEY (pathway_id, split_method, split_id)
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pathway_splits_lookup
        ON pathway_splits (split_method, split_id, split_name, pathway_id);
    """)
    db_conn.commit()

    existing_count = db_conn.execute(
        """
        SELECT COUNT(*)
        FROM pathway_splits
        WHERE split_method = ?
          AND split_id = ?
        """,
        (split_method, split_id),
    ).fetchone()[0]

    if existing_count > 0 and not force_re_split:
        db_conn.close()
        print(f"{split_method} split already exists")
        return

    print("Adding split to database")
    if args.split_type == "roots":
        splits = _compute_pathway_groups(db_conn)
    else:
        raise ValueError(f"Invalid split_type: {args.split_type}")
    print('Writing splits to database')
    _write_split_table(db_conn, splits, split_method)
    db_conn.close()
    return
