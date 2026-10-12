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

from torch.optim import AdamW
from torch import no_grad, ne, eq, arange, transpose, zeros
from torch.nn import (TransformerEncoderLayer, TransformerEncoder, Linear, CrossEntropyLoss, LogSoftmax, Embedding,
                      LayerNorm, ModuleList)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils.utils import *
from models.BaseModel import BaseModel
from utils.attention import KVCacheDecoderLayer


class PathFormer(BaseModel):
    def __init__(self, config: dict, tokenizer: tuple[dict, dict], args, **kwargs):
        super().__init__(args)
        if "model_size" in config:
            size = config["model_size"]
            if size == "small":
                config["num_layers"] = 6
                config["n_heads"] = 6
                config["d_model"] = 384
            elif size == "medium":
                config["num_layers"] = 12
                config["n_heads"] = 12
                config["d_model"] = 768
            elif size == "large":
                config["num_layers"] = 24
                config["n_heads"] = 16
                config["d_model"] = 1024
            else:
                raise ValueError(f"Unknown model size {size}")
        self.model_config: dict = config
        self.save_path: str = kwargs.get("save_path", f"results/BinnedPathFormer/{get_pathformer_version(args, config)}")
        os.makedirs(self.save_path, exist_ok=True)
        if config.get("seq_method", "bfs") == "bfs":
            self.seq_method = pathway_graph_to_seq_bfs
        else:
            raise ValueError(f"Unknown seq_method {config['seq_method']}")
        self.use_spectra = self.model_config.get('use_spectra', False)
        self.lr_mult: float = config["lr"]
        self.lr = 0
        self.num_beams = config.get("num_beams", 200)
        if "uspto" in args.data_name:
            print("Using 25 beams for USPTO data due to size")
            self.num_beams = max(25, self.num_beams)
        self.num_layers: int = config["num_layers"]
        self.d_model: int = config["d_model"]
        self.n_heads: int = config["n_heads"]
        self.max_len: int = config["max_len"]
        self.dropout: float = config["dropout"]
        self.feedforward: int = config["feedforward"]
        self.warm_up_steps: int = config["warm_up"]
        self.mz_bins = self.model_config["max_mz"] * (10 ** self.model_config["mz_rounding"])
        self.char_to_i, self.i_to_char = tokenizer
        self.pad_id: int = self.char_to_i["[nop]"]
        self.mz_pad_id: int = self.mz_bins + 1
        self.idx0_id: int = self.char_to_i["IDX_0"]
        self.start_id: int = self.char_to_i["<boc>"]
        self.end_id: int = self.char_to_i["<eog>"]
        self.mz_embedding = Embedding(self.mz_bins + 2, self.d_model, padding_idx=self.mz_pad_id)
        self.pre_mz_norm = LayerNorm(self.d_model)
        self.post_mz_norm = LayerNorm(self.d_model)
        encoder_layer = TransformerEncoderLayer(self.d_model, self.n_heads, batch_first=True,
                                                dim_feedforward=self.feedforward, dropout=self.dropout,
                                                activation="gelu", norm_first=True)
        self.mz_encoder = TransformerEncoder(encoder_layer, self.num_layers, norm=self.post_mz_norm)
        self.embedding = Embedding(len(self.char_to_i), self.d_model, padding_idx=self.pad_id)
        self.pos_embedding = PathFormerPositionalEncoding(self.d_model, max_len=self.max_len)
        self.pre_dec_norm = LayerNorm(self.d_model)
        self.post_dec_norm = LayerNorm(self.d_model)
        self.decoder_layers = ModuleList([
            KVCacheDecoderLayer(self.d_model, self.n_heads, self.feedforward, self.dropout)
            for _ in range(self.num_layers)
        ])
        self.linear = Linear(self.d_model, len(self.char_to_i))
        self.loss_fn = CrossEntropyLoss(ignore_index=self.pad_id, reduction='none')
        self.softmax = LogSoftmax(dim=-1)
        self.model_outputs = {"train": [], "val": [], "test": []}
        if os.path.exists(metric_path := os.path.join(self.save_path, "train_metrics.json")):
            with open(metric_path) as m_file:
                self.metric_history = json.load(m_file)
        else:
            self.metric_history = {"loss": {"train": [], "val": []}, "char_acc": {"train": [], "val": []},
                                   "seq_acc": {"train": [], "val": []}}
        self.final_test_metrics = {}
        with open(os.path.join(self.save_path, "config.json"), "w") as c_file:
            json.dump(self.model_config, c_file, indent=4)
        self._init_params()

    def on_fit_start(self) -> None:
        if self.args.freeze is not None:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.batch_norm.parameters():
                p.requires_grad = False
            for p in self.pos_embedding.parameters():
                p.requires_grad = False
        if isinstance(self.args.freeze, int):
            if self.args.freeze == -1:
                for layer in self.transformer.layers:
                    for p in layer.parameters():
                        p.requires_grad = False
            else:
                for layer in self.transformer.layers[:self.args.freeze + 1]:
                    for p in layer.parameters():
                        p.requires_grad = False

    def encode_data(self):
        db_conn = load_pathway_dataset_sql(self.args.data_name)
        train = self.encode_graphs(db_conn, "train")
        val = self.encode_graphs(db_conn, "val")
        test = self.encode_graphs(db_conn, "test")
        db_conn.close()
        train, val, test = (BigDataset(*train, pad_id=[self.pad_id, self.mz_pad_id]),
                            BigDataset(*val, pad_id=[self.pad_id, self.mz_pad_id]),
                            BigDataset(*test, pad_id=[self.pad_id, self.mz_pad_id]))
        train_loader = DataLoader(train, batch_size=self.args.batch_size, shuffle=True, num_workers=4,
                                  persistent_workers=True, collate_fn=train.collate_batch, pin_memory=True)
        val_loader = DataLoader(val, batch_size=self.args.batch_size, shuffle=False, num_workers=4,
                                persistent_workers=True, collate_fn=val.collate_batch, pin_memory=True)
        test_loader = DataLoader(test, batch_size=self.args.batch_size // 4, shuffle=False, num_workers=4,
                                 persistent_workers=True, collate_fn=test.collate_batch, pin_memory=True)
        return train_loader, val_loader, test_loader

    def encode_graphs(self, db_conn, split_type):
        rows, count = get_rows_from_ids(db_conn, split_type, self.args.seed, self.args.split_type)
        batch_size = 20000 if not self.args.debug else 100
        total_batches = (count + batch_size - 1) // batch_size
        all_graphs = []
        all_spectra = []
        reached_max = False
        for batch in tqdm(chunked(rows, batch_size), total=total_batches, desc=f"Encoding {split_type}"):
            spectra = None
            enc_seqs = Parallel(n_jobs=get_workers(self.args), batch_size=200)(
                delayed(pathway_graph_to_seq_bfs)(p[0], use_ec=self.args.enzymes, use_spectra=self.use_spectra,
                                                  max_mz=self.model_config["max_mz"],
                                                  num_mz_peaks=self.model_config["num_mz_peaks"],
                                                  mz_rounding=self.model_config["mz_rounding"]) for p in batch)
            if self.use_spectra == "energy" or self.use_spectra == "binned_mass":
                enc_seqs, spectra = zip(*enc_seqs)
            for seq in zip(*self.encode_seq(enc_seqs, spectra=spectra)):
                if self.use_spectra == "energy" or self.use_spectra == "binned_mass":
                    seq, spectra = seq
                    all_spectra.append(spectra)
                else:
                    seq = seq[0]
                all_graphs.append(seq)
                if split_type == "test" and "uspto" in self.args.data_name and len(all_graphs) > 2000:
                    print("Reached 2000 graphs in uspto test set, stopping early to reduce runtime")
                    reached_max = True
                    break
            if reached_max:
                break
            if self.args.debug:
                break
        if self.use_spectra == "energy" or self.use_spectra == "binned_mass":
            return all_graphs, all_spectra
        return all_graphs,

    def encode_data_kfold(self, smiles) -> list:
        raise NotImplementedError()

    def encode_seq(self, seqs, spectra=None):
        seqs = Parallel(n_jobs=get_workers(self.args), batch_size=1000)(
            delayed(encode_seq)(s, char_to_i=self.char_to_i, max_len=self.max_len) for s in seqs)
        filtered_seqs = []
        filtered_spectra = []
        for i, seq in enumerate(seqs):
            if seq is not None:
                if spectra:
                    filtered_spectra.append(spectra[i])
                filtered_seqs.append(seq)
        dtype = torch.uint16 if self.mz_bins + 2 < 65535 else torch.uint32
        filtered_spectra = [tensor(list(s), dtype=dtype) if len(s) > 0 else zeros(0, dtype=dtype) for s in filtered_spectra]
        return filtered_seqs, filtered_spectra

    def smiles_to_pathway(self, root_smiles: list | str, spectra, device='cuda'):
        # Ensure correct formatting for single smiles inputs and consistent spectra frags type
        if isinstance(root_smiles, str):
            root_smiles = [root_smiles]
            spectra = [spectra]
        num_beams = 2 if self.args.debug else self.num_beams
        batch_size = 5
        with no_grad():
            self.eval()
            self.to(device)
            joined_graphs = []
            # Encode sequences to tensors and add spectra fragments
            for i in range(0, len(root_smiles), batch_size):
                root_smiles_batch = root_smiles[i:i + batch_size]
                spectra_batch = spectra[i:i + batch_size]
                if not isinstance(spectra, Tensor):
                    spectra_batch = [tensor(list(s), dtype=torch.long, device=self.device) for s in spectra_batch]
                start_seqs = [["<boc>"] + regex_tokenizer(root_smiles_batch[i]) + ["IDX_0"] for i in range(len(root_smiles_batch))]
                start_seqs = [tensor([self.char_to_i[token] for token in start_seq], device=self.device, dtype=torch.long) for start_seq
                              in start_seqs]
                start_seqs = pad_sequence(start_seqs, batch_first=True, padding_value=self.pad_id)
                spectra_batch = pad_sequence(spectra_batch, batch_first=True, padding_value=self.mz_pad_id)
                _, idx0_pos = self.get_root_mask(start_seqs)
                # Predict and then decode pathway graphs from tensors
                pred_seqs, pred_lls, char_lls = beam_decode_pathformer(self, start_seqs, idx0_pos, num_beams=num_beams, spectra=spectra_batch)
                decoded_pred = [
                    [" ".join(char for idx in seq if (char := self.i_to_char[idx.item()]) != "[nop]") for seq in beams]
                    for beams in pred_seqs]
                decoded_pred = [
                    [seq_to_pathway_graph(seq, seq_probs=char_lls[i][j][:len(seq)], root_smiles=root_smiles_batch[i])
                     for
                     j, seq in enumerate(beams)] for i, beams in enumerate(decoded_pred)]
                # Combine beams into one graph per root smiles
                for predicted in decoded_pred:
                    p_graph = nx.DiGraph()
                    predicted = list(reversed(predicted))  # Puts higher probs at end of list to overwrite low probs
                    for j, (graph, _) in enumerate(predicted):
                        for n in graph:
                            graph.nodes[n]["probability"] *= len(predicted) - j  # Further downweight low probs
                        p_graph.update(graph)
                    joined_graphs.append(p_graph)
        return joined_graphs

    def embed(self, x: Tensor, positions: Tensor = None) -> Tensor:
        x = self.embedding(x) * (self.d_model ** 0.5)
        return self.pos_embedding(x, positions=positions)

    def encode(self, spectra):
        spec_mask = eq(spectra, self.mz_pad_id)
        z_spec = self.mz_embedding(spectra)
        z_spec = self.pre_mz_norm(z_spec)
        z_spec = self.mz_encoder(z_spec, src_key_padding_mask=spec_mask)
        return z_spec, spec_mask

    def decode(self, trg, mz_enc_out, mz_mask, test=False, kv_cache=None, positions=None):
        B, L_trg = trg.shape

        pad_mask = eq(trg, self.pad_id)

        # Cache the running padding mask for future single-token steps
        if kv_cache is not None and "pad_mask" not in kv_cache:
            B, current_len = pad_mask.size()

            # Initialise padding cache
            cache_pad_mask = torch.ones(
                (B, self.max_len),
                dtype=torch.bool,
                device=pad_mask.device
            )
            # Assign initial mask using a slice
            cache_pad_mask[:, :current_len] = pad_mask

            # Store both the mask cache and the current length
            prefix_allow, idx0_pos = self.get_root_mask(trg)
            kv_cache["pad_mask"] = cache_pad_mask
            kv_cache["seq_len"] = idx0_pos
        elif kv_cache is not None and "pad_mask" in kv_cache:
            cache_pad_mask = kv_cache["pad_mask"]
            B = cache_pad_mask.size(0)

            # Increment the sequence length by 1 and insert new pad mask into that position
            kv_cache["seq_len"] += 1
            insert_positions = kv_cache['seq_len']
            batch_idx = torch.arange(B, device=cache_pad_mask.device)
            incoming_len = pad_mask.size(1)
            for i in range(incoming_len):
                cache_pad_mask[batch_idx, insert_positions + i] = pad_mask[:, i]

            # Extract the valid portion of the cache for the current attention step
            # We slice up to the maximum active length currently in the batch
            max_active_len = (insert_positions + incoming_len).max().item()
            pad_mask = cache_pad_mask[:, :max_active_len]

        total_len = pad_mask.size(1)

        # Construct unified SDPA mask (combining structure and padding)
        if kv_cache is not None and len(kv_cache) > 0 and L_trg == 1:
            # Decode step: No causal structure needed for 1 token, but MUST mask past padding
            # Shape requirements for broadcast: (B, 1, 1, L_total)
            attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        else:
            # Prefill step: Need causal mask + prefix mask + padding mask
            prefix_allow, idx0_pos = self.get_root_mask(trg)
            idx0_pos_shifted = idx0_pos
            square_mask = torch.triu(torch.ones(total_len, total_len, device=trg.device), diagonal=1).bool()
            square_mask = square_mask.unsqueeze(0).expand(B, total_len, total_len)
            pos = arange(total_len, device=trg.device).unsqueeze(0)
            prefix_allow_full = pos <= idx0_pos_shifted.unsqueeze(1)
            prefix_columns = prefix_allow_full.unsqueeze(1).expand(B, total_len, total_len)

            # Combine causal structure (True = block) with padding (True = block)
            expanded_pad_mask = pad_mask.unsqueeze(1).expand(B, total_len, total_len)
            attn_mask = (square_mask & (~prefix_columns)) | expanded_pad_mask
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, L_total, L_total)

        # Expand mz_mask for cross-attention shape requirements if provided
        if mz_mask is not None:
            # mz_mask is (B, L_mem) -> (B, 1, 1, L_mem)
            mem_mask = mz_mask.unsqueeze(1).unsqueeze(2)
        else:
            mem_mask = None

        trg_emb = self.embed(trg, positions)
        trg_emb = self.pre_dec_norm(trg_emb)

        for i, layer in enumerate(self.decoder_layers):
            trg_emb = layer(trg_emb, mz_enc_out, tgt_mask=attn_mask, memory_mask=mem_mask, cache=kv_cache, layer_idx=i, max_len=self.max_len)

        trg_emb = self.post_dec_norm(trg_emb)
        output = self.linear(trg_emb).float()

        if test:
            output = self.softmax(output)
        return output

    def forward(self, trg, spectra=None, test=False, kv_cache=None) -> Tensor:
        # mz_encoder only needs to run during the prefill phase, skip if caching is active
        if kv_cache is not None and "mz_enc_out" in kv_cache:
            z_spec, spec_mask = kv_cache["mz_enc_out"]
        else:
            z_spec, spec_mask = self.encode(spectra)
            if kv_cache is not None:
                kv_cache["mz_enc_out"] = (z_spec, spec_mask)

        return self.decode(trg, z_spec, spec_mask, test, kv_cache)

    def x_step(self, batch: Tensor, step_type: str, save_output: bool = False) -> Tensor:
        if self.use_spectra == "energy" or self.use_spectra == "binned_mass":
            batch, spectra = batch
        else:
            raise ValueError("Binned Pathformer only supports binned_mass spectra.")
        if isinstance(batch, list):
            batch = batch[0]
        # y is one token ahead of x
        x = batch[:, :-1]
        y = batch[:, 1:]
        # forward pass
        model_out = self.forward(x, spectra=spectra)
        # calculate loss, correctly masking pad_id and root node tokens
        per_token_loss = self.loss_fn(transpose(model_out, 1, 2), y)
        gen_mask = ~self.get_root_mask(y)[0]
        gen_mask &= ne(y, self.pad_id)
        loss = (per_token_loss * gen_mask).sum() / gen_mask.sum()
        # calculate other metrics and log everything
        seq_acc, char_acc = seq_metrics_pathformer(model_out.detach().argmax(2), y, gen_mask)
        self.log(f"{step_type}_loss", loss.item(), on_step=step_type == "train", logger=True, sync_dist=True,
                 prog_bar=True, batch_size=x.size(0))
        if step_type == "train":
            self.log("lr", self.lr, on_step=True, prog_bar=True, batch_size=1)
        self.log(f"{step_type}_seq_acc", seq_acc, sync_dist=True, prog_bar=True, batch_size=x.size(0))
        self.log(f"{step_type}_char_acc", char_acc, sync_dist=True, prog_bar=True, batch_size=x.size(0))
        self.metric_history["loss"][step_type].append(loss.item())
        self.metric_history["char_acc"][step_type].append(char_acc.item())
        self.metric_history["seq_acc"][step_type].append(seq_acc.item())

        if save_output:
            self.model_outputs[step_type].append([model_out.detach().cpu(), y.cpu()])
        return loss

    def test_step(self, batch: Tensor, batch_id: int) -> None:
        spectra = None
        if self.model_config.get("use_spectra", False) == "energy" or self.use_spectra == "binned_mass":
            batch, spectra = batch
        if isinstance(batch, list):
            batch = batch[0]
        x = batch[:, :-1]
        _, idx0_pos = self.get_root_mask(x)
        roots = []
        for i in range(x.size(0)):
            roots.append(x[i, :idx0_pos[i] + 1])
        roots = pad_sequence(roots, batch_first=True, padding_value=self.pad_id)
        sorted_mols, sorted_lls, all_lls = beam_decode_pathformer(self, roots, idx0_pos,
                                                                  num_beams=2 if self.args.debug else self.num_beams,
                                                                  spectra=spectra)
        self.model_outputs["test"].append((roots.cpu(), [sorted_mols.cpu(), x.cpu(), all_lls.cpu()]))

    def on_test_epoch_end(self) -> None:
        outputs = self.model_outputs["test"]
        self.final_test_metrics.update(process_seq_test_outputs_pathformer(outputs,
                                                                           self.save_path, self.i_to_char,
                                                                           self.model_config["length_norm"], self.args))
        with open(os.path.join(self.save_path, "final_test_metrics.json"), "w") as t_file:
            json.dump(self.final_test_metrics, t_file)

    def get_root_mask(self, batch):
        idx0_pos = (batch == self.idx0_id).int()
        idx0_pos = idx0_pos.argmax(dim=1)
        pos = arange(batch.size(1), device=batch.device).unsqueeze(0)
        prefix_allow = pos <= idx0_pos.unsqueeze(1)
        return prefix_allow, idx0_pos

    def _transformer_lr(self, step):
        mult = (self.d_model // 2) ** -0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step ** -0.5, step * (self.warm_up_steps ** -1.5))
        new_lr = self.lr_mult * mult * lr
        self.lr = new_lr
        return new_lr

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimiser = AdamW(trainable_params, lr=self.lr)
        if self.model_config["scheduler"] == "trans":
            trans_sch = FuncLR(optimiser, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}
            return [optimiser], [sch]
        elif self.model_config["scheduler"] == "trans_plateau":
            trans_sch = FuncLR(optimiser, lr_lambda=self._transformer_lr)
            reduce = TransformerReduceLROnPlateau(optimiser, self, mode="min", factor=0.8, patience=4, min_lr=1e-10)
            return [optimiser], [{"scheduler": trans_sch, "interval": "step", "frequency": 1},
                                 {"scheduler": reduce, "monitor": "val_loss", "interval": "epoch", "frequency": 1}]
        elif self.model_config["scheduler"] == "plateau":
            scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.8, patience=4, min_lr=1e-10)
            return {"optimizer": optimiser, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        elif self.model_config["scheduler"] == "cosine":
            scheduler = CosineAnnealingLR(optimiser, self.args.epochs, eta_min=1e-10)
            return [optimiser], [scheduler]
        elif self.model_config["scheduler"] == "cosine_plateau":
            cosine = CosineAnnealingLR(optimiser, self.args.epochs, eta_min=1e-10)
            reduce = ReduceLROnPlateau(optimiser, mode="min", factor=0.8, patience=4, min_lr=1e-10)
            return [optimiser], [{"scheduler": cosine, "interval": "epoch", "frequency": 1},
                                 {"scheduler": reduce, "monitor": "val_loss", "interval": "epoch", "frequency": 1}]
        else:
            raise ValueError(f"Unknown lr scheduler: {self.model_config['scheduler']}")

    def reset_model(self):
        self.final_test_metrics = {}
        self.model_outputs = {"train": [], "val": [], "test": []}
        self._init_params()
