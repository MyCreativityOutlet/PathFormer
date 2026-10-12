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

from models.PathFormer import PathFormer
from utils.utils import *
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import lightning as L


def get_config(args):
    if args.model_name == "PathFormer":
        with open("models/PathFormer_config.json") as c_file:
            config = json.load(c_file)
    else:
        raise ValueError(f"Unknown pathformer model {args.data_name}")
    return config


def build_trainer(args, config: dict, monitor_value="val_char_acc", version="") -> L.Trainer:
    full_version = get_pathformer_version(args, config) + "/" + version
    logger = TensorBoardLogger(f"results/", name=args.model_name, version=full_version)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [lr_monitor]
    if "uspto" not in args.data_name:
        config["grad_batch_size"] //= 2
    patience = 10
    monitor_mode = "max" if "acc" in monitor_value else "min"
    early_stopping = EarlyStopping(monitor=monitor_value, mode=monitor_mode, patience=patience,
                                   check_on_train_epoch_end=False, verbose=True, min_delta=0.0001)
    checkpoint_cb = ModelCheckpoint(monitor=monitor_value, mode=monitor_mode, filename="best-{epoch}-{" + monitor_value + ":.4f}")
    callbacks.append(early_stopping)
    callbacks.append(checkpoint_cb)
    if "uspto" in args.data_name:
        # This has to go after the first checkpoint callback as accessing it later auto uses the first one
        # Periodic checkpoint saved every 5 epochs
        every_5_epochs_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best-{epoch}-{val_loss:.4f}",
        )
        callbacks.append(every_5_epochs_cb)
    acc_grad_batch = max(1, int(config["grad_batch_size"] / args.batch_size))
    return L.Trainer(accelerator="gpu", logger=logger, strategy="auto", devices=args.gpus, callbacks=callbacks,
                     default_root_dir=f"results/{args.model_name}/{full_version}", detect_anomaly=args.debug,
                     max_epochs=67 if args.debug else args.epochs, gradient_clip_val=1.0,
                     accumulate_grad_batches=acc_grad_batch, log_every_n_steps=10, precision="bf16-mixed",
                     num_sanity_val_steps=0, deterministic=True, limit_train_batches=4 if args.debug else 1.0,
                     limit_test_batches=2 if args.debug else 1.0, limit_val_batches=4 if args.debug else 1.0)


def train(args, config) -> None:
    print(f"Debug: {args.debug}\nModel: {args.model_name}\nDataset: {args.data_name}\nPreprocessor: {args.preprocessor}")
    print("Setting up")
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('high')
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    if args.model_name == "PathFormer":
        model_class = PathFormer
    else:
        raise ValueError(f"Unknown model type {args.model_name}")
    config.update(args_dict)
    # Load data and tokenisers
    if config["enzymes"]:
        with open("data/tokenizers_pathformer_ec.json") as token_file:
            tokenizers = json.load(token_file)
    else:
        with open("data/tokenizers_pathformer.json") as token_file:
            tokenizers = json.load(token_file)
    try:
        char_to_i = tokenizers[args.tokenizer]
    except KeyError:
        raise KeyError(f"{args.tokenizer} is not a valid tokenizer")
    i_to_char = {i: char for char, i in char_to_i.items()}
    tokenizer = (char_to_i, i_to_char)
    result_path = f"results/{args.model_name}/{get_pathformer_version(args, config)}"
    print("Checking if split already exists in database")
    pathformer_split_disjoint(args)
    # Set up the model and load pretrained weights, if used
    print(f"Version: {get_pathformer_version(args, config)}")
    os.makedirs(result_path, exist_ok=True)
    model = model_class(config, tokenizer, args)
    max_epoch = False
    if os.path.exists(ckpt_folder := os.path.join(model.save_path, "checkpoints")) and len(os.listdir(ckpt_folder)) > 0:
        ckpts = [f for f in os.listdir(ckpt_folder) if ".ckpt" in f]
        file = max(ckpts, key=lambda x: int(x.split("=")[1].split("-")[0].split(".")[0]))
        restore_epoch = int(file.split("=")[1].split("-")[0].split(".")[0])
        if restore_epoch >= args.epochs - 10:
            print("Attempted to continue training from existing config checkpoint but max epoch reached. "
                  "Will not train, just test")
            max_epoch = True
        ckpt_path = os.path.join(ckpt_folder, file)
    elif args.weights_dataset:
        ckpt_folder = f"{args.weights_dataset}/checkpoints/"
        ckpts = [f for f in os.listdir(ckpt_folder) if ".ckpt" in f]
        file = max(ckpts, key=lambda x: int(x.split("=")[1].split("-")[0].split(".")[0]))
        ckpt_path = os.path.join(ckpt_folder, file)
    else:
        ckpt_path = None
    # Build the data loaders, train and test
    print("Encoding data to tensors")
    train_loader, val_loader, test_loader = model.encode_data()
    trainer = build_trainer(args, config)

    if not args.no_train and not max_epoch:
        print("Beginning training")
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path, weights_only=False)
    if not args.train_many or args.data_name != "uspto":
        trainer.test(model, test_loader,
                     ckpt_path=trainer.checkpoint_callback.best_model_path if not args.no_train else ckpt_path,
                     weights_only=False)
        # Analyse multi gen evaluation results on the test set
        multi_test_output = {"predictions": []}
        for root, result in model.final_test_metrics["predictions"].items():
            true = result["actual"]
            # Needs to be sorted in reverse so that lower probability nodes are added first
            # then overwritten with the higher probabilities.
            keys = sorted(list(result["predict"].keys()), key=lambda x: int(x.split("_")[-1]), reverse=True)
            p_graph = nx.DiGraph()
            for i, key in enumerate(keys):
                graphs = result["predict"][key]
                for j, graph in enumerate(graphs):
                    graph = nx.node_link_graph(graph)
                    for n in graph:
                        graph.nodes[n]["probability"] *= int(key.split("_")[-1])
                    p_graph.update(graph)
            predicted = nx.node_link_data(p_graph)
            invalid_smiles = result["invalid_smiles"]["top_1"]
            for t in true:
                multi_test_output["predictions"].append([t, predicted, invalid_smiles])

        thresholds = get_pathformer_thresholds(value_type=set)
        recall, precision = process_compare_pathways(args, multi_test_output, thresholds, save_path=model.save_path)
        multi_test_output["precision"] = precision
        multi_test_output["recall"] = recall
        # Save multi gen eval results
        multi_recall, multi_precision, multi_area = sort_recall_precision(multi_test_output["recall"],
                                                                          multi_test_output["precision"])
        multi_pr = [[multi_recall, multi_precision, f"{args.model_name} AUC: {multi_area:.3f}"]]
        plot_pr_curve([], [], 0, args.data_name, result_path, extra_pr=multi_pr,
                      file_name=f"multi_fold{args.seed}")
        with open(f"{result_path}/test_output_multi.json", "w") as out_file:
            json.dump(multi_test_output, out_file)


def train_one(args):
    config = get_config(args)
    train(args, config)


def train_many(args):
    search_space = {"seed": list(range(10)), "data_name": ["envipath_no_metal", "bbd_no_metal", "soil_no_metal", "sludge_no_metal", "sediment_no_metal"]}
    search_space = expand_search_space(search_space)
    config = get_config(args)
    for search in search_space:
        for k, v in search.items():
            if k in vars(args) and k not in config:
                setattr(args, k, v)
        for k, v in config.items():
            if k not in search:
                search[k] = v
        train(args, search)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import torch

    arguments = get_arguments()
    if arguments.train_many:
        train_many(arguments)
    else:
        train_one(arguments)
