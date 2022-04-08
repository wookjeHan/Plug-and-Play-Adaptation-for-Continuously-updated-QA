import argparse
import os
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import load_model_tokenizer
from utils import prepare_dataset

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def print_title(s):
    print()
    print('=' * 80)
    print(s)
    print('=' * 80)


def main(args):
    args.mode = "pretrain"
    # Loading Model & Tokenizer
    print_title("Loading model, tokenizer")
    model, tokenizer = load_model_tokenizer(args)
    # Loading train set & dev set
    print_title("Loading train set and dev set")
    data_module = prepare_dataset(args, tokenizer)

    # setup dataModule
    data_module.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_dir,
        filename="best-checkpoint",
        save_last=True,
        save_top_k=3,
        verbose=True,
        monitor="harmonic_score(normalized)",
        mode="max"
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args.n_epoch,
        gpus=args.n_gpus,
        accelerator="ddp",
        check_val_every_n_epoch=args.validation_freq
    )
    # When Pretraining, LoRA Parameters must not be updated
    model.freeze()
    no_decay = ['bias', 'layer_norm.weight']
    for n, p in model.named_parameters():
        if not 'lora' in n:
            p.requires_grad = True
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (not 'lora' in n) and not any(nd in n for nd in no_decay)],
            'lr': args.lr,
            'weight_decay':args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if (not 'lora' in n) and any(nd in n for nd in no_decay)],
            'lr': args.lr,
            'weight_decay':0
        }
    ]
    model.set_grouped_parameters(optimizer_grouped_parameters)
    model.train()
    print_title("Start Training...")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices = ["nq","zsRE"], default = "nq")
    parser.add_argument("--n_gpus", type=int, default=4)

    parser.add_argument("--train_path", type=str,
                        default=None, required=True)
    parser.add_argument("--dev_path", '--list', nargs='+', type=str,
                        default=None, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epoch", type=int, default=80)
    parser.add_argument("--validation_freq", type=int, default=4)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--max_source_len", type=int, default=25)
    parser.add_argument("--max_target_len", type=int, default=10)

    parser.add_argument("--out_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    main(args)
