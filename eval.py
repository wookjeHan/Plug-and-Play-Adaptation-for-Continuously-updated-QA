import argparse
import os
import warnings

import pytorch_lightning as pl

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
    args.mode = 'eval'
    # Loading Model & Tokenizer
    print_title("Loading model, tokenizer")
    model, tokenizer = load_model_tokenizer(args)
    # Loading train set & dev set
    print_title("Loading train set and dev set")
    data_module = prepare_dataset(args, tokenizer)

    # setup dataModule
    data_module.setup()
    trainer = pl.Trainer(
        max_epochs=0,
        gpus=args.n_gpus,
        accelerator="ddp",
        check_val_every_n_epoch=0
    )
    # When Pretraining, LoRA Parameters should not be updated
    trainer.test(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices = ["nq","zsRE"], default = "nq")
    parser.add_argument("--n_gpus", type=int, default=4)

    parser.add_argument("--dev_path", '--list', nargs='+', type=str,
                        default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, default=None, required=True)
    parser.add_argument("--max_source_len", type=int, default=25)
    parser.add_argument("--max_target_len", type=int, default=10)
    
    parser.add_argument("--adapter", type=str, choices=[
                        "LoRA", "K-adapter", "None"], default="LoRA")
    # args for Eval When LoRA
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--lora_attn_alpha", type=int, default=256*4)
    parser.add_argument("--attn_lora_rank", type=int, default=256)
    parser.add_argument("--lora_attn_attn_alpha", type=int, default=256*4)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r_dropout", type=float, default=0.1)
    # This argument is only of Ours+LoRA Method
    parser.add_argument("--ours_threshold", type=float, default=0.9)

    args = parser.parse_args()

    main(args)
