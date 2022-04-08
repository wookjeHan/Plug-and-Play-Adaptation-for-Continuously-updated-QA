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
    args.mode = "update"
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
        save_top_k=2,
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

    model.freeze()
    no_decay = ['bias', 'layer_norm.weight']

    # When Finetuning or RecAdam, LoRA Parameters should not be updated
    if args.method == "ft" or args.method == "rec":
        for n, p in model.named_parameters():
            if not "lora" in n:
                p.requires_grad = True
        if args.method == "ft":
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
        # When RecAdam we must save original model's parameters.
        elif args.method == "rec":
            orig_model, _ = load_model_tokenizer(args)
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and "lora" not in n],
                    "weight_decay": args.weight_decay,
                    "anneal_w": args.rec_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in orig_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and "lora" not in p_n]
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "anneal_w": args.rec_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in orig_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and "lora" not in p_n]
                }
            ]
    # When LoRA
    elif args.method == "LoRA":
        for n, p in model.named_parameters():
            if "lora" in n and ".0.weight" not in n:
                p.requires_grad = True
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if ("lora" in n and ".0.weight" not in n) and not any(nd in n for nd in no_decay)],
                'lr': args.lr,
                'weight_decay': args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if ('lora' in n and ".0.weight" not in n) and any(nd in n for nd in no_decay)],
                'lr': args.lr,
                'weight_decay':0
            }
        ]
    elif args.method == "Ours+LoRA":
        for n, p in model.named_parameters():
            if "lora" in n and ".{}.weight".format(model.config.lora_expert_num-1) in n:
                p.requires_grad = True
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if ('lora' in n and ".{}.weight".format(model.config.lora_expert_num-1) in n) and not any(nd in n for nd in no_decay)],
                'lr': args.lr,
                'weight_decay':args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if ('lora' in n and ".{}.weight".format(model.config.lora_expert_num-1) in n) and any(nd in n for nd in no_decay)],
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
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_source_len", type=int, default=25)
    parser.add_argument("--max_target_len", type=int, default=10)
    parser.add_argument("--method", type=str, choices=[
                        "ft", "rec", "LoRA", "Ours+LoRA"], required=True)

    # args for Rec
    parser.add_argument("--rec_anneal_fun", type=str, default="sigmoid")
    parser.add_argument("--rec_anneal_k", type=float, default=0.5)
    parser.add_argument("--rec_anneal_t0", type=float, default=10.0)
    parser.add_argument("--rec_anneal_w", type=float, default=1.0)
    parser.add_argument("--rec_pretrain_cof", type=float, default=5000.0)

    # args for LoRA
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--lora_attn_alpha", type=int, default=256*4)
    parser.add_argument("--attn_lora_rank", type=int, default=256)
    parser.add_argument("--lora_attn_attn_alpha", type=int, default=256*4)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r_dropout", type=float, default=0.1)
    parser.add_argument("--ours_threshold", type=float, default=0.9)
    # parser.add_argument("--lora_expert_num", type=int, default=6)

    parser.add_argument("--out_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    main(args)
