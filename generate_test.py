from utils import get_exact_match
import argparse
import os
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
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
    args.checkpoint = "LoRA/best-checkpoint.ckpt"
    args.mode = "update"
    # Loading Model & Tokenizer
    print_title("Loading model, tokenizer")
    model, tokenizer = load_model_tokenizer(args)
    # Loading train set & dev set
    print_title("Loading train set and dev set")
    data_module = prepare_dataset(args, tokenizer)
    model.eval()
    # setup dataModule
    data_module.setup()
    for idx, data in enumerate(data_module.test_datasets[0]):
        if idx < 12:
            continue

        def generate_answer(example, trained_model, switches=[1]):
            source_encoding = tokenizer(
                example["question"],
                # question["context"],
                max_length=args.max_source_len,
                padding="max_length",
                truncation="only_first",
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            generated_ids = trained_model.model.generate(
                input_ids=source_encoding["input_ids"].to(
                    trained_model.device),
                attention_mask=source_encoding["attention_mask"].to(
                    trained_model.device),
                num_beams=1,
                max_length=args.max_target_len,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                use_cache=True,
                switches=switches
            )
            preds = [
                tokenizer.decode(generated_id, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
                for generated_id in generated_ids
            ]

            return "".join(preds), source_encoding
        # print("GENERATE")
        # pred, source_encoding = generate_answer(data, model, switches=[1])
        source_encoding = tokenizer(
            data["question"],
            # question["context"],
            max_length=args.max_source_len,
            padding="max_length",
            truncation="only_first",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        print("FORWARD")
        loss, logits = model(input_ids=source_encoding["input_ids"].to(
            model.device), attention_mask=source_encoding["attention_mask"].to(model.device),
            labels=data["labels"].view(1, -1), switches=torch.tensor([1], device=model.device))
        print("GENERATED")
        print(logits.shape)
        print(logits[:, -1, :])
        quit()
        if not get_exact_match(pred, data['answer_text']):
            print("IDX")
            print(idx)
            print("PRED")
            print(pred)
            print("ANs")
            print(data["answer_text"])
            print("LOSS")
            print(loss)
            quit()


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
    parser.add_argument("--n_epoch", type=int, default=40)
    parser.add_argument("--validation_freq", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_source_len", type=int, default=25)
    parser.add_argument("--max_target_len", type=int, default=10)
    parser.add_argument("--method", type=str, choices=[
                        "ft", "rec", "LoRA", "Ours_LoRA"], required=True)

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

    parser.add_argument("--out_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    main(args)
