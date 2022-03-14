import argparse
import json
import os
import re
import string
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformers_2.data.metrics.squad_metrics import normalize_answer
from transformers_2 import (AdamW, T5Config, T5ForConditionalGeneration,
                            T5Tokenizer)

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def print_title(s):
    print()
    print('=' * 80)
    print(s)
    print('=' * 80)


def get_exact_match(prediction, groundtruth):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


class QAModel(pl.LightningModule):

    def __init__(self, model_name, tokenizer, args):
        super().__init__()
        self.config = T5Config.from_pretrained(model_name, return_dict=True)
        self.config.dropout_rate = 0
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, config=self.config)
        self.avg_accs = []
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids, attention_mask, labels=None, expert_labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            router_phase=1,
            expert_labels=expert_labels,
        )
        return output.loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # As Pretraining, we do not need to exploit LoRA parameters.
        types = torch.tensor(
            [0 for i in range(input_ids.shape[0])], device=self.device)

        loss = self(input_ids, attention_mask,
                    labels, expert_labels=types)
        loss = torch.mean(loss)

        return {"loss": loss}

    def training_step_end(self, batch_parts):
        return batch_parts

    def validation_step(self, batch, batch_idx):
        output = []
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        task_A = torch.tensor(
            [0 for j in range(len(batch['input_ids']))], device=self.device)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            max_length=self.args.max_target_len,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            task_A=task_A,
        )
        predictions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output.append((predictions, batch['aliases']))
        return output

    def validation_epoch_end(self, validation_step_outputs):
        correct = 0
        correct_normal = 0
        total_num = 0

        for output in validation_step_outputs:
            predictions, answers = output[0]
            for pred, ans in zip(predictions, answers):
                total_num += 1
                # skip if prediction is an empty string, as there are some empty string in aliases.
                if pred == "":
                    continue
                if pred in ans[0]:
                    correct += 1
                if get_exact_match(pred, ans[0]) == True:
                    correct_normal += 1

        self.log("Task_acc(normalized)", correct_normal/total_num, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        print("Task acc(noramlized):", correct_normal/total_num)

    def test_step(self, batch, batch_idx):
        output = []
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        task_A = torch.tensor(
            [0 for j in range(len(batch['input_ids']))], device=self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            max_length=args.max_target_len,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            task_A=task_A,
        )
        predictions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output.append((input_ids, predictions, batch['aliases']))
        return output

    def test_epoch_end(self, test_step_outputs):
        correct = 0
        correct_normal = 0
        total_num = 0

        for output in test_step_outputs:
            predictions, answers = output[0]
            for pred, ans in zip(predictions, answers):
                total_num += 1
                if pred == "":
                    continue
                if pred in ans[0]:
                    correct += 1
                if get_exact_match(pred, ans[0]) == True:
                    correct_normal += 1

        self.log("task_acc(normalized)", correct_normal/total_num, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        print("task acc(noramlized):", correct_normal/total_num)

    def set_grouped_parameters(self, grouped_parameters):
        self.grouped_parameters = grouped_parameters

    def configure_optimizers(self):
        if self.grouped_parameters is None:
            return AdamW(self.parameters(), lr=5e-5)
        else:
            return AdamW(self.grouped_parameters, betas=(0.9, 0.999), eps=1e-6,
                         correct_bias=True)


class QADataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int,
        target_max_token_len: int
    ):

        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        source_encoding = self.tokenizer(
            data_row["question"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation="only_first",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            data_row["answer_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation="only_first",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            question=data_row["question"],
            aliases=[data_row['aliases']],
            answer_text=data_row["answer_text"],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
        )


class QADataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int,
        source_max_token_len: int,
        target_max_token_len: int
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self):
        self.train_dataset = QADataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = QADataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

    def collate_fn(self, batch):
        questions = [b['question'] for b in batch]
        aliases = [b['aliases'] for b in batch]
        answers = [b['answer_text'] for b in batch]
        input_ids = torch.tensor([list(b['input_ids']) for b in batch])
        attention_masks = torch.tensor(
            [list(b['attention_mask']) for b in batch])
        labels = torch.tensor([list(b['labels']) for b in batch])

        batches = {
            "question": questions,
            "aliases": aliases,
            "answers": answers,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }
        return batches

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=24,
            drop_last=False,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
            drop_last=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=24,
            drop_last=False
        )


def extract_questions_and_answers(json_dataset):
    data_rows = []

    for json_data in json_dataset:
        question = json_data['question']
        answer_text = json_data['answer']
        aliases = json_data["aliases"]
        data_rows.append({
            "question": question,
            'aliases': aliases,
            "answer_text": answer_text,
        })
    return pd.DataFrame(data_rows)


def load_model_tokenizer(args):
    tokenizer = T5Tokenizer.from_pretrained("google/t5-large-ssm")
    if args.init_checkpoint is None:
        model = QAModel("google/t5-large-ssm", tokenizer, args)
    else:
        assert os.path.exists(args.init_checkpoint)
        model = QAModel.load_from_checkpoint(
            args.init_checkpoint, strict=False)
    return model, tokenizer


def prepare_dataset(train_path, dev_path, tokenizer):
    with open(train_path, 'r') as f:
        train_dataset = json.load(f)
        size = len(train_dataset)
        print(size)
        f.close()

    with open(dev_path, 'r') as f:
        dev_dataset = json.load(f)
        size = len(dev_dataset)
        print(size)
        f.close()

    train_df = extract_questions_and_answers(train_dataset)
    dev_df = extract_questions_and_answers(dev_dataset)
    data_module = QADataModule(
        train_df=train_df, test_df=dev_df, tokenizer=tokenizer, batch_size=args.batch_size, source_max_token_len=args.max_source_len, target_max_token_len=args.max_target_len)
    return data_module


def main(args):
    # Loading Model & Tokenizer
    print_title("Loading model, tokenizer")
    model, tokenizer = load_model_tokenizer(args)
    # Loading train set & dev set
    print_title("Loading train set and dev set")
    data_module = prepare_dataset(args.train_path, args.dev_path, tokenizer)

    # setup dataModule
    data_module.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out_dir,
        filename="best-checkpoint",
        save_last=True,
        save_top_k=3,
        verbose=True,
        monitor="task_acc(normalized)",
        mode="max"
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        # checkpoint_callback=checkpoint_callback,
        max_epochs=args.n_epoch,
        gpus=args.n_gpus,
        accelerator="ddp",
        check_val_every_n_epoch=args.validation_freq
    )
    # When Pretraining, LoRA Parameters should not be updated
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
    parser.add_argument("--n_gpus", type=int, default=4)

    parser.add_argument("--train_path", type=str,
                        default="Dataset/zsRE/train/zeroshot_train_A.json")
    parser.add_argument("--dev_path", type=str,
                        default="Dataset/zsRE/dev/zeroshot_dev_A.json")
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
