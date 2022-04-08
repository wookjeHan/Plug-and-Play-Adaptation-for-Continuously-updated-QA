from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.trainer.supporters import CombinedLoader

from transformers import T5Tokenizer


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
        test_df: List[pd.DataFrame],
        tokenizer: T5Tokenizer,
        batch_size: int,
        source_max_token_len: int,
        target_max_token_len: int
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_dfs = test_df
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

        self.test_datasets = [QADataset(
            test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        ) for test_df in self.test_dfs]

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
        loaders = {}
        for i, test_dataset in enumerate(self.test_datasets):
            loaders[chr(ord('a') + i)] = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=24,
                drop_last=False,
                collate_fn=self.collate_fn
            )
        return CombinedLoader(loaders, "max_size_cycle")

    def test_dataloader(self):
        loaders = {}
        for i, test_dataset in enumerate(self.test_datasets):
            loaders[chr(ord('a') + i)] = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=24,
                drop_last=False,
                collate_fn=self.collate_fn
            )
        return CombinedLoader(loaders, "max_size_cycle")
