import json
import os
import re
import string

import numpy as np
import pandas as pd

from data import QADataModule


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


def extract_questions_and_answers(json_dataset):
    data_rows = []

    for json_data in json_dataset[:1320]:
        question = json_data['question']
        answer_text = json_data['answer']
        aliases = json_data["aliases"]
        data_rows.append({
            "question": question,
            'aliases': aliases,
            "answer_text": answer_text,
        })
    return pd.DataFrame(data_rows)


def prepare_dataset(args, tokenizer):
    train_path = args.train_path if "train_path" in args else None
    dev_paths = args.dev_path
    # train_path is None if we are evaluating..
    if train_path is not None:
        with open(train_path, 'r') as f:
            train_dataset = json.load(f)
            size = len(train_dataset)
            print(size)
            f.close()
        train_df = extract_questions_and_answers(train_dataset)
    # consider multiple dev_paths to evaluate multiple dev_datasets concurrently.
    dev_datasets = []
    for dev_path in dev_paths:
        with open(dev_path, 'r') as f:
            dev_dataset = json.load(f)
            dev_datasets.append(dev_dataset[:1000])
            size = len(dev_dataset)
            print(size)
            f.close()

    dev_df = [extract_questions_and_answers(
        dev_dataset) for dev_dataset in dev_datasets]
    data_module = QADataModule(
        train_df=train_df if train_path is not None else dev_df, test_df=dev_df, tokenizer=tokenizer, batch_size=args.batch_size, source_max_token_len=args.max_source_len, target_max_token_len=args.max_target_len)
    return data_module
