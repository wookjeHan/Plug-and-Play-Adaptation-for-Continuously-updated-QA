import os

import numpy as np
import torch
import pytorch_lightning as pl
import torch.distributed as dist

from transformers import (AdamW, T5Config, T5ForConditionalGeneration,
                          T5Tokenizer)
from utils import get_exact_match
from RecAdam import RecAdam


def load_model_tokenizer(args):
    tokenizer = T5Tokenizer.from_pretrained("google/t5-large-ssm")
    if args.mode == 'pretrain':
        if args.init_checkpoint is None:
            model = QAModel("google/t5-large-ssm", tokenizer, args)
        else:
            assert os.path.exists(args.init_checkpoint)
            model = QAModel.load_from_checkpoint(
                args.init_checkpoint, strict=False, model_name="google/t5-large-ssm", tokenizer=tokenizer, args=args)
        return model, tokenizer
    elif args.mode == "eval" or args.mode == "update":
        assert args.checkpoint is not None and os.path.exists(
            args.checkpoint), "Must set model's valid checkpoint to be evaluated"

        model = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        if args.mode == "update" and args.method != "ft" and args.method != "rec":
            args.lora_expert_num = model["lora_expert_num"] + \
                1 if "lora_expert_num" in model else 2
        else:
            args.lora_expert_num = model["lora_expert_num"] if "lora_expert_num" in model else 2
        model = QAModel.load_from_checkpoint(
            args.checkpoint, strict=False, model_name="google/t5-large-ssm", tokenizer=tokenizer, args=args)
        return model, tokenizer


class QAModel(pl.LightningModule):

    def __init__(self, model_name, tokenizer, args):
        super().__init__()
        self.config = T5Config.from_pretrained(model_name, return_dict=True)
        
        if (args.mode != "pretrain"):
            self.config.is_lora = args.lora_rank
            self.config.lora_attn_alpha = args.lora_attn_alpha
            self.config.lora_attn_attn_alpha = args.lora_attn_attn_alpha
            self.config.lora_dropout = args.lora_dropout
            self.config.lora_r_dropout = args.lora_r_dropout
            self.config.attn_is_lora = args.attn_lora_rank
            self.config.dropout_rate = 0
            self.config.lora_expert_num = args.lora_expert_num

        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, config=self.config)
        self.tokenizer = tokenizer
        self.args = args
        self._grouped_parameters = None
        self.embedding_memory = []
        self._type = None

    def forward(self, input_ids, attention_mask, labels=None, switches=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            switches=switches,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # If LoRA we should switch all experts
        if self.args.mode == "update" and self.args.method == "LoRA":
            switch_labels = torch.tensor(
                [-1 for i in range(input_ids.shape[0])], device=self.device
            )
        elif self.args.mode == "update" and self.args.method == "Ours+LoRA":
            # TODO CHANGE
            switch_labels = torch.tensor(
                [self.config.lora_expert_num - 1 for i in range(input_ids.shape[0])], device=self.device
            )
        # If we are not exploiting LoRA switches should be 0
        else:
            switch_labels = torch.tensor(
                [0 for i in range(input_ids.shape[0])], device=self.device)

        _, avg_embedding = _, avg_embedding = self.model.expert_prepare(
                        input_ids=input_ids, attention_mask=attention_mask)
        loss, _ = self(input_ids, attention_mask,
                                      labels, switches=switch_labels)
        
        loss = torch.mean(loss)

        return {"loss": loss, "avg_embedding": avg_embedding}

    def training_step_end(self, batch_parts):
        return batch_parts

    def training_epoch_end(self, training_step_outputs):
        if self.args.mode == "update" and self.args.method == "Ours+LoRA":
            if self.current_epoch == 0:
                temp = []
                for i in training_step_outputs:
                    target = i["avg_embedding"]  # (batch,hidden)

                    norm = target.norm(p=2, dim=1, keepdim=True)
                    target = target.div(norm)

                    for tar_ in target:
                        temp.append(tar_)

                temp = torch.stack(temp)  # (example, hidden)

                def gather_list_and_concat(tensor):
                    gather_t = [torch.ones_like(tensor)
                                for _ in range(dist.get_world_size())]
                    dist.all_gather(gather_t, tensor)
                    return torch.cat(gather_t)

                self.embedding_memory.append(gather_list_and_concat(temp))
    def validation_step(self, batches, batch_idx):
        output = []
        for dataset in batches.keys():
            batch = batches[dataset]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            # If LoRA we should switch all experts
            if self.args.mode == "update" and self.args.method == "LoRA":
                switches = torch.tensor(
                    [-1 for i in range(input_ids.shape[0])], device=self.device
                )
            elif self.args.mode == "update" and self.args.method == "Ours+LoRA":
                switches = torch.tensor(
                    [-1 for i in range(input_ids.shape[0])], device=self.device
                )
                if len(self.embedding_memory) > 0:
                    _, avg_embedding = self.model.expert_prepare(
                        input_ids=input_ids, attention_mask=attention_mask)
                    norm = avg_embedding.norm(p=2, dim=1, keepdim=True)
                    pred = avg_embedding.div(norm)  # (batch, hidden_size)

                    scores = []
                    for tensor in self.embedding_memory:
                        score_cand = torch.matmul(
                            pred, tensor.transpose(1, 0).to(self.device))
                        mx, ind = torch.max(score_cand, dim=1)  # (batch, )
                        # score is list of tensor which shape is (batch,)
                        scores.append(mx)
                    mx_scores = []
                    for i in range(input_ids.shape[0]):
                        mx = 0
                        index_of_mx = 0
                        for ind, j in enumerate(scores):
                            if j[i] > mx:
                                mx = j[i]
                                index_of_mx = ind
                        mx_scores.append((mx, index_of_mx))
                    switches = []
                    for i in mx_scores:
                        if i[0] >= self.args.ours_threshold:
                            switches.append(i[1] + 1)
                        else:
                            switches.append(0)
                    switches = torch.tensor(
                        switches, device=self.device, requires_grad=False
                    )
            # If we are not exploiting LoRA switches should be 0
            else:
                switches = torch.tensor(
                    [0 for i in range(input_ids.shape[0])], device=self.device)
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=1,
                max_length=self.args.max_target_len,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                use_cache=True,
                switches=switches,
            )
            predictions = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output.append((predictions, batch['aliases']))
        return output

    def validation_epoch_end(self, validation_step_outputs):
        dataset_num = len(validation_step_outputs[0])
        corrects = [0 for i in range(dataset_num)]
        correct_normals = [0 for i in range(dataset_num)]
        total_nums = [0 for i in range(dataset_num)]

        for outputs in validation_step_outputs:
            for i, output in enumerate(outputs):
                predictions, answers = output
                for pred, ans in zip(predictions, answers):
                    total_nums[i] += 1
                    if pred == "":
                        continue
                    if pred in ans[0]:
                        corrects[i] += 1
                    if get_exact_match(pred, ans[0]) == True:
                        correct_normals[i] += 1
        # Let's calculate harmonic mean.
        inverse_norm_scores = [t / (c + 1e-6) for c,
                               t in zip(correct_normals, total_nums)]
        avg_inverse_norm_scores = sum(
            inverse_norm_scores) / len(inverse_norm_scores)
        harmonic_scores = 1 / avg_inverse_norm_scores
        for i in range(dataset_num):
            self.log("{}'s acc(normalized)".format(self.args.dev_path[i].split("/")[-1]), correct_normals[i]/total_nums[i], on_step=False,
                     on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("harmonic_score(normalized)", harmonic_scores, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batches, batch_idx):
        output = []
        for dataset in batches.keys():
            batch = batches[dataset]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            if self._type == "ft" or self._type == "rec":
                switches = torch.tensor(
                    [0 for j in range(len(batch['input_ids']))], device=self.device)
            elif self._type == "lora":
                switches = torch.tensor(
                    [-1 for j in range(len(batch['input_ids']))], device=self.device)
            else:
                switches = torch.tensor(
                        [-1 for i in range(input_ids.shape[0])], device=self.device
                    )
                if len(self.embedding_memory) > 0:
                    _, avg_embedding = self.model.expert_prepare(
                        input_ids=input_ids, attention_mask=attention_mask)
                    norm = avg_embedding.norm(p=2, dim=1, keepdim=True)
                    pred = avg_embedding.div(norm)  # (batch, hidden_size)

                    scores = []
                    for tensor in self.embedding_memory:
                        score_cand = torch.matmul(
                            pred, tensor.transpose(1, 0).to(self.device))
                        mx, ind = torch.max(score_cand, dim=1)  # (batch, )
                        # score is list of tensor which shape is (batch,)
                        scores.append(mx)
                    mx_scores = []
                    for i in range(input_ids.shape[0]):
                        mx = 0
                        index_of_mx = 0
                        for ind, j in enumerate(scores):
                            if j[i] > mx:
                                mx = j[i]
                                index_of_mx = ind
                        mx_scores.append((mx, index_of_mx))
                    switches = []
                    for i in mx_scores:
                        if i[0] >= self.args.ours_threshold:
                            switches.append(i[1] + 1)
                        else:
                            switches.append(0)
                    switches = torch.tensor(
                        switches, device=self.device, requires_grad=False
                    )
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=1,
                max_length=self.args.max_target_len,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                use_cache=True,
                switches=switches,
            )
            predictions = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            output.append((predictions, batch['aliases']))
        return output

    def test_epoch_end(self, test_step_outputs):
        dataset_num = len(test_step_outputs[0])
        corrects = [0 for i in range(dataset_num)]
        correct_normals = [0 for i in range(dataset_num)]
        total_nums = [0 for i in range(dataset_num)]

        for outputs in test_step_outputs:
            for i, output in enumerate(outputs):
                predictions, answers = output
                for pred, ans in zip(predictions, answers):
                    total_nums[i] += 1
                    if pred == "":
                        continue
                    if pred in ans[0]:
                        corrects[i] += 1
                    if get_exact_match(pred, ans[0]) == True:
                        correct_normals[i] += 1
        for i in range(dataset_num):
            self.log("{}'s acc(normalized): ".format(self.args.dev_path[i]), correct_normals[i]/total_nums[i], on_step=False,
                     on_epoch=True, prog_bar=True, sync_dist=True)

    def set_grouped_parameters(self, grouped_parameters):
        self._grouped_parameters = grouped_parameters

    def configure_optimizers(self):
        if self.args.mode == "update" and self.args.method == "rec":
            return RecAdam(self._grouped_parameters, lr=self.args.lr, eps=1e-6, anneal_fun=self.args.rec_anneal_fun,
                           anneal_t0=self.args.rec_anneal_t0, anneal_k=self.args.rec_anneal_k, pretrain_cof=self.args.rec_pretrain_cof)
        return AdamW(self._grouped_parameters, betas=(0.9, 0.999), eps=1e-6,
                     correct_bias=True)

    def on_save_checkpoint(self, checkpoint):
        if self.args.mode != "pretrain" and (self.args.method == "Ours+LoRA" or self.args.method == "LoRA"):
            checkpoint["embedding_memory"] = self.embedding_memory
            checkpoint["lora_expert_num"] = self.config.lora_expert_num
        if self.args.mode != "pretrain":
            checkpoint["type"] = self.args.method
        else:
            checkpoint["type"] = "ft"

    def on_load_checkpoint(self, checkpoint):
        if "embedding_memory" in checkpoint:
            self.embedding_memory = checkpoint["embedding_memory"]
        else:
            self.embedding_memory = []
        if "type" in checkpoint:
            self._type = checkpoint["type"]