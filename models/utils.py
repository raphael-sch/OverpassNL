import json
import random
import copy

import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class OverpassDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, split, comments_dataset=None, inference_only=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.split = split
        self.inference_only = inference_only

        self.comments_dataset = comments_dataset

        self.max_input_length = 0
        self.max_output_length = 0
        self.max_total_length = 0

    def __getitem__(self, idx):

        if idx >= len(self.texts):

            if self.comments_dataset:
                idx = random.randint(a=0, b=len(self.comments_dataset)-1)
                return self.comments_dataset[idx]

        # regular supervised instance
        text = self.texts[idx][:]
        label = self.labels[idx]
        item = self.tokenizer(text)

        if self.inference_only:
            return item

        with self.tokenizer.as_target_tokenizer():
            item["labels"] = self.tokenizer(label).input_ids
        self.max_input_length = max(self.max_input_length, len(item["input_ids"]))
        self.max_output_length = max(self.max_output_length, len(item["labels"]))
        self.max_total_length = max(self.max_total_length, len(item["labels"]) + len(item["input_ids"]))

        return item

    def __len__(self):
        length = len(self.texts)
        if self.comments_dataset:
            length *= 2
        return length



def read_overpass_split(path, prefix='', num=None):
    nls = list()
    with open(path + '.nl') as f:
        for line in f:
            nls.append(prefix + line.strip())

    queries = list()
    with open(path + '.query') as f:
        for line in f:
            queries.append(line.strip())

    assert len(nls) == len(queries)

    if num is not None:
        nls = nls[:num]
        queries = queries[:num]

    return nls, queries

def get_comment_queries(path, comment_max_count=5, comment_min_content_ratio=0.85):
    comment_count = dict()
    instances = list()
    with open(path) as f:
        for line in f:
            instance = json.loads(line.strip())

            for comment in instance['comments']:
                if comment in comment_count:
                    comment_count[comment] += 1
                else:
                    comment_count[comment] = 1

            query = instance['query']
            if len(query) > 800:
                continue

            instances.append(instance)

    texts = list()
    labels = list()
    for instance in instances:
        comments = instance['comments']
        content_ratios = instance['comment_content_ratio']
        assert len(comments) == len(content_ratios)

        new_comments = list()
        for comment, content_ratio in zip(comments, content_ratios):
            if content_ratio < comment_min_content_ratio:
                continue
            if comment_count[comment] > comment_max_count:
                continue

            comment = comment.strip()
            if comment[-1] != '.':
                comment += '.'

            new_comments.append(comment)

        if len(new_comments) >= 1:

            text = ' '.join(new_comments)
            if len(text) < 800:
                texts.append(text)
                labels.append(instance['query'])

    return texts, labels

