# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

from utils import common
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict
import config
from pathlib import Path


smnli_label2std_label = defaultdict(lambda: "o")  # o stands for all other label that is invalid.
smnli_label2std_label.update({
    "entailment": "e",
    "neutral": "n",
    "contradiction": "c",
    "hidden": "h",
})

fever_label2std_label = defaultdict(lambda: "o")
fever_label2std_label.update({
    'SUPPORTS': "e",
    'NOT ENOUGH INFO': "n",
    'REFUTES': "c",
    'hidden': "h",
})

anli_label2std_label = defaultdict(lambda: "o")
anli_label2std_label.update({
    'e': "e",
    'n': "n",
    'c': "c",
    'hidden': "h",
})

# standard output format: {uid, premise, hypothesis, label, extra_dataset_related_field.}


def sm_nli2std_format(d_list, filter_invalid=True):
    p_list: List[Dict] = []
    for item in d_list:
        formatted_item: Dict = dict()
        formatted_item['uid']: str = item["pairID"]
        formatted_item['premise']: str = item["sentence1"]
        formatted_item['hypothesis']: str = item["sentence2"]
        formatted_item['label']: str = smnli_label2std_label[item["gold_label"]]
        if filter_invalid and formatted_item['label'] == 'o':
            continue  # Skip example with invalid label.

        p_list.append(formatted_item)
    return p_list


def fever_nli2std_format(d_list, filter_invalid=True):
    p_list: List[Dict] = []
    for item in d_list:
        formatted_item: Dict = dict()
        formatted_item['uid']: str = item["fid"]
        formatted_item['premise']: str = item["context"]
        formatted_item['hypothesis']: str = item["query"]
        formatted_item['label']: str = fever_label2std_label[item["label"]]
        if filter_invalid and formatted_item['label'] == 'o':
            continue  # Skip example with invalid label.

        p_list.append(formatted_item)
    return p_list


def a_nli2std_format(d_list, filter_invalid=True):
    p_list: List[Dict] = []
    for item in d_list:
        formatted_item: Dict = dict()
        formatted_item['uid']: str = item["uid"]
        formatted_item['premise']: str = item["context"]
        formatted_item['hypothesis']: str = item["hypothesis"]
        formatted_item['label']: str = anli_label2std_label[item["label"]]
        formatted_item['reason']: str = item["reason"]
        if filter_invalid and formatted_item['label'] == 'o':
            continue  # Skip example with invalid label.

        p_list.append(formatted_item)
    return p_list


if __name__ == '__main__':
    pass