# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
# from transformers import XLNetTokenizer
# from modeling.dummy_modeling_xlnet import XLNetForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BartTokenizer, BartForSequenceClassification
from transformers import ElectraTokenizer, ElectraForSequenceClassification

from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import config
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from flint.data_utils.batchbuilder import BaseBatchBuilder, move_to_device
from flint.data_utils.fields import RawFlintField, LabelFlintField, ArrayIndexFlintField
from utils import common, list_dict_data_tool, save_tool
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

import numpy as np
import random
import torch
from tqdm import tqdm
import math
import copy

import pprint

pp = pprint.PrettyPrinter(indent=2)

# from fairseq.data.data_utils import collate_tokens

MODEL_CLASSES = {
    "bert-base": {
        "model_name": "bert-base-uncased",
        "tokenizer": BertTokenizer,
        "sequence_classification": BertForSequenceClassification,
        # "padding_token_value": 0,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "do_lower_case": True,
    },
    "bert-large": {
        "model_name": "bert-large-uncased",
        "tokenizer": BertTokenizer,
        "sequence_classification": BertForSequenceClassification,
        # "padding_token_value": 0,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "do_lower_case": True,
        "internal_model_name": "bert",
        'insight_supported': True,
    },

    "xlnet-base": {
        "model_name": "xlnet-base-cased",
        "tokenizer": XLNetTokenizer,
        "sequence_classification": XLNetForSequenceClassification,
        # "padding_token_value": 0,
        "padding_segement_value": 4,
        "padding_att_value": 0,
        "left_pad": True,
        "internal_model_name": ["transformer", "word_embedding"],
    },
    "xlnet-large": {
        "model_name": "xlnet-large-cased",
        "tokenizer": XLNetTokenizer,
        "sequence_classification": XLNetForSequenceClassification,
        "padding_segement_value": 4,
        "padding_att_value": 0,
        "left_pad": True,
        "internal_model_name": ["transformer", "word_embedding"],
        'insight_supported': True,
    },

    "roberta-base": {
        "model_name": "roberta-base",
        "tokenizer": RobertaTokenizer,
        "sequence_classification": RobertaForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "internal_model_name": "roberta",
        'insight_supported': True,
    },
    "roberta-large": {
        "model_name": "roberta-large",
        "tokenizer": RobertaTokenizer,
        "sequence_classification": RobertaForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "internal_model_name": "roberta",
        'insight_supported': True,
    },

    "albert-xxlarge": {
        "model_name": "albert-xxlarge-v2",
        "tokenizer": AlbertTokenizer,
        "sequence_classification": AlbertForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "do_lower_case": True,
        "internal_model_name": "albert",
        'insight_supported': True,
    },

    "distilbert": {
        "model_name": "distilbert-base-cased",
        "tokenizer": DistilBertTokenizer,
        "sequence_classification": DistilBertForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
    },

    "bart-large": {
        "model_name": "facebook/bart-large",
        "tokenizer": BartTokenizer,
        "sequence_classification": BartForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "internal_model_name": ["model", "encoder", "embed_tokens"],
        'insight_supported': True,
    },

    "electra-base": {
        "model_name": "google/electra-base-discriminator",
        "tokenizer": ElectraTokenizer,
        "sequence_classification": ElectraForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "internal_model_name": "electra",
        'insight_supported': True,
    },

    "electra-large": {
        "model_name": "google/electra-large-discriminator",
        "tokenizer": ElectraTokenizer,
        "sequence_classification": ElectraForSequenceClassification,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "internal_model_name": "electra",
        'insight_supported': True,
    }
}

registered_path = {
    'snli_train': config.PRO_ROOT / "data/build/snli/train.jsonl",
    'snli_dev': config.PRO_ROOT / "data/build/snli/dev.jsonl",
    'snli_test': config.PRO_ROOT / "data/build/snli/test.jsonl",

    'mnli_train': config.PRO_ROOT / "data/build/mnli/train.jsonl",
    'mnli_m_dev': config.PRO_ROOT / "data/build/mnli/m_dev.jsonl",
    'mnli_mm_dev': config.PRO_ROOT / "data/build/mnli/mm_dev.jsonl",

    'fever_train': config.PRO_ROOT / "data/build/fever_nli/train.jsonl",
    'fever_dev': config.PRO_ROOT / "data/build/fever_nli/dev.jsonl",
    'fever_test': config.PRO_ROOT / "data/build/fever_nli/test.jsonl",

    'anli_r1_train': config.PRO_ROOT / "data/build/anli/r1/train.jsonl",
    'anli_r1_dev': config.PRO_ROOT / "data/build/anli/r1/dev.jsonl",
    'anli_r1_test': config.PRO_ROOT / "data/build/anli/r1/test.jsonl",

    'anli_r2_train': config.PRO_ROOT / "data/build/anli/r2/train.jsonl",
    'anli_r2_dev': config.PRO_ROOT / "data/build/anli/r2/dev.jsonl",
    'anli_r2_test': config.PRO_ROOT / "data/build/anli/r2/test.jsonl",

    'anli_r3_train': config.PRO_ROOT / "data/build/anli/r3/train.jsonl",
    'anli_r3_dev': config.PRO_ROOT / "data/build/anli/r3/dev.jsonl",
    'anli_r3_test': config.PRO_ROOT / "data/build/anli/r3/test.jsonl",
}

nli_label2index = {
    'e': 0,
    'n': 1,
    'c': 2,
    'h': -1,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NLIDataset(Dataset):
    def __init__(self, data_list, transform) -> None:
        super().__init__()
        self.d_list = data_list
        self.len = len(self.d_list)
        self.transform = transform

    def __getitem__(self, index: int):
        return self.transform(self.d_list[index])

    # you should write schema for each of the input elements

    def __len__(self) -> int:
        return self.len


class NLITransform(object):
    def __init__(self, model_name, tokenizer, max_length=None):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, sample):
        processed_sample = dict()
        processed_sample['uid'] = sample['uid']
        processed_sample['gold_label'] = sample['label']
        processed_sample['y'] = nli_label2index[sample['label']]

        # premise: str = sample['premise']
        premise: str = sample['context'] if 'context' in sample else sample['premise']
        hypothesis: str = sample['hypothesis']

        if premise.strip() == '':
            premise = 'empty'

        if hypothesis.strip() == '':
            hypothesis = 'empty'

        tokenized_input_seq_pair = self.tokenizer.encode_plus(premise, hypothesis,
                                                              max_length=self.max_length,
                                                              return_token_type_ids=True, truncation=True)

        processed_sample.update(tokenized_input_seq_pair)

        return processed_sample


def build_eval_dataset_loader_and_sampler(d_list, data_transformer, batching_schema, batch_size_per_gpu_eval):
    d_dataset = NLIDataset(d_list, data_transformer)
    d_sampler = SequentialSampler(d_dataset)
    d_dataloader = DataLoader(dataset=d_dataset,
                              batch_size=batch_size_per_gpu_eval,
                              shuffle=False,  #
                              num_workers=0,
                              pin_memory=True,
                              sampler=d_sampler,
                              collate_fn=BaseBatchBuilder(batching_schema))  #
    return d_dataset, d_sampler, d_dataloader


def sample_data_list(d_list, ratio):
    if ratio <= 0:
        raise ValueError("Invalid training weight ratio. Please change --train_weights.")
    upper_int = int(math.ceil(ratio))
    if upper_int == 1:
        return d_list # if ratio is 1 then we just return the data list
    else:
        sampled_d_list = []
        for _ in range(upper_int):
            sampled_d_list.extend(copy.deepcopy(d_list))
        if np.isclose(ratio, upper_int):
            return sampled_d_list
        else:
            sampled_length = int(ratio * len(d_list))
            random.shuffle(sampled_d_list)
            return sampled_d_list[:sampled_length]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="If set, we only use CPU.")
    parser.add_argument("--single_gpu", action="store_true", help="If set, we only use single GPU.")
    parser.add_argument("--fp16", action="store_true", help="If set, we will use fp16.")

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    # environment arguments
    parser.add_argument('-s', '--seed', default=1, type=int, metavar='N',
                        help='manual random seed')
    parser.add_argument('-n', '--num_nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('-g', '--gpus_per_node', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--node_rank', default=0, type=int,
                        help='ranking within the nodes')

    # experiments specific arguments
    parser.add_argument('--debug_mode',
                        action='store_true',
                        dest='debug_mode',
                        help='weather this is debug mode or normal')

    parser.add_argument(
        "--model_class_name",
        type=str,
        help="Set the model class of the experiment.",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Set the name of the experiment. [model_name]/[data]/[task]/[other]",
    )

    parser.add_argument(
        "--save_prediction",
        action='store_true',
        dest='save_prediction',
        help='Do we want to save prediction')

    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="If we want to resume model training, we need to set the resume path to restore state dicts.",
    )
    parser.add_argument(
        "--global_iteration",
        type=int,
        default=0,
        help="This argument is only used if we resume model training.",
    )

    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--total_step', default=-1, type=int, metavar='N',
                        help='number of step to update, default calculate with total data size.'
                             'if we set this step, then epochs will be 100 to run forever.')

    parser.add_argument('--sampler_seed', default=-1, type=int, metavar='N',
                        help='The seed the controls the data sampling order.')

    parser.add_argument(
        "--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument("--max_length", default=160, type=int, help="Max length of the sequences.")

    parser.add_argument("--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument(
        "--eval_frequency", default=1000, type=int, help="set the evaluation frequency, evaluate every X global step.",
    )

    parser.add_argument("--train_data",
                        type=str,
                        help="The training data used in the experiments.")

    parser.add_argument("--train_weights",
                        type=str,
                        help="The training data weights used in the experiments.")

    parser.add_argument("--eval_data",
                        type=str,
                        help="The training data used in the experiments.")

    args = parser.parse_args()

    if args.cpu:
        args.world_size = 1
        train(-1, args)
    elif args.single_gpu:
        args.world_size = 1
        train(0, args)
    else:  # distributed multiGPU training
        #########################################################
        args.world_size = args.gpus_per_node * args.num_nodes  #
        # os.environ['MASTER_ADDR'] = '152.2.142.184'  # This is the IP address for nlp5
        # maybe we will automatically retrieve the IP later.
        os.environ['MASTER_PORT'] = '88888'  #
        mp.spawn(train, nprocs=args.gpus_per_node, args=(args,))  # spawn how many process in this node
        # remember train is called as train(i, args).
        #########################################################


def train(local_rank, args):
    # debug = False
    # print("GPU:", gpu)
    # world_size = args.world_size
    args.global_rank = args.node_rank * args.gpus_per_node + local_rank
    args.local_rank = local_rank
    # args.warmup_steps = 20
    debug_count = 1000

    if args.total_step > 0:
        num_epoch = 10000  # if we set total step, num_epoch will be forever.
    else:
        num_epoch = args.epochs

    actual_train_batch_size = args.world_size * args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    args.actual_train_batch_size = actual_train_batch_size

    set_seed(args.seed)
    num_labels = 3      # we are doing NLI so we set num_labels = 3, for other task we can change this value.

    max_length = args.max_length

    model_class_item = MODEL_CLASSES[args.model_class_name]
    model_name = model_class_item['model_name']
    do_lower_case = model_class_item['do_lower_case'] if 'do_lower_case' in model_class_item else False

    tokenizer = model_class_item['tokenizer'].from_pretrained(model_name,
                                                              cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                              do_lower_case=do_lower_case)

    model = model_class_item['sequence_classification'].from_pretrained(model_name,
                                                                        cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                                        num_labels=num_labels)

    padding_token_value = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_segement_value = model_class_item["padding_segement_value"]
    padding_att_value = model_class_item["padding_att_value"]
    left_pad = model_class_item['left_pad'] if 'left_pad' in model_class_item else False

    batch_size_per_gpu_train = args.per_gpu_train_batch_size
    batch_size_per_gpu_eval = args.per_gpu_eval_batch_size

    if not args.cpu and not args.single_gpu:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.global_rank
        )

    train_data_str = args.train_data
    train_data_weights_str = args.train_weights
    eval_data_str = args.eval_data

    train_data_name = []
    train_data_path = []
    train_data_list = []
    train_data_weights = []

    eval_data_name = []
    eval_data_path = []
    eval_data_list = []

    train_data_named_path = train_data_str.split(',')
    weights_str = train_data_weights_str.split(',') if train_data_weights_str is not None else None

    eval_data_named_path = eval_data_str.split(',')

    for named_path in train_data_named_path:
        ind = named_path.find(':')
        name = named_path[:ind]
        path = named_path[ind + 1:]
        if name in registered_path:
            d_list = common.load_jsonl(registered_path[name])
        else:
            d_list = common.load_jsonl(path)

        train_data_name.append(name)
        train_data_path.append(path)

        train_data_list.append(d_list)

    if weights_str is not None:
        for weights in weights_str:
            train_data_weights.append(float(weights))
    else:
        for i in range(len(train_data_list)):
            train_data_weights.append(1)

    for named_path in eval_data_named_path:
        ind = named_path.find(':')
        name = named_path[:ind]
        path = named_path[ind + 1:]
        if name in registered_path:
            d_list = common.load_jsonl(registered_path[name])
        else:
            d_list = common.load_jsonl(path)
        eval_data_name.append(name)
        eval_data_path.append(path)

        eval_data_list.append(d_list)

    assert len(train_data_weights) == len(train_data_list)

    batching_schema = {
        'uid': RawFlintField(),
        'y': LabelFlintField(),
        'input_ids': ArrayIndexFlintField(pad_idx=padding_token_value, left_pad=left_pad),
        'token_type_ids': ArrayIndexFlintField(pad_idx=padding_segement_value, left_pad=left_pad),
        'attention_mask': ArrayIndexFlintField(pad_idx=padding_att_value, left_pad=left_pad),
    }

    data_transformer = NLITransform(model_name, tokenizer, max_length)
    # data_transformer = NLITransform(model_name, tokenizer, max_length, with_element=True)

    eval_data_loaders = []
    for eval_d_list in eval_data_list:
        d_dataset, d_sampler, d_dataloader = build_eval_dataset_loader_and_sampler(eval_d_list, data_transformer,
                                                                                   batching_schema,
                                                                                   batch_size_per_gpu_eval)
        eval_data_loaders.append(d_dataloader)

    # Estimate the training size:
    training_list = []
    for i in range(len(train_data_list)):
        print("Build Training Data ...")
        train_d_list = train_data_list[i]
        train_d_name = train_data_name[i]
        train_d_weight = train_data_weights[i]
        cur_train_list = sample_data_list(train_d_list, train_d_weight)  # change later  # we can apply different sample strategy here.
        print(f"Data Name:{train_d_name}; Weight: {train_d_weight}; "
              f"Original Size: {len(train_d_list)}; Sampled Size: {len(cur_train_list)}")
        training_list.extend(cur_train_list)
    estimated_training_size = len(training_list)
    print("Estimated training size:", estimated_training_size)
    # Estimate the training size ends:

    # t_total = estimated_training_size // args.gradient_accumulation_steps * num_epoch
    # t_total = estimated_training_size * num_epoch // args.actual_train_batch_size
    if args.total_step <= 0:
        t_total = estimated_training_size * num_epoch // args.actual_train_batch_size
    else:
        t_total = args.total_step

    if args.warmup_steps <= 0:  # set the warmup steps to 0.1 * total step if the given warmup step is -1.
        args.warmup_steps = int(t_total * 0.1)

    if not args.cpu:
        torch.cuda.set_device(args.local_rank)
        model.cuda(args.local_rank)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    global_step = 0

    if args.resume_path:
        print("Resume Training")
        global_step = args.global_iteration
        print("Resume Global Step: ", global_step)
        model.load_state_dict(torch.load(str(Path(args.resume_path) / "model.pt"), map_location=torch.device('cpu')))
        optimizer.load_state_dict(torch.load(str(Path(args.resume_path) / "optimizer.pt"), map_location=torch.device('cpu')))
        scheduler.load_state_dict(torch.load(str(Path(args.resume_path) / "scheduler.pt"), map_location=torch.device('cpu')))
        print("State Resumed")

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if not args.cpu and not args.single_gpu:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                    output_device=local_rank, find_unused_parameters=True)

    args_dict = dict(vars(args))
    file_path_prefix = '.'
    if args.global_rank in [-1, 0]:
        print("Total Steps:", t_total)
        args.total_step = t_total
        print("Warmup Steps:", args.warmup_steps)
        print("Actual Training Batch Size:", actual_train_batch_size)
        print("Arguments", pp.pprint(args))

    is_finished = False

    # Let build the logger and log everything before the start of the first training epoch.
    if args.global_rank in [-1, 0]:  # only do logging if we use cpu or global_rank=0
        resume_prefix = ""
        # if args.resume_path:
        #     resume_prefix = "resumed_"

        if not args.debug_mode:
            file_path_prefix, date = save_tool.gen_file_prefix(f"{args.experiment_name}")
            # # # Create Log File
            # Save the source code.
            script_name = os.path.basename(__file__)
            with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
                out_f.write(it.read())
                out_f.flush()

            # Save option file
            common.save_json(args_dict, os.path.join(file_path_prefix, "args.json"))
            checkpoints_path = Path(file_path_prefix) / "checkpoints"
            if not checkpoints_path.exists():
                checkpoints_path.mkdir()
            prediction_path = Path(file_path_prefix) / "predictions"
            if not prediction_path.exists():
                prediction_path.mkdir()

            # if this is a resumed, then we save the resumed path.
            if args.resume_path:
                with open(os.path.join(file_path_prefix, "resume_log.txt"), 'w') as out_f:
                    out_f.write(str(args.resume_path))
                    out_f.flush()

    # print(f"Global Rank:{args.global_rank} ### ", 'Init!')

    for epoch in tqdm(range(num_epoch), desc="Epoch", disable=args.global_rank not in [-1, 0]):
        # Let's build up training dataset for this epoch
        training_list = []
        for i in range(len(train_data_list)):
            print("Build Training Data ...")
            train_d_list = train_data_list[i]
            train_d_name = train_data_name[i]
            train_d_weight = train_data_weights[i]
            cur_train_list = sample_data_list(train_d_list, train_d_weight)  # change later  # we can apply different sample strategy here.
            print(f"Data Name:{train_d_name}; Weight: {train_d_weight}; "
                  f"Original Size: {len(train_d_list)}; Sampled Size: {len(cur_train_list)}")
            training_list.extend(cur_train_list)

        random.shuffle(training_list)
        train_dataset = NLIDataset(training_list, data_transformer)

        train_sampler = SequentialSampler(train_dataset)
        if not args.cpu and not args.single_gpu:
            print("Use distributed sampler.")
            train_sampler = DistributedSampler(train_dataset, args.world_size, args.global_rank,
                                               shuffle=True)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size_per_gpu_train,
                                      shuffle=False,  #
                                      num_workers=0,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      collate_fn=BaseBatchBuilder(batching_schema))  #
        # training build finished.

        print(debug_node_info(args), "epoch: ", epoch)

        if not args.cpu and not args.single_gpu:
            if args.sampler_seed == -1:
                train_sampler.set_epoch(epoch)  # setup the epoch to ensure random sampling at each epoch
            else:
                train_sampler.set_epoch(epoch + args.sampler_seed)

        for forward_step, batch in enumerate(tqdm(train_dataloader, desc="Iteration",
                                                  disable=args.global_rank not in [-1, 0]), 0):
            model.train()

            batch = move_to_device(batch, local_rank)
            # print(batch['input_ids'], batch['y'])
            if args.model_class_name in ["distilbert", "bart-large"]:
                outputs = model(batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['y'])
            else:
                outputs = model(batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                token_type_ids=batch['token_type_ids'],
                                labels=batch['y'])
            loss, logits = outputs[:2]
            # print(debug_node_info(args), loss, logits, batch['uid'])
            # print(debug_node_info(args), loss, batch['uid'])

            # Accumulated loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # if this forward step need model updates
            # handle fp16
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

                # Gradient clip: if max_grad_norm < 0
            if (forward_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                global_step += 1

                if args.global_rank in [-1, 0] and args.eval_frequency > 0 and global_step % args.eval_frequency == 0:
                    r_dict = dict()
                    # Eval loop:
                    for i in range(len(eval_data_name)):
                        cur_eval_data_name = eval_data_name[i]
                        cur_eval_data_list = eval_data_list[i]
                        cur_eval_dataloader = eval_data_loaders[i]
                        # cur_eval_raw_data_list = eval_raw_data_list[i]

                        evaluation_dataset(args, cur_eval_dataloader, cur_eval_data_list, model, r_dict,
                                           eval_name=cur_eval_data_name)

                    # saving checkpoints
                    current_checkpoint_filename = \
                        f'e({epoch})|i({global_step})'

                    for i in range(len(eval_data_name)):
                        cur_eval_data_name = eval_data_name[i]
                        current_checkpoint_filename += \
                            f'|{cur_eval_data_name}#({round(r_dict[cur_eval_data_name]["acc"], 4)})'

                    if not args.debug_mode:
                        # save model:
                        model_output_dir = checkpoints_path / current_checkpoint_filename
                        if not model_output_dir.exists():
                            model_output_dir.mkdir()
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training

                        torch.save(model_to_save.state_dict(), str(model_output_dir / "model.pt"))
                        torch.save(optimizer.state_dict(), str(model_output_dir / "optimizer.pt"))
                        torch.save(scheduler.state_dict(), str(model_output_dir / "scheduler.pt"))

                    # save prediction:
                    if not args.debug_mode and args.save_prediction:
                        cur_results_path = prediction_path / current_checkpoint_filename
                        if not cur_results_path.exists():
                            cur_results_path.mkdir(parents=True)
                        for key, item in r_dict.items():
                            common.save_jsonl(item['predictions'], cur_results_path / f"{key}.jsonl")

                        # avoid saving too many things
                        for key, item in r_dict.items():
                            del r_dict[key]['predictions']
                        common.save_json(r_dict, cur_results_path / "results_dict.json", indent=2)

                if args.total_step > 0 and global_step == t_total:
                    # if we set total step and global step s t_total.
                    is_finished = True
                    break

        # End of epoch evaluation.
        if args.global_rank in [-1, 0] and args.total_step <= 0:
            r_dict = dict()
            # Eval loop:
            for i in range(len(eval_data_name)):
                cur_eval_data_name = eval_data_name[i]
                cur_eval_data_list = eval_data_list[i]
                cur_eval_dataloader = eval_data_loaders[i]
                # cur_eval_raw_data_list = eval_raw_data_list[i]

                evaluation_dataset(args, cur_eval_dataloader, cur_eval_data_list, model, r_dict,
                                   eval_name=cur_eval_data_name)

            # saving checkpoints
            current_checkpoint_filename = \
                f'e({epoch})|i({global_step})'

            for i in range(len(eval_data_name)):
                cur_eval_data_name = eval_data_name[i]
                current_checkpoint_filename += \
                    f'|{cur_eval_data_name}#({round(r_dict[cur_eval_data_name]["acc"], 4)})'

            if not args.debug_mode:
                # save model:
                model_output_dir = checkpoints_path / current_checkpoint_filename
                if not model_output_dir.exists():
                    model_output_dir.mkdir()
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training

                torch.save(model_to_save.state_dict(), str(model_output_dir / "model.pt"))
                torch.save(optimizer.state_dict(), str(model_output_dir / "optimizer.pt"))
                torch.save(scheduler.state_dict(), str(model_output_dir / "scheduler.pt"))

            # save prediction:
            if not args.debug_mode and args.save_prediction:
                cur_results_path = prediction_path / current_checkpoint_filename
                if not cur_results_path.exists():
                    cur_results_path.mkdir(parents=True)
                for key, item in r_dict.items():
                    common.save_jsonl(item['predictions'], cur_results_path / f"{key}.jsonl")

                # avoid saving too many things
                for key, item in r_dict.items():
                    del r_dict[key]['predictions']
                common.save_json(r_dict, cur_results_path / "results_dict.json", indent=2)

        if is_finished:
            break


id2label = {
    0: 'e',
    1: 'n',
    2: 'c',
    -1: '-',
}


def count_acc(gt_list, pred_list):
    assert len(gt_list) == len(pred_list)
    gt_dict = list_dict_data_tool.list_to_dict(gt_list, 'uid')
    pred_list = list_dict_data_tool.list_to_dict(pred_list, 'uid')
    total_count = 0
    hit = 0
    for key, value in pred_list.items():
        if gt_dict[key]['label'] == value['predicted_label']:
            hit += 1
        total_count += 1
    return hit, total_count


def evaluation_dataset(args, eval_dataloader, eval_list, model, r_dict, eval_name):
    # r_dict = dict()
    pred_output_list = eval_model(model, eval_dataloader, args.global_rank, args)
    predictions = pred_output_list
    hit, total = count_acc(eval_list, pred_output_list)

    print(debug_node_info(args), f"{eval_name} Acc:", hit, total, hit / total)

    r_dict[f'{eval_name}'] = {
        'acc': hit / total,
        'correct_count': hit,
        'total_count': total,
        'predictions': predictions,
    }


def eval_model(model, dev_dataloader, device_num, args):
    model.eval()

    uid_list = []
    y_list = []
    pred_list = []
    logits_list = []

    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader, 0):
            batch = move_to_device(batch, device_num)

            if args.model_class_name in ["distilbert", "bart-large"]:
                outputs = model(batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['y'])
            else:
                outputs = model(batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                token_type_ids=batch['token_type_ids'],
                                labels=batch['y'])

            loss, logits = outputs[:2]

            uid_list.extend(list(batch['uid']))
            y_list.extend(batch['y'].tolist())
            pred_list.extend(torch.max(logits, 1)[1].view(logits.size(0)).tolist())
            logits_list.extend(logits.tolist())

    assert len(pred_list) == len(logits_list)
    assert len(pred_list) == len(logits_list)

    result_items_list = []
    for i in range(len(uid_list)):
        r_item = dict()
        r_item['uid'] = uid_list[i]
        r_item['logits'] = logits_list[i]
        r_item['predicted_label'] = id2label[pred_list[i]]

        result_items_list.append(r_item)

    return result_items_list


def debug_node_info(args):
    names = ['global_rank', 'local_rank', 'node_rank']
    values = []

    for name in names:
        if name in args:
            values.append(getattr(args, name))
        else:
            return "Pro:No node info "

    return "Pro:" + '|'.join([f"{name}:{value}" for name, value in zip(names, values)]) + "||Print:"


if __name__ == '__main__':
    main()