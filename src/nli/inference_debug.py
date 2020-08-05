# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
import uuid
import numpy as np

import config
from flint.data_utils.batchbuilder import move_to_device
from flint.data_utils.fields import RawFlintField, LabelFlintField, ArrayIndexFlintField
from utils import common, list_dict_data_tool, save_tool
from src.nli.training import MODEL_CLASSES, registered_path, build_eval_dataset_loader_and_sampler, NLITransform, \
    NLIDataset, count_acc, evaluation_dataset, eval_model

import torch

import pprint

pp = pprint.PrettyPrinter(indent=2)


id2label = {
    0: 'e',
    1: 'n',
    2: 'c',
    -1: '-',
}


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(np.asarray(x) - np.max(x))
    return e_x / e_x.sum()


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
                                labels=None)
            else:
                outputs = model(batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                token_type_ids=batch['token_type_ids'],
                                labels=None)

            # print(outputs)
            logits = outputs[0]

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
        r_item['probability'] = softmax(r_item['logits'])
        r_item['predicted_label'] = id2label[pred_list[i]]

        result_items_list.append(r_item)

    return result_items_list


def inference(model_class_name, model_checkpoint_path, max_length, premise, hypothesis, cpu=True):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # CPU for now
    if cpu:
        args.global_rank = -1
    else:
        args.global_rank = 0

    model_checkpoint_path = model_checkpoint_path
    args.model_class_name = model_class_name
    num_labels = 3
    # we are doing NLI so we set num_labels = 3, for other task we can change this value.

    max_length = max_length

    model_class_item = MODEL_CLASSES[model_class_name]
    model_name = model_class_item['model_name']
    do_lower_case = model_class_item['do_lower_case'] if 'do_lower_case' in model_class_item else False

    tokenizer = model_class_item['tokenizer'].from_pretrained(model_name,
                                                              cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                              do_lower_case=do_lower_case)

    model = model_class_item['sequence_classification'].from_pretrained(model_name,
                                                                        cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                                        num_labels=num_labels)

    model.load_state_dict(torch.load(model_checkpoint_path))

    padding_token_value = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_segement_value = model_class_item["padding_segement_value"]
    padding_att_value = model_class_item["padding_att_value"]
    left_pad = model_class_item['left_pad'] if 'left_pad' in model_class_item else False

    batch_size_per_gpu_eval = 16

    eval_data_list = [{
        'uid': str(uuid.uuid4()),
        'premise': premise,
        'hypothesis': hypothesis,
        'label': 'h'    # hidden
    }]

    batching_schema = {
        'uid': RawFlintField(),
        'y': LabelFlintField(),
        'input_ids': ArrayIndexFlintField(pad_idx=padding_token_value, left_pad=left_pad),
        'token_type_ids': ArrayIndexFlintField(pad_idx=padding_segement_value, left_pad=left_pad),
        'attention_mask': ArrayIndexFlintField(pad_idx=padding_att_value, left_pad=left_pad),
    }

    data_transformer = NLITransform(model_name, tokenizer, max_length)

    d_dataset, d_sampler, d_dataloader = build_eval_dataset_loader_and_sampler(eval_data_list, data_transformer,
                                                                               batching_schema,
                                                                               batch_size_per_gpu_eval)

    if not cpu:
        torch.cuda.set_device(0)
        model.cuda(0)

    pred_output_list = eval_model(model, d_dataloader, args.global_rank, args)
    # r_dict = dict()
    # Eval loop:
    # print(pred_output_list)
    return pred_output_list[0]


if __name__ == '__main__':
    # model_class_name = "roberta-large"
    # model_checkpoint_path = config.PRO_ROOT / "saved_models/06-29-22:16:24_roberta-large|snli+mnli+fnli+r1*10+r2*20+r3*10|nli/checkpoints/e(0)|i(22000)|snli_dev#(0.9255)|mnli_m_dev#(0.8951)|mnli_mm_dev#(0.8993)|anli_r1_dev#(0.744)|anli_r1_test#(0.735)|anli_r2_dev#(0.489)|anli_r2_test#(0.497)|anli_r3_dev#(0.47)|anli_r3_test#(0.4442)/model.pt"

    # model_class_name = "xlnet-large"
    # model_checkpoint_path = config.PRO_ROOT / ""
    #
    # model_class_name = "bart-large"
    # model_checkpoint_path = config.PRO_ROOT / "saved_models/06-30-08:23:44_bart-large|snli+mnli+fnli+r1*10+r2*20+r3*10|nli/checkpoints/e(0)|i(25264)|snli_dev#(0.9302)|mnli_m_dev#(0.8985)|mnli_mm_dev#(0.8966)|anli_r1_dev#(0.723)|anli_r1_test#(0.713)|anli_r2_dev#(0.528)|anli_r2_test#(0.502)|anli_r3_dev#(0.5125)|anli_r3_test#(0.4992)/model.pt"
    #
    model_class_name = "electra-large"
    model_checkpoint_path = config.PRO_ROOT / "saved_models/08-02-08:58:05_electra-large|snli+mnli+fnli+r1*10+r2*20+r3*10|nli/checkpoints/e(0)|i(12000)|snli_dev#(0.9168)|mnli_m_dev#(0.8597)|mnli_mm_dev#(0.8661)|anli_r1_dev#(0.672)|anli_r1_test#(0.678)|anli_r2_dev#(0.536)|anli_r2_test#(0.522)|anli_r3_dev#(0.55)|anli_r3_test#(0.5217)/model.pt"

    max_length = 156

    premise = "Two women are embracing while holding to go packages."
    hypothesis = "The men are fighting outside a deli."

    pred_output = inference(model_class_name, model_checkpoint_path, max_length, premise, hypothesis, cpu=True)
    print(pred_output)