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
from nli.training import MODEL_CLASSES, registered_path, build_eval_dataset_loader_and_sampler, NLITransform, \
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
    # model_checkpoint_path = config.PRO_ROOT / "saved_models/06-29-22:16:24_roberta-large|snli+mnli+fnli+r1*10+r2*20+r3*10|nli/checkpoints/e(0)|i(24000)|snli_dev#(0.9252)|mnli_m_dev#(0.899)|mnli_mm_dev#(0.9002)|anli_r1_dev#(0.74)|anli_r1_test#(0.742)|anli_r2_dev#(0.506)|anli_r2_test#(0.498)|anli_r3_dev#(0.4667)|anli_r3_test#(0.455)/model.pt"

    # model_class_name = "xlnet-large"
    # model_checkpoint_path = config.PRO_ROOT / "saved_models/06-29-23:04:33_xlnet-large|snli+mnli+fnli+r1*10+r2*20+r3*10|nli/checkpoints/e(1)|i(30000)|snli_dev#(0.9274)|mnli_m_dev#(0.8981)|mnli_mm_dev#(0.8947)|anli_r1_dev#(0.735)|anli_r1_test#(0.701)|anli_r2_dev#(0.521)|anli_r2_test#(0.514)|anli_r3_dev#(0.5075)|anli_r3_test#(0.4975)/model.pt"

    model_class_name = "albert-xxlarge"
    model_checkpoint_path = config.PRO_ROOT / "saved_models/06-29-23:09:03_albert-xxlarge|snli+mnli+fnli+r1*10+r2*20+r3*10|nli/checkpoints/e(0)|i(16000)|snli_dev#(0.9246)|mnli_m_dev#(0.8948)|mnli_mm_dev#(0.8932)|anli_r1_dev#(0.733)|anli_r1_test#(0.711)|anli_r2_dev#(0.571)|anli_r2_test#(0.57)|anli_r3_dev#(0.5817)|anli_r3_test#(0.5375)/model.pt"
    #
    # model_class_name = "bart-large"
    # model_checkpoint_path = config.PRO_ROOT / "saved_models/06-30-08:23:44_bart-large|snli+mnli+fnli+r1*10+r2*20+r3*10|nli/checkpoints/e(1)|i(40000)|snli_dev#(0.9298)|mnli_m_dev#(0.8941)|mnli_mm_dev#(0.8973)|anli_r1_dev#(0.736)|anli_r1_test#(0.72)|anli_r2_dev#(0.533)|anli_r2_test#(0.514)|anli_r3_dev#(0.5058)|anli_r3_test#(0.5042)/model.pt"
    #
    # model_class_name = "electra-large"
    # model_checkpoint_path = config.PRO_ROOT / "saved_models/08-02-08:58:05_electra-large|snli+mnli+fnli+r1*10+r2*20+r3*10|nli/checkpoints/e(0)|i(12000)|snli_dev#(0.9168)|mnli_m_dev#(0.8597)|mnli_mm_dev#(0.8661)|anli_r1_dev#(0.672)|anli_r1_test#(0.678)|anli_r2_dev#(0.536)|anli_r2_test#(0.522)|anli_r3_dev#(0.55)|anli_r3_test#(0.5217)/model.pt"

    max_length = 184

    premise = "Two women are embracing while holding to go packages."
    hypothesis = "The men are fighting outside a deli."

    pred_output = inference(model_class_name, model_checkpoint_path, max_length, premise, hypothesis, cpu=True)
    print(pred_output)
