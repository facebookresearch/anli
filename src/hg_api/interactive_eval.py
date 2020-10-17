# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json


def get_prediction(tokenizer, model, premise, hypothesis, max_length=256):
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)

    predicted_probability = torch.softmax(outputs[0], dim=1)[0]  # batch_size only one
    predicted_index = torch.argmax(predicted_probability)
    predicted_probability = predicted_probability.tolist()

    return predicted_probability, predicted_index


if __name__ == '__main__':
    premise = "Two women are embracing while holding to go packages."
    hypothesis = "The men are fighting outside a deli."

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    snli_dev = []
    SNLI_DEV_FILE_PATH = "../../data/snli_1.0/snli_1.0_dev.jsonl"   # you can change this to other path.
    with open(SNLI_DEV_FILE_PATH, mode='r', encoding='utf-8') as in_f:
        for line in in_f:
            if line:
                cur_item = json.loads(line)
                if cur_item['gold_label'] != '-':
                    snli_dev.append(cur_item)

    total = 0
    correct = 0
    label_mapping = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction',
    }

    print("Start evaluating...")        # this might take a while.
    for item in snli_dev:
        _, pred_index = get_prediction(tokenizer, model, item['sentence1'], item['sentence2'])
        if label_mapping[int(pred_index)] == item['gold_label']:
            correct += 1
        total += 1
        if total % 200 == 0 and total != 0:
            print(f"{total} finished.")

    print("Total / Correct / Accuracy:", f"{total} / {correct} / {correct / total}")