# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.


import torch
import logging
from captum.attr import LayerIntegratedGradients

logger = logging.getLogger(__name__)


def summarize_attributions(attributions):
    """
    Summarises the attribution across multiple runs
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def get_model_prediction(input_ids, attention_mask, token_type_ids, model, model_class_item, with_gradient=False):
    model.eval()

    if not with_gradient:
        with torch.no_grad():
            if model_class_item['model_class_name'] in ["distilbert", "bart-large"]:
                outputs = model(input_ids,
                                attention_mask=attention_mask,
                                labels=None)
            else:
                outputs = model(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=None)
    else:
        if model_class_item['model_class_name'] in ["distilbert", "bart-large"]:
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=None)
        else:
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=None)

    return outputs[0]


def get_lig_object(model, model_class_item):
    insight_supported = model_class_item['insight_supported'] if 'insight_supported' in model_class_item else False
    internal_model_name = model_class_item['internal_model_name']
    lig = None  # default is None.
    if not insight_supported:
        logger.warning(f"Inspection for model '{model_class_item['model_class_name']}' is not supported.")
        return lig

    if isinstance(internal_model_name, list):
        current_layer = model
        for layer_n in internal_model_name:
            current_layer = current_layer.__getattr__(layer_n)
        # print(current_layer)
        lig = LayerIntegratedGradients(get_model_prediction, current_layer)
    else:
        lig = LayerIntegratedGradients(get_model_prediction,
                                       model.__getattr__(internal_model_name).embeddings.word_embeddings)
    return lig


def get_tokenized_input_tokens(tokenizer, token_ids):
    raw_words_list = tokenizer.convert_ids_to_tokens(token_ids)
    string_tokens = [tokenizer.convert_tokens_to_string(word) for word in raw_words_list]
    # still need some cleanup, remove space within tokens
    output_tokens = []
    for t in string_tokens:
        output_tokens.append(t.replace(" ", ""))
    return output_tokens


def cleanup_tokenization_special_tokens(tokens, importance, tokenizer):
    filtered_tokens = []
    filtered_importance = []
    for t, i in zip(tokens, importance):
        if t in tokenizer.all_special_tokens:
            continue
        else:
            filtered_tokens.append(t)
            filtered_importance.append(i)
    return filtered_tokens, filtered_importance
