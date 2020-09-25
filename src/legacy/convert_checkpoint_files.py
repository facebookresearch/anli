from nli.training import MODEL_CLASSES
import argparse
import torch


def hacking_transfer_checkpoint_roberta(model, old_checkpoint_dict):
    new_checkpoint = dict()
    model_state_dict = model.state_dict()

    roberta_new_to_old_mapping = {
        # embeddings:
        'roberta.embeddings.word_embeddings.weight': 'roberta.model.decoder.sentence_encoder.embed_tokens.weight',
        'roberta.embeddings.position_embeddings.weight': 'roberta.model.decoder.sentence_encoder.embed_positions.weight',

        'roberta.embeddings.LayerNorm.weight': 'roberta.model.decoder.sentence_encoder.emb_layer_norm.weight',
        'roberta.embeddings.LayerNorm.bias': 'roberta.model.decoder.sentence_encoder.emb_layer_norm.bias',

        # classifiers:
        'classifier.dense.weight': 'roberta.model.classification_heads.new_task.dense.weight',
        'classifier.dense.bias': 'roberta.model.classification_heads.new_task.dense.bias',
        'classifier.out_proj.weight': 'roberta.model.classification_heads.new_task.out_proj.weight',
        'classifier.out_proj.bias': 'roberta.model.classification_heads.new_task.out_proj.bias',
    }

    number_of_layers = 24
    for i in range(number_of_layers):
        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.self.query.weight'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn.q_proj.weight'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.self.query.bias'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn.q_proj.bias'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.self.key.weight'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn.k_proj.weight'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.self.key.bias'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn.k_proj.bias'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.self.value.weight'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn.v_proj.weight'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.self.value.bias'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn.v_proj.bias'

        # roberta.encoder.layer.{i}.attention.output.dense.weight
        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.output.dense.weight'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn.out_proj.weight'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.output.dense.bias'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn.out_proj.bias'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.output.LayerNorm.weight'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn_layer_norm.weight'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.attention.output.LayerNorm.bias'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.self_attn_layer_norm.bias'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.intermediate.dense.weight'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.fc1.weight'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.intermediate.dense.bias'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.fc1.bias'

        #
        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.output.dense.weight'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.fc2.weight'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.output.dense.bias'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.fc2.bias'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.output.LayerNorm.weight'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.final_layer_norm.weight'

        roberta_new_to_old_mapping[f'roberta.encoder.layer.{i}.output.LayerNorm.bias'] = \
            f'roberta.model.decoder.sentence_encoder.layers.{i}.final_layer_norm.bias'

    for key in model_state_dict:
        if key in roberta_new_to_old_mapping:
            old_key = roberta_new_to_old_mapping[key]
            new_checkpoint[key] = old_checkpoint_dict[old_key]
        else:
            new_checkpoint[key] = model_state_dict[key]
            print(key, " Not Found. Please manually check this.")

    return new_checkpoint


def hacking_transfer_checkpoint_bert(model, old_checkpoint_dict):
    new_checkpoint = dict()
    model_state_dict = model.state_dict()

    roberta_new_to_old_mapping = {
        # embeddings:
        'bert.embeddings.word_embeddings.weight': 'bert_encoder.embeddings.word_embeddings.weight',
        'bert.embeddings.position_embeddings.weight': 'bert_encoder.embeddings.position_embeddings.weight',
        'bert.embeddings.token_type_embeddings.weight': 'bert_encoder.embeddings.token_type_embeddings.weight',

        'bert.embeddings.LayerNorm.weight': 'bert_encoder.embeddings.LayerNorm.weight',
        'bert.embeddings.LayerNorm.bias': 'bert_encoder.embeddings.LayerNorm.bias',

        # classifiers:
        'bert.pooler.dense.weight': 'bert_encoder.pooler.dense.weight',
        'bert.pooler.dense.bias': 'bert_encoder.pooler.dense.bias',
        'classifier.weight': 'classifier.weight',
        'classifier.bias': 'classifier.bias',
    }

    number_of_layers = 24
    for i in range(number_of_layers):
        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.self.query.weight'] = \
            f'bert_encoder.encoder.layer.{i}.attention.self.query.weight'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.self.query.bias'] = \
            f'bert_encoder.encoder.layer.{i}.attention.self.query.bias'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.self.key.weight'] = \
            f'bert_encoder.encoder.layer.{i}.attention.self.key.weight'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.self.key.bias'] = \
            f'bert_encoder.encoder.layer.{i}.attention.self.key.bias'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.self.value.weight'] = \
            f'bert_encoder.encoder.layer.{i}.attention.self.value.weight'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.self.value.bias'] = \
            f'bert_encoder.encoder.layer.{i}.attention.self.value.bias'

        # roberta.encoder.layer.{i}.attention.output.dense.weight
        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.output.dense.weight'] = \
            f'bert_encoder.encoder.layer.{i}.attention.output.dense.weight'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.output.dense.bias'] = \
            f'bert_encoder.encoder.layer.{i}.attention.output.dense.bias'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight'] = \
            f'bert_encoder.encoder.layer.{i}.attention.output.LayerNorm.weight'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias'] = \
            f'bert_encoder.encoder.layer.{i}.attention.output.LayerNorm.bias'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.intermediate.dense.weight'] = \
            f'bert_encoder.encoder.layer.{i}.intermediate.dense.weight'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.intermediate.dense.bias'] = \
            f'bert_encoder.encoder.layer.{i}.intermediate.dense.bias'

        #
        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.output.dense.weight'] = \
            f'bert_encoder.encoder.layer.{i}.output.dense.weight'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.output.dense.bias'] = \
            f'bert_encoder.encoder.layer.{i}.output.dense.bias'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.output.LayerNorm.weight'] = \
            f'bert_encoder.encoder.layer.{i}.output.LayerNorm.weight'

        roberta_new_to_old_mapping[f'bert.encoder.layer.{i}.output.LayerNorm.bias'] = \
            f'bert_encoder.encoder.layer.{i}.output.LayerNorm.bias'

    for key in model_state_dict:
        if key in roberta_new_to_old_mapping:
            old_key = roberta_new_to_old_mapping[key]
            new_checkpoint[key] = old_checkpoint_dict[old_key]
        else:
            new_checkpoint[key] = model_state_dict[key]
            print(key, " Not Found. Please manually check this.")

    return new_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_class_name",
        type=str,
        help="Set the model class of the experiment.",
        required=True
    )

    parser.add_argument(
        "--old_checkpoint_path",
        type=str,
        help='Set the path to load the old checkpoint file.', required=True)

    parser.add_argument(
        "--output_checkpoint_path",
        type=str,
        default=None,
        help='Set the path to save the checkpoint in new file.', required=True)

    args = parser.parse_args()
    num_labels = 3  # for NLI we have three labels.
    model_class_item = MODEL_CLASSES[args.model_class_name]
    model_name = model_class_item['model_name']
    model = model_class_item['sequence_classification'].from_pretrained(model_name, \
                                                                        num_labels=num_labels)

    loaded_state_dict = torch.load(args.old_checkpoint_path, \
                                   map_location=torch.device('cpu'))

    if args.model_class_name == 'bert-large':
        transferred_checkpoint = hacking_transfer_checkpoint_bert(model, loaded_state_dict)
    elif args.model_class_name == 'roberta-large':
        transferred_checkpoint = hacking_transfer_checkpoint_roberta(model, loaded_state_dict)
    else:
        raise ValueError("Model Class Name not supported.")

    torch.save(transferred_checkpoint, args.output_checkpoint_path)


if __name__ == '__main__':
    main()

# the following messages is expected.
# roberta.embeddings.token_type_embeddings.weight  Not Found. Please manually check this.
# roberta.pooler.dense.weight  Not Found. Please manually check this.
# roberta.pooler.dense.bias  Not Found. Please manually check this.