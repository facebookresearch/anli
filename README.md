# Adversarial NLI

## Paper
[**Adversarial NLI: A New Benchmark for Natural Language Understanding**](https://arxiv.org/abs/1910.14599)

## Dataset
Version 1.0 is available here: https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip.

## Leaderboard

If you want to have your model added to the leaderboard, please reach out to us or submit a PR.

Model | Publication | A1 | A2 | A3
---|---|---|---|---
InfoBERT (RoBERTa Large) | [Wang et al., 2020](https://arxiv.org/pdf/2010.02329.pdf) | 75.0 | 50.5 | 47.7
ALUM (RoBERTa Large) | [Liu et al., 2020](https://arxiv.org/abs/2004.08994) | 72.3 | 52.1 | 48.4
GPT-3 | [Brown et al., 2020](https://arxiv.org/abs/2005.14165) | 36.8 | 34.0 | 40.2
XLNet Large | [Yang et al., 2019](https://arxiv.org/abs/1906.08237) | 70.3 | 50.9 | 49.4
RoBERTa Large | [Liu et al., 2019](https://arxiv.org/abs/1907.11692) | 72.4 | 49.8 | 44.4

## Implementation

To facilitate research in the field of NLI, we provide an easy-to-use codebase for NLI data preparation and modeling.
The code is built upon [Transformers](https://huggingface.co/transformers/) with a special focus on NLI.

We welcome researchers from various fields (linguistics, machine learning, cognitive science, psychology, etc.) to try NLI. 
You can use the code to reproduce the results in our paper or even as a starting point for your research.

Please read more in [**Start your NLI research**](mds/start_your_nli_research.md).  

An important detail in our experiments is that we combine SNLI+MNLI+FEVER-NLI and up-sample different rounds of ANLI to train the models.  
**We highly recommend you refer to the above link for reproducing the results and training your models such that the results will be comparable to the ones on the leaderboard.**

## Pre-trained Models
Pre-trained NLI models can be easily called through huggingface model hub.  

Version information:
```
python==3.7
torch==1.7
transformers==3.0.2 or later (tested: 3.0.2, 3.1.0, 4.0.0)
```

Models: `RoBERTa`, `ALBert`, `BART`, `ELECTRA`, `XLNet`.  

The training data is a combination of [`SNLI`](https://nlp.stanford.edu/projects/snli/), [`MNLI`](https://cims.nyu.edu/~sbowman/multinli/), [`FEVER-NLI`](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md), [`ANLI (R1, R2, R3)`](https://github.com/facebookresearch/anli). Please also cite the datasets if you are using the pre-trained model.  
  
Please try the code snippet below.
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

if __name__ == '__main__':
    max_length = 256

    premise = "Two women are embracing while holding to go packages."
    hypothesis = "The men are fighting outside a deli."

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)
    # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contradiction"
    # },

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    print("Premise:", premise)
    print("Hypothesis:", hypothesis)
    print("Entailment:", predicted_probability[0])
    print("Neutral:", predicted_probability[1])
    print("Contradiction:", predicted_probability[2])
```

More in [here](https://github.com/facebookresearch/anli/blob/master/src/hg_api/interactive_eval.py).

## Rules

When using this dataset, we ask that you obey some very simple rules:

1. We want to make it easy for people to provide ablations on test sets without being rate limited, so we release labeled test sets with this distribution. We trust that you will act in good faith, and will not tune on the test set (this should really go without saying)! We may release unlabeled test sets later.

2. **Training data is for training, development data is for development, and test data is for reporting test numbers.** This means that you should not e.g. train on the train+dev data from rounds 1 and 2 and then report an increase in performance on the test set of round 3.

3. We will host a leaderboard on this page. If you want to be added to the leaderboard, please contact us and/or submit a PR with a link to your paper, a link to your code in a public repository (e.g. Github), together with the following information: number of parameters in your model, data used for (pre-)training, and your dev and test results for *each* round, as well as the total over *all* rounds.

## Reason

AdversarialNLI dataset contains a reason field for each examples in the `dev` and `test` split and for some examples in the `train` split. The reason is collected by asking annotator "Please write a reason for your statement belonging to the category and why you think it was difficult for the system.". 

## Other NLI Reference

We used following NLI resources in training the backend model of the adversarial collection:
- [**SNLI**](https://nlp.stanford.edu/projects/snli/)
- [**MultiNLI**](https://www.nyu.edu/projects/bowman/multinli/)
- [**NLI style FEVER**](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md)

## Citation
```
@inproceedings{nie-etal-2020-adversarial,
    title = "Adversarial {NLI}: A New Benchmark for Natural Language Understanding",
    author = "Nie, Yixin  and
      Williams, Adina  and
      Dinan, Emily  and
      Bansal, Mohit  and
      Weston, Jason  and
      Kiela, Douwe",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```

## License
ANLI is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.
