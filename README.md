# Adversarial NLI

## Papers

### Dataset
[**Adversarial NLI: A New Benchmark for Natural Language Understanding**](https://arxiv.org/abs/1910.14599)

### Annotations of the Dataset for Error Analysis
[**ANLIzing the Adversarial Natural Language Inference Dataset**](https://arxiv.org/abs/2010.12729)

## Dataset
Version 1.0 is available here: https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip.
### Format
The dataset files are all in JSONL format (one JSON per line). Below is one example (in JSON format) with self-explanatory fields.  
Note that each example (each line) in the files contains a `uid` field represents **a unique id** across all the examples in all there rounds of ANLI.
```
{   
    "uid": "8a91e1a2-9a32-4fd9-b1b6-bd2ee2287c8f",
    "premise": "Javier Torres (born May 14, 1988 in Artesia, California) is an undefeated Mexican American professional boxer in the Heavyweight division.
                Torres was the second rated U.S. amateur boxer in the Super Heavyweight division and a member of the Mexican Olympic team.",
    "hypothesis": "Javier was born in Mexico",
    "label": "c",
    "reason": "The paragraph states that Javier was born in the California, US."
}
```

### Reason
AdversarialNLI dataset contains a reason field for each examples in the `dev` and `test` split and for some examples in the `train` split. The reason is collected by asking annotator "Please write a reason for your statement belonging to the category and why you think it was difficult for the system.".



### Verifier Labels (Updated on May 11, 2022)
All the examples in our dev and test sets are verified by 2 or 3 (if the first 2 verifiers do not agree with each other) verifiers. We released additional verifier labels in [`verifier_labels/verifier_labels_R1-3.jsonl`](https://github.com/facebookresearch/anli/blob/main/verifier_labels/verifier_labels_R1-3.jsonl).  
Please refer to the [verifier_labels_readme](https://github.com/facebookresearch/anli/blob/main/mds/verifier_labels.md) or Sec 2.1, Appendix C and Figure 7 in the [ANLI paper](https://arxiv.org/pdf/1910.14599.pdf) for more details about the verifier labels.


### Annotations for Error Analysis

An in-depth error analysis of the dataset is available here: https://github.com/facebookresearch/anli/tree/main/anlizinganli

We use a fine-grained annotation scheme of the different aspects of inference that are responsible for the gold classification labels, and use it to hand-code all three of the ANLI development sets. These annotations can be used to answer a variety of interesting questions: which inference types are most common, which models have the highest performance on each reasoning type, and which types are the most challenging for state of-the-art models?


## Leaderboard

If you want to have your model added to the leaderboard, please reach out to us or submit a PR.

Model | Publication | A1 | A2 | A3
---|---|---|---|---
InfoBERT (RoBERTa Large) | [Wang et al., 2020](https://openreview.net/forum?id=hpH98mK5Puk) | 75.5 | 51.4 | 49.8
ALUM (RoBERTa Large) | [Liu et al., 2020](https://arxiv.org/abs/2004.08994) | 72.3 | 52.1 | 48.4
GPT-3 | [Brown et al., 2020](https://arxiv.org/abs/2005.14165) | 36.8 | 34.0 | 40.2
ALBERT ( [using the checkpoint in this codebase](#albert) ) | [Lan et al., 2019](https://arxiv.org/abs/1909.11942) | 73.6 | 58.6 | 53.4
XLNet Large | [Yang et al., 2019](https://arxiv.org/abs/1906.08237) | 67.6 | 50.7 | 48.3
RoBERTa Large | [Liu et al., 2019](https://arxiv.org/abs/1907.11692) | 73.8 | 48.9 | 44.4
BERT Large | [Devlin et al., 2018](https://arxiv.org/abs/1810.04805) | 57.4 | 48.3 | 43.5

(Updated on Jan 21 2021: The three entries at the bottom show the test set numbers from Table 3 in the [ANLI paper](https://arxiv.org/abs/1910.14599). We recommend that you report test set results in your paper. Dev scores, obtained for the models in this code base, are reported [below](#checkpoint_results).)

## Implementation

To facilitate research in the field of NLI, we provide an easy-to-use codebase for NLI data preparation and modeling.
The code is built upon [Transformers](https://huggingface.co/transformers/) with a special focus on NLI.

We welcome researchers from various fields (linguistics, machine learning, cognitive science, psychology, etc.) to try NLI.
You can use the code to reproduce the results in our paper or even as a starting point for your research.

Please read more in [**Start your NLI research**](mds/start_your_nli_research.md).  

An important detail in our experiments is that we combine SNLI+MNLI+FEVER-NLI and up-sample different rounds of ANLI to train the models.  
**We highly recommend you refer to the above link for reproducing the results and training your models such that the results will be comparable to the ones on the leaderboard.**

(Updated on May 11, 2022)  
Thanks to [Jared Contrascere](https://github.com/contracode). Now, Researchers can use the [notebook](https://github.com/facebookresearch/anli/blob/main/script/example_scripts/ANLI_on_Google_Colab.ipynb) to run experiments quickly via Google Colab.

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

If you are using our pre-trained model checkpoints with the above code snippet, you would expect to get the following numbers.

<a name="checkpoint_results">Huggingface Model Hub Checkpoint</a> | A1 (dev) | A2 (dev) | A3 (dev) | A1 (test) | A2 (test) | A3 (test)
---|---|---|---|---|---|---
ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli | 73.8 | 50.8 | 46.1 | 73.6 | 49.3 | 45.5
ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli | 73.4 | 52.3 | 50.8 | 70.0 | 51.4 | 49.8
<a name="albert">ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli</a> | 76.0 | 57.0 | 57.0 | 73.6 | 58.6 | 53.4


More in [here](https://github.com/facebookresearch/anli/blob/master/src/hg_api/interactive_eval.py).

## Rules

When using this dataset, we ask that you obey some very simple rules:

1. We want to make it easy for people to provide ablations on test sets without being rate limited, so we release labeled test sets with this distribution. We trust that you will act in good faith, and will not tune on the test set (this should really go without saying)! We may release unlabeled test sets later.

2. **Training data is for training, development data is for development, and test data is for reporting test numbers.** This means that you should not e.g. train on the train+dev data from rounds 1 and 2 and then report an increase in performance on the test set of round 3.

3. We will host a leaderboard on this page. If you want to be added to the leaderboard, please contact us and/or submit a PR with a link to your paper, a link to your code in a public repository (e.g. Github), together with the following information: number of parameters in your model, data used for (pre-)training, and your dev and test results for *each* round, as well as the total over *all* rounds.

## Other NLI Reference

We used following NLI resources in training the backend model of the adversarial collection:
- [**SNLI**](https://nlp.stanford.edu/projects/snli/)
- [**MultiNLI**](https://www.nyu.edu/projects/bowman/multinli/)
- [**NLI style FEVER**](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md)

## Citations

### Dataset
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

### Annotations of the Dataset for Error Analysis
```
@article{williams-etal-2020-anlizing,
  title = "ANLIzing the Adversarial Natural Language Inference Dataset",
  author = "Adina Williams and
    Tristan Thrush and
    Douwe Kiela",
  booktitle = "Proceedings of the 5th Annual Meeting of the Society for Computation in Linguistics",
  year = "2022",
  publisher = "Association for Computational Linguistics",
}
```

## License
ANLI is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.
