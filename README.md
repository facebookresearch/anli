# Adversarial NLI

## Dataset
Version 0.1 is available here: https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip.

NOTE: This is an early version of the dataset! We may clean it further, will add additional analysis, and possibly add more rounds, at any stage in the future.
Please keep this in mind as you use it in your work.

## Leaderboard

A leaderboard will be hosted on this page.

## Rules

When using this dataset, we ask that you obey some very simple rules:

1. We want to make it easy for people to provide ablations on test sets without being rate limited, so we release labeled test sets with this distribution. We trust that you will act in good faith, and will not tune on the test set (this should really go without saying)! We may release unlabeled test sets later.

2. **Training data is for training, development data is for development, and test data is for reporting test numbers.** This means that you should not e.g. train on the train+dev data from rounds 1 and 2 and then report an increase in performance on the test set of round 3.

3. We will host a leaderboard on this page. If you want to be added to the leaderboard, please contact us and/or submit a PR with a link to your paper, a link to your code in a public repository (e.g. Github), together with the following information: number of parameters in your model, data used for (pre-)training, and your dev and test results for *each* round, as well as the total over *all* rounds.

## Other NLI Resources

We used following NLI resources in training the backend model of the adversarial collection:
- [**SNLI**](https://nlp.stanford.edu/projects/snli/)
- [**MultiNLI**](https://www.nyu.edu/projects/bowman/multinli/)
- [**NLI style FEVER**](https://github.com/easonnie/combine-FEVER-NSMN/blob/master/other_resources/nli_fever.md)


## License
ANLI is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.
