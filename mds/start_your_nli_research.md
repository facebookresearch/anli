# Start your NLI Research
This tutorial gives detailed instructions to help you start research on NLI with pre-trained state-of-the-art models (June 2020).

## Requirements
- python 3.6+
- tqdm
- torch 1.4.0                   https://pytorch.org/
- transformers 3.0.2            https://github.com/huggingface/transformers/

## Initial Setup
### 1. Setup your python environment and install the requirements.
### 2. Clone this repo.
```
git clone https://github.com/facebookresearch/anli.git
```
### 3. Run the following command in your terminal. 
```
source setup.sh
```
The command will assign the environment variables `$DIR_TMP` and `$PYTHONPATH`
to the project root path and src path, respectively. 
These two variables are needed to run any following scripts in this tutorial.

## Data Preparation
### 1. Run the following command to download the data.
```
cd $DIR_TMP                     # Make sure that you run all the scripts in the root of this repo.
bash script/download_data.sh
```
All the data (SNLI, MNLI, FEVER-NLI, ANLI) will be downloaded in the `data` directory.
If any data is missing, you can remove the unchecked folders in the `data` directory and re-download.

### 2. Run the following python script to build the data.
```bash
cd $DIR_TMP                                 # Make sure that you run all the scripts in the root of this repo.
python src/dataset_tools/build_data.py      # If you encounter import errors, please make sure you have run `source setup.sh` to set up the `$PYTHONPATH`
```
The script will convert SNLI, MNLI, FEVER-NLI, ANLI all into the same unified NLI format, and will also remove examples in SNLI and MNLI that do not have a gold label (as in prior work).

#### Data Directory
Once the `build_data.py` script has completed successfully, the `data` directory (in the project root) should contains a directory called `build` containing the dataset files in the unified data format. Your data directory should have a structure like the one attached below.
```
data
├── build
│   ├── anli
│   │   ├── r1
│   │   │   ├── dev.jsonl
│   │   │   ├── test.jsonl
│   │   │   └── train.jsonl
│   │   ├── r2
│   │   │   ├── dev.jsonl
│   │   │   ├── test.jsonl
│   │   │   └── train.jsonl
│   │   └── r3
│   │       ├── dev.jsonl
│   │       ├── test.jsonl
│   │       └── train.jsonl
│   ├── fever_nli
│   │   ├── dev.jsonl
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   ├── mnli
│   │   ├── m_dev.jsonl
│   │   ├── mm_dev.jsonl
│   │   └── train.jsonl
│   └── snli
│       ├── dev.jsonl
│       ├── test.jsonl
│       └── train.jsonl
├── anli_v1.0
│   ├── ...     # The unzipped ANLI data with original format.
├── multinli_1.0
│   ├── ...     # The unzipped MNLI data with original format.
├── nli_fever
│   ├── ...     # The unzipped NLI-FEVER data with original format.
└── snli_1.0
    ├── ...     # The unzipped SNLI data with original format.
```

#### Data Format
NLI is basically a 3-way sequence-to-label classification task where the inputs are two textual sequences 
called `premise` and `hypothesis`, and the output is a discrete label that is either `entailment`, `contradiction`, or `neutral`. 
(Some works consider it to be a 2-way classification tasks. The code here can easily be converted to any sequence-to-label task with some hacking.)

The training script will load NLI data with a unified `jsonl` format in which each line is a JSON object for one NLI example.
The JSON object should have the following fields:
- "uid": unique id of the example;
- "premise": the premise of the NLI example;
- "hypothesis": the hypothesis of the NLI example;
- "label": the label of the example. The label is from the set {"e", "n", "c"}, denoting the 3 classes "entailment", "neutral", or "contradiction", respectively.
- Additional dataset specific fields...

Here is one example from SNLI:
```json
{ 
  "uid": "4705552913.jpg#2r1n", 
  "premise": "Two women are embracing while holding to go packages.", 
  "hypothesis": "The sisters are hugging goodbye while holding to go packages after just eating lunch.", 
  "label": "n"
}
```

Note that some training examples and all the development and test examples in ANLI have a `reason` field showing the reason for the annotation. Please read the paper for details.

If you want to train or evaluate the model on data other than SNLI, MNLI, FEVER-NLI, or ANLI, we recommend that you refer to `src/dataset_tools/build_data.py` and `src/dataset_tools/format_convert.py` and use the tools in the repo to build your own data to avoid any exceptions. 

## Model Training
Now, you can use the following script to start training your NLI models.

```bash
export MASTER_ADDR=localhost

python src/nli/training.py \
    --model_class_name "roberta-large" \
    -n 1 \
    -g 2 \
    -nr 0 \
    --max_length 156 \
    --gradient_accumulation_steps 1 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --save_prediction \
    --train_data snli_train:none,mnli_train:none \
    --train_weights 1,1 \
    --eval_data snli_dev:none \
    --eval_frequency 2000 \
    --experiment_name "roberta-large|snli|nli"
```

### Argument Explanation
- "--model_class_name": The argument specify the model class we will be using in the training. We currently support "roberta-large", "roberta-base", "xlnet-large", "xlnet-base", "bert-large", "bert-base", "albert-xxlarge", "bart-large".
Note that the model_class_name should be the same when you load checkpoint for evaluation.
- "-n": The number of nodes (machines) for training. In most cases, we recommend to set it to 1. MultiNode training is not tested.
- "-g": The number of GPUs for training.
- "-nr": Node rank. In most cases, we recommend to set it to 0. MultiNode training is not tested.
- "--train_data": Specify source of training data separated by commas. The string before the colon is the name of the source and the string after the colon is the location of the data. Note that for SNLI, MNLI, FEVER-NLI, and ANLI, you should just give "none" as the location because their location have been manually registered in the script. For customized input data, you will need to specify the actual path of the data. 
- "--train_weights": Specify the size of the training data from different source. At every training epoch, the training data is re-sampled from different source and then combined altogether. The weights indicate multiplication of the sampled training size for the correspondent source. "1" means we just add all the data from that source to the training data. "0.5" means we sample half of the data from that source. "3.8" means we sample (with replacement) the training data such that the resulting size is 3.8 * the size of that source.  
Notes: The two argument above gives important information about the data used at every training epoch. The number of values for `--train_weights` needs to match the number of items in `--train_data`.  
For example, suppose snli_train has 100 examples, and `[name_of_your_data]` has 200 examples. Then, `--train_data snli_train:none,[name_of_your_data]:[path_to_your_data]/[filename_of_your_data].jsonl --train_weights 0.5,2` means training the model with 100 * 0.5 = 50 snli training examples and 200 * 2 = 400 `[name_of_your_data]` examples sampled with replacement at every epoch.  
- "--eval_data" Specify source of evaluation data separated by ",". (Same as "--train_data")
- "--eval_frequency": The number of iteration steps between two saved model checkpoints.
- "--experiment_name": The name of the experiment. During training, the checkpoints will be saved in `saved_models/{TRAINING_START_TIME}_[experiment_name]` directory (in the project root). So, the name will be an important identifier for finding the saved checkpoints.

The other arguments should be self-explanatory. We recommend that you read the code if you are unsure about a specific argument.

The example scripts `script/example_scripts/train_roberta.sh` and `script/example_scripts/train_xlnet.sh` can be used to reproduce the leaderboard RoBERTa and XLNet results, respectively.  
An **important detail** is that the training data used in above experiments are SNLI + MNLI + FEVER-NLI + 10*A1 (10 times upsampled ANLI-R1) + 20*A2 (20 times upsampled ANLI-R2) + 10*A3 (10 times upsampled ANLI-R3), as specified in the following arguments:
```
    --train_data snli_train:none,mnli_train:none,fever_train:none,anli_r1_train:none,anli_r2_train:none,anli_r3_train:none \
    --train_weights 1,1,1,10,20,10 \
```
The scripts were tested on a machine with 8 Tesla V100 (16GB).

You can try a smaller RoBERTa-base model with `script/example_scripts/train_roberta_small.sh` which can be run on single GPU with 12GB memory.  

During training, model checkpoints will be automatically saved in `saved_models` directory.

#### Batch Size
Training batch size might be a factor for performance. The actual training batch size can be calculated as `[number_of_nodes](-n)` * `[number_of_gpus_per_node](-g)` * `[per_gpu_train_batch_size]` * `[gradient_accumulation_steps]`.  
If the GPU memory is limited, you can set small forward batch size but with more gradient accumulation steps. E.g. `-n 1 -g 2 -nr 0 --gradient_accumulation_steps 8 --per_gpu_train_batch_size 8` can still give you a 2 * 8 * 8 = 128 training batch size.

#### Distributed Data Parallel
The code uses pytorch distributed data parallel for multiGPU training which technologically can support any number of GPU usage.  
You need to set `$MASTER_ADDR` variable to pass IP address of the master process to the python script.  
In most cases, you will just use one machine and you can just set this variable to "localhost".  

## Evaluating Trained Models
```bash
python src/nli/evaluation.py \ 
    --model_class_name "roberta-large" \
    --max_length 156 \
    --per_gpu_eval_batch_size 16 \
    --model_checkpoint_path \
    "[the directory that contains your checkpoint]/model.pt" \
    --eval_data anli_r1_test:none,anli_r2_test:none,anli_r3_test:none \
    --output_prediction_path [the path of the directory you want the output to be saved]
```
Notice:
1. The "model_class_name" need to be the same as the one used in `training.py`.
2. You need to specify the path to your model parameter (the file named `model.pt`). 
3. Evaluation is done in single GPU.
