#!/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

#ANLI_VERSION=0.1
ANLI_VERSION=1.0
#echo ${ANLI_VERSION}

if [[ -z "$DIR_TMP" ]]; then    # If project root not defined.
    # get the directory of this file
    export CURRENT_FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    # setup root directory.
    export DIR_TMP=$(cd "${CURRENT_FILE_DIR}/.."; pwd)
fi

export DIR_TMP=$(cd "${DIR_TMP}"; pwd)
echo "The path of project root: ${DIR_TMP}"


# check if data exist.
if [[ ! -d ${DIR_TMP}/data ]]; then
    mkdir ${DIR_TMP}/data
fi

# download the snli data.
cd ${DIR_TMP}/data
if [[ ! -d  snli_1.0 ]]; then
    wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    unzip "snli_1.0.zip"
    rm -rf "snli_1.0.zip" && rm -rf "__MACOSX"
    echo "SNLI Ready"
fi

# download the mnli data.
cd ${DIR_TMP}/data
if [[ ! -d  multinli_1.0 ]]; then
    wget "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    unzip "multinli_1.0.zip"
    rm -rf "multinli_1.0.zip" && rm -rf "__MACOSX"
    echo "MNLI Ready"
fi

# download the fever nli data.
cd ${DIR_TMP}/data
if [[ ! -d  nli_fever ]]; then
    wget "https://www.dropbox.com/s/hylbuaovqwo2zav/nli_fever.zip"
    unzip "nli_fever.zip"
    rm -rf "nli_fever.zip" && rm -rf "__MACOSX"
    echo "FEVER NLI Ready"
fi

# download the anli_v0.1
cd ${DIR_TMP}/data
if [[ ! -d  anli_v${ANLI_VERSION} ]]; then
    wget "https://dl.fbaipublicfiles.com/anli/anli_v${ANLI_VERSION}.zip"
    unzip "anli_v${ANLI_VERSION}.zip"
    rm -rf "anli_v${ANLI_VERSION}.zip" && rm -rf "__MACOSX"
    echo "ANLI Ready"
fi

ALL_DATA_CHECKED=true

# Check data SNLI:
cd ${DIR_TMP}/data
if [[ -f snli_1.0/snli_1.0_train.jsonl ]] && [[ -f snli_1.0/snli_1.0_dev.jsonl ]] && [[ -f snli_1.0/snli_1.0_test.jsonl ]]; then
    echo "SNLI checked."
else
    echo "Some SNLI files are not ready. Please remove the \"snli_1.0\" directory and run download.sh again."
    ALL_DATA_CHECKED=false
fi

# Check data MNLI:
cd ${DIR_TMP}/data
if [[ -f multinli_1.0/multinli_1.0_train.jsonl ]] && [[ -f multinli_1.0/multinli_1.0_dev_mismatched.jsonl ]] && [[ -f multinli_1.0/multinli_1.0_dev_matched.jsonl ]]; then
    echo "MNLI checked."
else
    echo "Some MNLI files are not ready. Please remove the \"multinli_1.0\" directory and run download.sh again."
    ALL_DATA_CHECKED=false
fi

# Check data FEVER NLI:
cd ${DIR_TMP}/data
if [[ -f nli_fever/train_fitems.jsonl ]] && \
[[ -f nli_fever/test_fitems.jsonl ]] && \
[[ -f nli_fever/dev_fitems.jsonl ]]; then
    echo "FEVER NLI checked."
else
    echo "Some FEVER NLI files are not ready. Please remove the \"nli_fever\" directory and run download.sh again."
    ALL_DATA_CHECKED=false
fi

# Check data ANLI:
cd ${DIR_TMP}/data
if [[ -f anli_v${ANLI_VERSION}/R1/train.jsonl ]] && \
[[ -f anli_v${ANLI_VERSION}/R1/dev.jsonl ]] && \
[[ -f anli_v${ANLI_VERSION}/R1/test.jsonl ]] && \
[[ -f anli_v${ANLI_VERSION}/R2/train.jsonl ]] && \
[[ -f anli_v${ANLI_VERSION}/R2/dev.jsonl ]] && \
[[ -f anli_v${ANLI_VERSION}/R2/test.jsonl ]] && \
[[ -f anli_v${ANLI_VERSION}/R3/train.jsonl ]] && \
[[ -f anli_v${ANLI_VERSION}/R3/dev.jsonl ]] && \
[[ -f anli_v${ANLI_VERSION}/R3/test.jsonl ]]; \
then
    echo "ANLI checked."
else
    echo "Some ANLI files are not ready. Please remove the \"anli_v${ANLI_VERSION}\" directory and run download.sh again."
    ALL_DATA_CHECKED=false
fi

if [[ ${ALL_DATA_CHECKED} == true ]];
then
    echo "Data download completed and checked."
else
    echo "Some data is missing. Please examine again or delete the data directory and re-run download.sh."
fi