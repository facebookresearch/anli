# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import config
from dataset_tools.format_convert import sm_nli2std_format, fever_nli2std_format, a_nli2std_format
from utils import common

# ANLI_VERSION = 1.0


def build_snli(path: Path):
    snli_data_root_path = (path / "snli")
    if not snli_data_root_path.exists():
        snli_data_root_path.mkdir()
    o_train = common.load_jsonl(config.PRO_ROOT / "data/snli_1.0/snli_1.0_train.jsonl")
    o_dev = common.load_jsonl(config.PRO_ROOT / "data/snli_1.0/snli_1.0_dev.jsonl")
    o_test = common.load_jsonl(config.PRO_ROOT / "data/snli_1.0/snli_1.0_test.jsonl")

    d_trian = sm_nli2std_format(o_train)
    d_dev = sm_nli2std_format(o_dev)
    d_test = sm_nli2std_format(o_test)

    print("SNLI examples without gold label have been filtered.")
    print("SNLI Train size:", len(d_trian))
    print("SNLI Dev size:", len(d_dev))
    print("SNLI Test size:", len(d_test))

    common.save_jsonl(d_trian, snli_data_root_path / 'train.jsonl')
    common.save_jsonl(d_dev, snli_data_root_path / 'dev.jsonl')
    common.save_jsonl(d_test, snli_data_root_path / 'test.jsonl')


def build_mnli(path: Path):
    data_root_path = (path / "mnli")
    if not data_root_path.exists():
        data_root_path.mkdir()
    o_train = common.load_jsonl(config.PRO_ROOT / "data/multinli_1.0/multinli_1.0_train.jsonl")
    o_mm_dev = common.load_jsonl(config.PRO_ROOT / "data/multinli_1.0/multinli_1.0_dev_mismatched.jsonl")
    o_m_dev = common.load_jsonl(config.PRO_ROOT / "data/multinli_1.0/multinli_1.0_dev_matched.jsonl")

    d_trian = sm_nli2std_format(o_train)
    d_mm_dev = sm_nli2std_format(o_mm_dev)
    d_m_test = sm_nli2std_format(o_m_dev)

    print("MNLI examples without gold label have been filtered.")
    print("MNLI Train size:", len(d_trian))
    print("MNLI MisMatched Dev size:", len(d_mm_dev))
    print("MNLI Matched dev size:", len(d_m_test))

    common.save_jsonl(d_trian, data_root_path / 'train.jsonl')
    common.save_jsonl(d_mm_dev, data_root_path / 'mm_dev.jsonl')
    common.save_jsonl(d_m_test, data_root_path / 'm_dev.jsonl')


def build_fever_nli(path: Path):
    data_root_path = (path / "fever_nli")
    if not data_root_path.exists():
        data_root_path.mkdir()

    o_train = common.load_jsonl(config.PRO_ROOT / "data/nli_fever/train_fitems.jsonl")
    o_dev = common.load_jsonl(config.PRO_ROOT / "data/nli_fever/dev_fitems.jsonl")
    o_test = common.load_jsonl(config.PRO_ROOT / "data/nli_fever/test_fitems.jsonl")

    d_trian = fever_nli2std_format(o_train)
    d_dev = fever_nli2std_format(o_dev)
    d_test = fever_nli2std_format(o_test)

    print("FEVER-NLI Train size:", len(d_trian))
    print("FEVER-NLI Dev size:", len(d_dev))
    print("FEVER-NLI Test size:", len(d_test))

    common.save_jsonl(d_trian, data_root_path / 'train.jsonl')
    common.save_jsonl(d_dev, data_root_path / 'dev.jsonl')
    common.save_jsonl(d_test, data_root_path / 'test.jsonl')


def build_anli(path: Path, round=1, version='1.0'):
    data_root_path = (path / "anli")
    if not data_root_path.exists():
        data_root_path.mkdir()

    round_tag = str(round)

    o_train = common.load_jsonl(config.PRO_ROOT / f"data/anli_v{version}/R{round_tag}/train.jsonl")
    o_dev = common.load_jsonl(config.PRO_ROOT / f"data/anli_v{version}/R{round_tag}/dev.jsonl")
    o_test = common.load_jsonl(config.PRO_ROOT / f"data/anli_v{version}/R{round_tag}/test.jsonl")

    d_trian = a_nli2std_format(o_train)
    d_dev = a_nli2std_format(o_dev)
    d_test = a_nli2std_format(o_test)

    print(f"ANLI (R{round_tag}) Train size:", len(d_trian))
    print(f"ANLI (R{round_tag}) Dev size:", len(d_dev))
    print(f"ANLI (R{round_tag}) Test size:", len(d_test))

    if not (data_root_path / f"r{round_tag}").exists():
        (data_root_path / f"r{round_tag}").mkdir()

    common.save_jsonl(d_trian, data_root_path / f"r{round_tag}" / 'train.jsonl')
    common.save_jsonl(d_dev, data_root_path / f"r{round_tag}" / 'dev.jsonl')
    common.save_jsonl(d_test, data_root_path / f"r{round_tag}" / 'test.jsonl')


def build_data():
    processed_data_root = config.PRO_ROOT / "data" / "build"
    if not processed_data_root.exists():
        processed_data_root.mkdir()
    build_snli(processed_data_root)
    build_mnli(processed_data_root)
    build_fever_nli(processed_data_root)
    for round in [1, 2, 3]:
        build_anli(processed_data_root, round)

    print("NLI data built!")


if __name__ == '__main__':
    build_data()