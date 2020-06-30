# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import config
from datetime import datetime

from utils import common


class ScoreLogger(object):
    def __init__(self, init_tracking_dict) -> None:
        super().__init__()
        self.logging_item_list = []
        self.score_tracker = dict()
        self.score_tracker.update(init_tracking_dict)

    def incorporate_results(self, score_dict, save_key, item=None) -> bool:
        assert len(score_dict.keys()) == len(self.score_tracker.keys())
        for fieldname in score_dict.keys():
            assert fieldname in self.score_tracker

        valid_improvement = False
        for fieldname, value in score_dict.items():
            if score_dict[fieldname] >= self.score_tracker[fieldname]:
                self.score_tracker[fieldname] = score_dict[fieldname]
                valid_improvement = True

        self.logging_item_list.append({'k': save_key, 'v': item})

        return valid_improvement

    def logging_to_file(self, filename):
        if Path(filename).is_file():
            old_logging_list = common.load_json(filename)
            current_saved_key = set()

            for item in self.logging_item_list:
                current_saved_key.add(item['k'])

            for item in old_logging_list:
                if item['k'] not in current_saved_key:
                    raise ValueError("Previous logged item can not be found!")

        common.save_json(self.logging_item_list, filename, indent=2, sort_keys=True)


def gen_file_prefix(model_name, directory_name='saved_models', date=None):
    date_now = datetime.now().strftime("%m-%d-%H:%M:%S") if not date else date
    file_path = os.path.join(config.PRO_ROOT / directory_name / '_'.join((date_now, model_name)))
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path, date_now


def get_cur_time_str():
    date_now = datetime.now().strftime("%m-%d[%H:%M:%S]")
    return date_now


if __name__ == "__main__":
    # print(gen_file_prefix("this_is_my_model."))
    # print(get_cur_time_str())
    score_logger = ScoreLogger({'a_score': -1, 'b_score': -1})
    print(score_logger.incorporate_results({'a_score': 2, 'b_score': -1}, 'key-1', {'a_score': 2, 'b_score': -1}))
    print(score_logger.incorporate_results({'a_score': 2, 'b_score': 3}, 'key-2', {'a_score': 2, 'b_score': 3}))
    print(score_logger.incorporate_results({'a_score': 2, 'b_score': 4}, 'key-2', {'a_score': 2, 'b_score': 4}))
    print(score_logger.incorporate_results({'a_score': 1, 'b_score': 2}, 'key-2', {'a_score': 1, 'b_score': 2}))
    print(score_logger.score_tracker)
    score_logger.logging_to_file('for_testing.json')