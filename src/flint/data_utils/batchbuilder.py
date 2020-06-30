# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Dict, Type

from flint.data_utils.fields import FlintField, RawFlintField


class BaseBatchBuilder(object):
    def __init__(self, batching_schema: Dict[str, FlintField]) -> None:
        super().__init__()
        self.batching_schema: Dict[str, FlintField] = batching_schema

    def __call__(self, batch):
        field_names = batch[0].keys()
        batched_data = dict()

        for field_name in field_names:
            if field_name not in self.batching_schema:
                # default is RawFlintField
                batched_data[field_name] = RawFlintField.batching([item[field_name] for item in batch])

            else:
                batched_data[field_name] = self.batching_schema[field_name].batching([item[field_name] for item in batch])

        return batched_data


def has_tensor(obj) -> bool:
    """
    Given a possibly complex data structure,
    check if it has any torch.Tensors in it.
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def move_to_device(obj, cuda_device: int):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """

    if cuda_device < 0 or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(cuda_device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, cuda_device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, cuda_device) for item in obj)
    else:
        return obj


if __name__ == '__main__':
    print(RawFlintField.batching)