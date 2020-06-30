# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

import torch


class FlintField(object):
    @classmethod
    def batching(cls, batched_data):
        raise NotImplemented()


class RawFlintField(FlintField):
    @classmethod
    def batching(cls, batched_data):
        return batched_data


class LabelFlintField(FlintField):
    def batching(self, batched_data):
        return torch.tensor(batched_data)


class ArrayIndexFlintField(FlintField):
    def __init__(self, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.left_pad = left_pad
        self.move_eos_to_beginning = move_eos_to_beginning

    def collate_tokens(self, values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        """
        Convert a list of 1d tensors into a padded 2d tensor.
        """
        if not torch.is_tensor(values[0]):
            values = [torch.tensor(v) for v in values]

        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    def batching(self, batched_data):
        return self.collate_tokens(batched_data,
                                   self.pad_idx,
                                   self.eos_idx,
                                   self.left_pad,
                                   self.move_eos_to_beginning)
