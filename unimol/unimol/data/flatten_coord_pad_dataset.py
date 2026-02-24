# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unicore.data import BaseWrapperDataset


def flatten_list_of_2Dtensors(list_of_tensors):
    output = []
    for tensor in list_of_tensors:
        if tensor.ndim == 2:
            output.append(tensor)
        else:
            # assuming 3D
            for i in range(tensor.size(-1)):
                output.append(tensor[:, :, i])
    return output

def collate_tokens_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    values = flatten_list_of_2Dtensors(values)

    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    
            
    res = values[0].new(len(values), size, 3).fill_(pad_idx)
    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res


class FlattenRightPadDatasetCoord(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_tokens_coords(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8
        )
