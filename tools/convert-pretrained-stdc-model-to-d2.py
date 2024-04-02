#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle as pkl
import sys

import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")["state_dict"]

    # Removing from the checkpoint part of the head since Detectron2 is not able to match such keys and trows
    # an error.
    obj_fixed = obj.copy()
    for key in obj.keys():
        if key.startswith('fc.') or key.startswith('bn.') or key.startswith('linear.'):
            obj_fixed.pop(key)

    res = {"model": obj_fixed, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
