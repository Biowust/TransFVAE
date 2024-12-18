import json
import numpy as np
import pandas as pd
import os

import copy
import json
import torch
import sys
import cython
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos

import numpy as np

zinc250_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]  # 0 is for virtual node.
max_atoms = 38
n_bonds = 4


def one_hot_zinc250k(data, out_size=38):
    num_max_id = len(zinc250_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = zinc250_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b


def transform_fn_zinc250k(data):
    node, adj, label = data  # node (9,), adj (4,9,9), label (15,)
    # convert to one-hot vector
    node_ = one_hot_zinc250k(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj_ = copy.deepcopy(adj)
    adj_[1, :, :] = (pd.DataFrame(adj_[1, :, :]) * 2).values
    adj_[2, :, :] = (pd.DataFrame(adj_[2, :, :]) * 3).values
    adj_value = np.sum(adj_[:3], axis=0, keepdims=True)
    adj_value = torch.tensor(adj_value, dtype=torch.long)

    N = node.shape[0]
    temp = np.sum(adj[:3], axis=0, keepdims=True)
    temp = temp.reshape(N, N)
    adj_bool = torch.from_numpy(temp > 0)

    adj_value = adj_value.transpose(0, 2)
    # node adj matrix [N, N] bool
    shortest_path_result, path = algos.floyd_warshall(adj_bool.numpy())
    max_dist = np.amax(shortest_path_result)
    max_dist = 38
    edge_input = algos.gen_edge_input(max_dist, path, adj_value.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    adj_new = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                             axis=0).astype(np.float32)

    edge_input = torch.from_numpy(edge_input).long()
    edge_input = edge_input + 1
    node = torch.from_numpy(node.reshape(N, 1)).long()
    in_degree = adj_bool.long().sum(dim=1).view(-1)
    item = {}
    item["node"] = node  # 9,1
    item["adj_bool"] = adj_bool  # 9 9
    item["attn_bias"] = attn_bias  # 10 10
    item["attn_edge_type"] = adj_value  # 9 9 1
    item["rel_pos"] = rel_pos  # 9 9
    item["in_degree"] = in_degree  # 9
    item["out_degree"] = in_degree  # 9
    item["edge_input"] = edge_input  # 9 9 510 1

    return node_, adj_new, item


def get_val_ids():
    file_path = '../data/valid_idx_zinc.json'
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [idx-1 for idx in data]
    return val_ids
