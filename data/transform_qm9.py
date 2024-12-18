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



def one_hot(data, out_size=9, num_max_id=5):
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id))
    # data = data[data > 0]
    # 6 is C: Carbon, we adopt 6:C, 7:N, 8:O, 9:F only. the last place (4) is for padding virtual node.
    indices = np.where(data >= 6, data - 6, num_max_id - 1)
    b[np.arange(out_size), indices] = 1
    # print('[DEBUG] data', data, 'b', b)
    return b


def transform_fn(data):
    """

    :param data: ((9,), (4,9,9), (15,))
    :return:
    """
    node, adj, label = data  # node (9,), adj (4,9,9), label (15,)
    # convert to one-hot vector
    node_ = one_hot(node).astype(np.float32)
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
    file_path = '../data/valid_idx_qm9.json'
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [int(idx)-1 for idx in data['valid_idxs']]
    return val_ids
