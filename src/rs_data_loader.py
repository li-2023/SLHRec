import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as Data
import random

def load_rs_data(args):
    
    dataset = args.dataset
    
    file_path = '../data/' + args.dataset + '/' 
    
    # load interaction
    train_data = torch.from_numpy(np.loadtxt(file_path + "CF_data/final_train.txt", dtype=np.int32))
    test_data  = torch.from_numpy(np.loadtxt(file_path + "CF_data/final_test.txt", dtype=np.int32))
    eval_data  = torch.from_numpy(np.loadtxt(file_path + "CF_data/final_valid.txt", dtype=np.int32))
    
    interaction_data = (train_data, eval_data, test_data)
    print('Interaction Data Loaded!')

    item_set = set(list(train_data[:, 1].numpy()) + list(test_data[:, 1].numpy()) + list(eval_data[:, 1].numpy()))
    # print("item_set:", item_set)

    
    graph_np = np.loadtxt(file_path + "CF_data/final_graph.txt", dtype=np.int32)
    edge_index_ex = torch.from_numpy(np.array([graph_np[:, 0], graph_np[:, 2]])).long()
    edge_type_ex = torch.from_numpy(graph_np[:, 1]).long()
    n_relation_ex = len(set(graph_np[:, 1]))

    graph_np = np.loadtxt(file_path + "CF_data/latent_graph.txt", dtype=np.int32)
    edge_index_im = torch.from_numpy(np.array([graph_np[:, 0], graph_np[:, 2]])).long()
    edge_type_im = torch.from_numpy(graph_np[:, 1]).long()
    n_relation_im = len(set(graph_np[:, 1]))

    
    print('Fact Data Loaded!')

    return interaction_data, item_set, edge_index_ex, edge_type_ex, n_relation_ex, edge_index_im, edge_type_im, n_relation_im

