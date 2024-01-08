from model import GRUModel, MLP_VAE_conditional_plain
import networkx as nx
from data import GraphSequenceSampler
from train import train
import torch
import random
import json
import numpy as np


def load_json_file(file_name):
    with open(file_name) as f:
        data = json.load(f)

    x_d = np.array(data["TrainX"])
    graphs = []
    print("Graph loading is started")
    for idx, row in enumerate(x_d):
        composed_graph = create_graph(row)
        graphs.append(composed_graph)
    print("Graph loading is ended")
    return graphs


def load_test_json_file(file_name):
    with open(file_name) as f:
        data = json.load(f)

    y_d = data["TrainY"]
    graphs = []
    print("Graph loading is started")
    for idx, row in enumerate(y_d):
        composed_graph = create_graph(row)
        graphs.append(composed_graph)
    print("Graph loading is ended")
    return graphs


def create_graph(sequence_arr):
    G = nx.Graph()

    for i in range(len(sequence_arr) - 1):
        G.add_edge(sequence_arr[i], sequence_arr[i + 1])

    return G


if __name__ == '__main__':
    #=================== Model Paramters ==========================================
    batch_size = 32  # normal: 32, and the rest should be changed accordingly
    test_batch_size = 32
    batch_ratio = 32
    parameter_shrink = 1
    hidden_size_rnn = int(128 / parameter_shrink)  # hidden size for main RNN
    hidden_size_rnn_output = 16
    embedding_size_rnn_output = 8
    embedding_size_rnn = int(64 / parameter_shrink)
    num_layers = 4
    max_prev_node = 10
    embedding_size_output = int(64 / parameter_shrink)
    #===============================================================================

    graphs = load_json_file("../train.json")

    # split datasets
    random.seed(123)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8 * graphs_len)]
    graphs_validate = graphs[0:int(0.2 * graphs_len)]

    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)

    max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_test_node = max([graphs_validate[i].number_of_nodes() for i in range(len(graphs_validate))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args.max_num_node = 2000
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    print('max number node: {}'.format(max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    print('max previous node: {}'.format(max_prev_node))

    ### dataset initialization

    dataset = GraphSequenceSampler(graphs_train, max_prev_node=max_prev_node, max_num_node=max_num_node)
    dataset_test = GraphSequenceSampler(graphs_validate, max_prev_node=max_prev_node, max_num_node=max_num_test_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=batch_size * batch_ratio,
                                                                     replacement=True)

    sample_strategy_test = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset_test) for i in range(len(dataset_test))],
                                                                     num_samples=batch_size * batch_ratio,
                                                                     replacement=False)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sample_strategy)

    dataset_test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, sampler=sample_strategy_test)

    rnn = GRUModel(input_size=max_prev_node, embedding_size=embedding_size_rnn,
                        hidden_size=hidden_size_rnn, num_layers=num_layers, has_input=True,
                        has_output=False, output_size=hidden_size_rnn_output)
    output = MLP_VAE_conditional_plain(h_size=hidden_size_rnn, embedding_size=embedding_size_output, y_size=max_prev_node)
    del graphs
    del graphs_train
    del graphs_validate
    ### start training
    train(dataset_loader, rnn, output, max_num_node, dataset_test_loader)

    ### graph completion
    # train_graph_completion(args,dataset_loader,rnn,output)

    ### nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)




