import torch
from train import save_graph_list, decode_adj, get_graph
from torch.autograd import Variable
import torch.nn.functional as F
from main import load_test_json_file
from data import GraphSequenceSampler
import json
import networkx as nx
import numpy as np


def save_as_json_file(G_list):
    json_file_path = 'test_graph.json'

    with open(json_file_path, 'w') as jsonfile:
        json.dump(G_list, jsonfile, indent=2)


def train_graph_completion(dataset_test, rnn, output, max_num_node):

    for sample_time in range(1,4):
        G_pred = test_vae_partial_epoch(rnn, output, dataset_test, max_num_node, sample_time=sample_time)
        # save graphs
        fname = 'graph_completion.dat'
        save_as_json_file(G_pred)
    print('graph completion done, graphs saved')


def test_vae_partial_epoch(rnn, output, data_loader,max_num_node, save_histogram=False,sample_time=1):
    # max_num_node isn't added
    max_prev_node = 10
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        grapg_id = data['idx']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, max_prev_node)) # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node,max_prev_node)) # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,max_prev_node))
        print("======max_num_node: ", max_num_node)
        for i in range(max_num_node):
            h = rnn(x_step)
            y_pred_step, _, _ = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:], current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data)
        print("max_num_node", max_num_node)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_json_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_id = {'pred': G_pred, 'idx': grapg_id[i].item()}
            G_pred_list.append(G_pred_id)
        print("Saving Graph batch_idx:", batch_idx)
    return G_pred_list


def get_json_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_array(adj)
    return nx.node_link_data(G)


def sample_sigmoid_supervised(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2)))
    # loop over all batches
    for i in range(y_result.size(0)):
        iter_no = 1
        # using supervision
        if current<y_len[i]:
            while True:
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2)))
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                y_diff = y_result[i].data-y[i]
                if (y_diff>=0).all():
                    break
                iter_no += 1
                if iter_no > 10000:
                    print("Break by iter no")
                    break
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2)))
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data>0).any():
                    break
    return y_result

if __name__ == '__main__':
    graphs = load_test_json_file("../test.json")
    max_prev_node = 10
    batch_size = 32  # normal: 32, and the rest should be changed accordingly
    batch_ratio = 32

    graphs_len = len(graphs)

    max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    dataset_test = GraphSequenceSampler(graphs, max_prev_node=max_prev_node, max_num_node=max_num_node)

    sample_strategy_test = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(dataset_test) for i in range(len(dataset_test))],
        num_samples=batch_size * batch_ratio,
        replacement=False)

    dataset_test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, sampler=sample_strategy_test)

    rnn_loaded_model = torch.load('rnn_model.pth')
    output_loaded_model = torch.load('output_model.pth')
    train_graph_completion(dataset_test_loader, rnn_loaded_model, output_loaded_model, max_num_node)


