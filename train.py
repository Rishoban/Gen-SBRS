import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time as tm
import networkx as nx
import pickle


def binary_cross_entropy_weight(y_pred, y,has_weight=False, weight_length=1, weight_max=10):
    '''

    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    '''
    if has_weight:
        weight = torch.ones(y.size(0),y.size(1),y.size(2))
        weight_linear = torch.arange(1,weight_length+1)/weight_length*weight_max
        weight_linear = weight_linear.view(1,weight_length,1).repeat(y.size(0),1,y.size(2))
        weight[:,-1*weight_length:,:] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight.cuda())
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def train_rnn_epoch(epoch,rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    num_layers = 4
    graph_type = 'yoochoose'
    parameter_shrink = 2
    batch_ratio = 32
    epochs_log = 25
    note = 'GraphRNN_RNN'
    epochs = 50
    hidden_size_rnn = int(128 / parameter_shrink)
    fname = note + '_' + graph_type + '_' + str(num_layers) + '_' + str(hidden_size_rnn) + '_'
    hidden_size_rnn = int(128 / parameter_shrink)


    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()

        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()

        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x)
        y = Variable(y)
        output_x = Variable(output_x)
        output_y = Variable(output_y)

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(num_layers-1, h.size(0), h.size(1)))
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, epochs,loss.data, graph_type, num_layers, hidden_size_rnn))

        # logging
        print('loss_'+fname, loss.data, epoch*batch_ratio+batch_idx)
        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.data*feature_dim
    return loss_sum/(batch_idx+1)


def predict_rnn(rnn, output, data_loader):
    num_layers = 4
    graph_type = 'yoochoose'
    parameter_shrink = 2
    batch_ratio = 32
    epochs_log = 25
    note = 'GraphRNN_RNN'
    epochs = 50
    hidden_size_rnn = int(128 / parameter_shrink)
    fname = note + '_' + graph_type + '_' + str(num_layers) + '_' + str(hidden_size_rnn) + '_'
    hidden_size_rnn = int(128 / parameter_shrink)

    rnn.eval()
    output.eval()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()

        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()

        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        # rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x)
        y = Variable(y)
        output_x = Variable(output_x)
        output_y = Variable(output_y)

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(num_layers-1, h.size(0), h.size(1)))
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]

        loss = binary_cross_entropy_weight(y_pred, output_y)
        print("Test error: ", loss)


def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y = F.sigmoid(y)
    # do sampling
    if sample:
        if sample_time>1:
            y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2)))
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = Variable(torch.rand(y.size(1), y.size(2)))
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data>0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2)))
            y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*thresh)
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def test_rnn_epoch(rnn, output,max_num_node,  test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()
    max_prev_node = 10
    num_layers = 4
    # generate graphs
    max_num_node = int(max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, max_prev_node)) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,max_prev_node))
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(num_layers - 1, h.size(0), h.size(2)))
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,max_prev_node))
        output_x_step = Variable(torch.ones(test_batch_size,1,1))
        for j in range(min(max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data)
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list


def get_graph(adj):
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
    return G


def train(dataset_train, rnn, output, max_num_node, dataset_test_loader):
    # check if load existing model
    #=============================== Model Paramters ====================================
    epoch = 1
    lr = 0.003
    milestones = [400, 1000]
    lr_rate = 0.3
    epochs = 50
    epochs_test_start = 100
    test_total_size = 1000
    test_batch_size = 32
    epochs_test = 100
    graph_type = 'ladder'
    note = 'GraphRNN_RNN'
    epochs_save = 100
    parameter_shrink = 1
    hidden_size_rnn = int(128 / parameter_shrink)
    num_layers = 4
    fname_pred = note + '_' + graph_type + '_' + str(num_layers) + '_' + str(hidden_size_rnn) + '_pred_'
    fname = note + '_' + graph_type + '_' + str(num_layers) + '_' + str(hidden_size_rnn) + '_'
    #=======================================================================================================
    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=milestones, gamma=lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=milestones, gamma=lr_rate)

    # start main loop
    time_all = np.zeros(epochs)
    while epoch <= epochs:
        time_start = tm.time()
        # train

        train_vae_epoch(epoch, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)

        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % epochs_test == 0 and epoch >= epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                while len(G_pred) < test_total_size:

                    G_pred_step = test_vae_epoch(epoch, rnn, output,max_num_node, test_batch_size=test_batch_size,
                                                     sample_time=sample_time)
                    G_pred.extend(G_pred_step)
                # save graphs
                save_graph_list(G_pred, fname)
            print('test done, graphs saved')
        epoch += 1
    torch.save(rnn, 'rnn_model.pth')
    torch.save(output, 'output_model.pth')


def train_vae_epoch(epoch, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    #=============================== Model Parameters =====================================
    num_layers = 4
    graph_type = 'yoochoose'
    parameter_shrink = 2
    batch_ratio = 32
    epochs_log = 25
    note = 'GraphRNN_RNN'
    epochs = 50
    hidden_size_rnn = int(128 / parameter_shrink)
    fname = note + '_' + graph_type + '_' + str(num_layers) + '_' + str(hidden_size_rnn) + '_'
    hidden_size_rnn = int(128 / parameter_shrink)
    #========================================================================================

    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x)
        y = Variable(y)

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        y_pred,z_mu,z_lsgms = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
        z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
        # use cross entropy loss
        loss_bce = binary_cross_entropy_weight(y_pred, y)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= y.size(0)*y.size(1)*sum(y_len) # normalize
        loss = loss_bce + loss_kl
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        z_mu_mean = torch.mean(z_mu.data)
        z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
        z_mu_min = torch.min(z_mu.data)
        z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
        z_mu_max = torch.max(z_mu.data)
        z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


        if epoch % epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, epochs,loss_bce.item(), loss_kl.item(), graph_type, num_layers, hidden_size_rnn))
            print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

        loss_sum += loss.item()
    return loss_sum/(batch_idx+1)


def test_vae_epoch(epoch,  rnn, output,max_num_node, test_batch_size=16, save_histogram=False, sample_time = 1):
    max_prev_node = 10
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, max_prev_node)) # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, max_prev_node)) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,max_prev_node))
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step, _, _ = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)


    return G_pred_list


def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)




