import numpy as np
import math
import torch
from torch import nn
from torch import optim


class ExStream(nn.Module):
    """
    The ExStream clustering + MLP model from "Memory Efficient Experience Replay for Streaming Learning."
    This method stores the new sample and merges the two closest samples using the L2 distance metric in a buffer.
    The contents of the buffer is used to train the MLP for a single epoch after each new sample is seen.
    """

    def __init__(self, shape, capacity, buffer_type='class', lr=2e-3, weight_decay=0.0, dropout=0.0, batch_norm=True,
                 activation='relu', batch_size=128, pred_batch_size=512, gpu_id=0, seed=111):
        super(ExStream, self).__init__()

        # parameter settings
        self.input_shape = shape[0]  # size of feature vectors
        self.num_classes = shape[-1]
        self.capacity = capacity  # number of samples stored per class

        # buffer_type = 'class' or buffer_type = 'single'
        self.buffers = buffer_type  # whether to use class-specific buffers (like the paper) or a single buffer
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.gpu_id = gpu_id
        self.idx = 0  # used for storing samples in the buffer

        # initialize buffer arrays
        full_capacity = capacity * self.num_classes
        self.X_b = torch.zeros((full_capacity, self.input_shape)).cuda(self.gpu_id)  # buffer data
        self.y_b = torch.zeros(full_capacity, dtype=torch.long).cuda(self.gpu_id)  # buffer labels
        self.c_b = torch.zeros(full_capacity).cuda(self.gpu_id)  # buffer cluster counts
        self.buffer_counts = torch.zeros(self.num_classes).cuda(self.gpu_id)  # total number of samples in each buffer

        # make the MLP classifier
        self.classifier = self.make_mlp_classifier(seed, shape, activation, batch_norm, dropout)

        # make the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def init_weights(self, m):
        # for initializing network weights
        if type(m) == nn.Linear:
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            # glorot normal weight initialization
            torch.nn.init.normal_(m.weight, mean=0., std=math.sqrt(2 / (fan_in + fan_out)))
            m.bias.data.fill_(1)

    def forward(self, x):
        return self.classifier(x)

    def make_mlp_classifier(self, seed, shape, activation, batch_norm, dropout):
        """
        Set up the MLP classifier.
        :param seed: random seed for intitializing weights
        :param shape: list of layer sizes [input_shape, hidden_layer1_shape, hidden_layer2_shape, ... , output_shape]
        :param activation: choice of relu or elu activation
        :param batch_norm: True for batch norm
        :param dropout: dropout probability
        :return: classifier
        """
        torch.manual_seed(seed)
        classifier = nn.Sequential()
        for i in range(len(shape) - 2):
            layer = nn.Linear(shape[i], shape[i + 1])
            layer.apply(self.init_weights)
            if activation == 'relu':
                act = nn.ReLU()
            elif activation == 'elu':
                act = nn.ELU()
            else:
                raise NotImplementedError('Activation not supported.')
            if batch_norm:
                if i != len(shape) - 1:
                    layer = nn.Sequential(layer, nn.BatchNorm1d(shape[i + 1]), act, nn.Dropout(p=dropout))
                else:
                    layer = nn.Sequential(layer, nn.BatchNorm1d(shape[i + 1]), act)  # no dropout before final layer
            else:
                if i != len(shape) - 1:
                    layer = nn.Sequential(layer, act, nn.Dropout(p=dropout))
                else:
                    layer = nn.Sequential(layer, act)  # no dropout before final layer
            classifier.add_module(str(i), layer)
        layer = nn.Linear(shape[-2], shape[-1])
        layer.apply(self.init_weights)
        classifier.add_module('output', layer)
        return classifier

    def fit(self, X, y):
        """
        Fit the streaming learner to a single training pair.
        :param X: a single data point (PyTorch Tensor)
        :param y: a single label (PyTorch Tensor)
        :return:
        """
        # store or cluster point
        if self.buffers == 'single':
            # use one single buffer that fills to capacity, then starts storing/merging
            self._stream_clustering_single_buffer(X, y)
        else:
            # use a single buffer of fixed capacity per class (like in our paper)
            self._stream_clustering_class_buffers(X, y)

        # consolidate buffer contents for one epoch
        if self.idx > 1:
            X_con = self.X_b[0:self.idx, :]
            y_con = self.y_b[0:self.idx]
            self._consolidate_one_epoch(X_con, y_con)

    def predict(self, X):
        """
        Make predictions on X.
        :param X: an Nxd array of data to make predictions on (PyTorch Tensor)
        :return: Numpy array of size N of predictions
        """
        samples = X.shape[0]
        mb = min(self.pred_batch_size, samples)
        with torch.no_grad():
            model = self.cuda(self.gpu_id)
            model.eval()
            output = torch.zeros((samples, self.num_classes))
            for i in range(0, samples, mb):
                start = min(i, samples - mb)
                end = start + mb
                X_ = X[start:end]
                output[start:end] = model(X_).data.cpu()
            preds = output.data.max(1)[1]
        return preds.numpy()

    def _consolidate_one_epoch(self, X, y):
        """
        Train the MLP for a single epoch on the buffer contents.
        :param X: an Nxd array of data from buffer (PyTorch Tensor)
        :param y: an N array of labels from buffer (PyTorch Tensor)
        :return:
        """

        model = self.cuda(self.gpu_id)
        model.train()
        optimizer = self.optimizer
        criterion = nn.CrossEntropyLoss()
        num_samples = X.shape[0]

        # shuffle data for epoch
        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # consolidate for one epoch
        mb = min(self.batch_size, num_samples)
        for i in range(0, num_samples, mb):
            start = min(i, num_samples - mb)
            end = start + mb
            X_ = X[start:end]
            y_ = y[start:end]

            output = model(X_)
            loss = criterion(output, y_)

            optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
            loss.backward()
            optimizer.step()

    def _stream_clustering_single_buffer(self, x, y):
        """
        If using a single buffer, store points in single buffer until full. Once buffer is full and new point arrives,
        merge two closest samples from class with most points in buffer and store new point.
        :param x: a single data point (PyTorch Tensor)
        :param y: a single label for data point (PyTorch Tensor)
        :return:
        """
        with torch.no_grad():
            self.cuda(self.gpu_id)
            self.eval()
            if self.idx != self.capacity:
                # buffer not full --> store point
                self.X_b[self.idx, :] = x
                self.y_b[self.idx] = y
                self.c_b[self.idx] = 1
                self.buffer_counts[y] += 1
                self.idx += 1
            else:
                # merge closest clusters from class with most points
                class_idx = torch.argmax(self.buffer_counts)
                class_idx2 = class_idx.type(torch.long).cuda(self.gpu_id)
                idxs = self.y_b == class_idx2
                X = self.X_b[idxs]  # num_points x input_shape
                c = self.c_b[idxs]

                # weighted average of closest clusters
                idx0, idx1 = self._l2_dist_metric(X)
                pt1 = X[idx0, :]
                pt2 = X[idx1, :]
                w1 = c[idx0]
                w2 = c[idx1]
                merged_pt = (pt1 * w1 + pt2 * w2) / (w1 + w2)

                # get idx for storing new points in large array
                idxs_arange = torch.arange(self.X_b.shape[0])
                idx0_ = idxs_arange[idxs][idx0]
                idx1_ = idxs_arange[idxs][idx1]

                # store new sample at idx0
                self.X_b[idx0_] = x
                self.y_b[idx0_] = y
                self.c_b[idx0_] = 1
                self.buffer_counts[y] += 1

                # store merged cluster at idx1
                self.X_b[idx1_] = merged_pt
                self.c_b[idx1_] = w1 + w2
                self.buffer_counts[class_idx] -= 1

    def _stream_clustering_class_buffers(self, x, y):
        """
        If using class buffers, store points in class buffer until full. Once class buffer is full and new point
        arrives, merge two closest samples from that class and store new point.
        :param x: a single data point (PyTorch Tensor)
        :param y: a single label for data point (PyTorch Tensor)
        :return:
        """
        with torch.no_grad():
            self.cuda(self.gpu_id)
            self.eval()
            curr_capacity = self.buffer_counts[y]
            if curr_capacity != self.capacity:
                # class buffer not full --> store point
                self.X_b[self.idx, :] = x
                self.y_b[self.idx] = y
                self.c_b[self.idx] = 1
                self.buffer_counts[y] += 1
                self.idx += 1
            else:
                # class buffer is full, merge closest clusters from current class
                idxs = self.y_b == y.type(torch.long)  # get indices for correct class

                # make sure the indices are within self.idx
                idxs = torch.mul(idxs, (torch.arange(idxs.shape[0]) < self.idx).cuda(self.gpu_id))
                X = self.X_b[idxs]  # num_points x input_shape
                c = self.c_b[idxs]

                # weighted average of closest clusters
                idx0, idx1 = self._l2_dist_metric(X)
                pt1 = X[idx0, :]
                pt2 = X[idx1, :]
                w1 = c[idx0]
                w2 = c[idx1]
                merged_pt = (pt1 * w1 + pt2 * w2) / (w1 + w2)

                # get idx for storing new points in large arrays
                idxs_arange = torch.arange(self.X_b.shape[0])
                idx0_ = idxs_arange[idxs][idx0]
                idx1_ = idxs_arange[idxs][idx1]

                # store new sample at idx0
                self.X_b[idx0_] = x
                self.c_b[idx0_] = 1

                # store merged cluster at idx1
                self.X_b[idx1_] = merged_pt
                self.c_b[idx1_] = w1 + w2

    def _l2_dist_metric(self, H):
        """
        Given an array of data, compute the indices of the two closest samples.
        :param H: an Nxd array of data (PyTorch Tensor)
        :return: the two indices of the closest samples
        """
        with torch.no_grad():
            M, d = H.shape
            H2 = torch.reshape(H, (M, 1, d))  # reshaping for broadcasting
            inside = H2 - H
            square_sub = torch.mul(inside, inside)  # square all elements
            psi = torch.sum(square_sub, dim=2)  # capacity x batch_size

            # infinity on diagonal
            mb = psi.shape[0]
            diag_vec = torch.ones(mb).cuda(self.gpu_id) * np.inf
            mask = torch.diag(torch.ones_like(diag_vec).cuda(self.gpu_id))
            psi = mask * torch.diag(diag_vec) + (1. - mask) * psi

            # grab indices
            idx = torch.argmin(psi)
            idx_row = idx / mb
            idx_col = idx % mb
        return torch.min(idx_row, idx_col), torch.max(idx_row, idx_col)
