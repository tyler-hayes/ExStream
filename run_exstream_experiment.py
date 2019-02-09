import os
import numpy as np
import torch
import utils
from ExStream import ExStream


def experiment(dataset, X_train, y_train, X_test, y_test, capacity):
    """
    Run the main streaming learning experiment.
    :param dataset: string of dataset to use
    :param X_train: training data (N x d)
    :param y_train: training labels (N)
    :param X_test: testing data (M x d)
    :param y_test: testing labels (M)
    :param capacity: how big the buffer for each class should be
    :return: list of incremental accuracies and mean-per-class accuracies
    """

    num_pts = X_train.shape[0]
    acc_list = []
    mpca_list = []
    msg = '\nSample %d/%d -- accuracy=%1.2f%% -- mean-class accuracy=%1.2f%%'

    params = utils.get_mlp_params(dataset)
    model = ExStream(params['layer_sizes'], capacity, buffer_type='class', lr=params['learning_rate'],
                     weight_decay=params['weight_decay'], dropout=params['dropout'], batch_norm=params['batch_norm'],
                     activation=params['activation'], batch_size=params['batch_size'])

    X_train = torch.Tensor(X_train).cuda()
    y_train = torch.LongTensor(y_train).cuda()
    X_test = torch.Tensor(X_test).cuda()

    # perform streaming learning
    for i in range(num_pts):

        # grab next point and label
        pt = X_train[i, :]
        label = y_train[i]

        # fit to point
        model.fit(pt, label)

        # compute and append accuracies
        preds = model.predict(X_test)
        acc, mpca = utils.compute_accuracies(preds, y_test)
        acc_list.append(acc)
        mpca_list.append(mpca)

        # occasionally print how the model is doing
        if i % 50 == 0:
            print(msg % (i, num_pts, acc, mpca))
    return acc_list, mpca_list


def main():
    gpu = '0'  # which GPU to use for experiment

    data_path = './cub200_resnet'  # path to data
    save_path = './results'  # path to save results
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    utils.assure_path_exists(save_path)

    dataset = 'cub200'
    capacity_list = utils.get_dataset_params(dataset)
    experiment_types = ['iid', 'class_iid']

    # loop over all experiment types
    for experiment_type in experiment_types:
        print('Experiment Type: ', experiment_type)

        # load and sort data
        X_train, y_train, X_test, y_test = utils.load_cub200(data_path, experiment_type)

        # loop over desired capacities
        for c in capacity_list:
            print('Capacity: ', c)
            acc, mpca = experiment(dataset, X_train, y_train, X_test, y_test, c)

            # save results out
            acc_name = 'acc_' + experiment_type + '_' + dataset + '_exstream_capacity_' + str(c)
            mpca_name = 'mpca_' + acc_name
            np.save(save_path + '/' + acc_name, np.asarray(acc))
            np.save(save_path + '/' + mpca_name, np.asarray(mpca))


if __name__ == "__main__":
    main()
