import numpy as np
import matplotlib.pyplot as plt


def main():
    # plot results for each experiment type and capacity
    save_path = './results'
    dataset = 'cub200'
    for experiment_type in ['iid', 'class_iid']:
        for c in [2, 4, 8, 16]:
            acc_name = 'acc_' + experiment_type + '_' + dataset + '_exstream_capacity_' + str(c)
            mpca_name = 'mpca_' + acc_name
            res = np.load(save_path + '/' + mpca_name + '.npy')

            plt.figure()
            plt.plot(res, label='ExStream')
            plt.xlabel('Sample Number', fontsize=14)
            plt.ylabel('Mean-Class Accuracy [%]', fontsize=14)
            plt.title('Type: %s -- Capacity: %d' % (experiment_type, c), fontsize=14)
            plt.legend()
            plt.show()


if __name__ == '__main__':
    main()
