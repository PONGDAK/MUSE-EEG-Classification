import numpy as np
import pandas as pd


def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


def standardize(data):
    """ Standardize data """
    np.seterr(divide='ignore', invalid='ignore')
    # Standardize train and train
    result = (data - np.mean(data, axis=0)[None, :, :]) / np.std(data, axis=0)[None, :, :]

    return result


def load_data(csvfile_data, csvfile_label, seq_len, n_channel=4):
    """ load data """
    data = np.loadtxt(csvfile_data, delimiter=',')
    data = MinMaxScaler(data)

    Y = np.zeros((int(len(data) / seq_len), seq_len, n_channel))
    for i in range(0, len(Y)):
        for j in range(0, len(Y[0])):
            for k in range(0, len(Y[0][0])):
                Y[i][j][k] = data[j + i * seq_len][k]
    data = Y

    labels = pd.read_csv(csvfile_label, header=None)

    return data, labels[0].values


def one_hot(labels, n_class=6):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels - 1].T

    return y


def get_batches(X, y, batch_size=100):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]
