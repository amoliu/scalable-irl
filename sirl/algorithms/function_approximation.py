
import numpy as np

__all__ = ['gp_predict', 'gp_covariance']


def gp_kernel(x, y, kernel_type='gaussian', **kwargs):
    """
    Gaussian process kernel score
    """
    if kernel_type == 'gaussian':
        beta = kwargs.get('beta', 0.3)
        return np.exp(-((x[0] - y[0]) ** 2 +
                      (x[1] - y[1]) ** 2) ** (0.5) / (2 * beta ** 2))
    else:
        raise NotImplementedError('Kernel ({}) not implemented'
                                  .format(kernel_type))


def gp_covariance(x, y, kernel_type='gaussian', **kwargs):
    """
    Compute Gram matrix for GP
    """
    return np.array([[gp_kernel(xi, yi, kernel_type) for xi in x] for yi in y])


def gp_predict(target, train_data, gram_matrix, train_labels):
    """
    Predict Value of a node sampled with gaussian process regression
    around neighboring nodes

    Parameters
    ------------
    target : array-like, shape (2)
        [abs, ord] of target point
    train_data : array-like, shape (2 x N)
        training data, [abs, ord] of the points in a certain
        radius of the target point
    train_labels : array-like, shape (N)
        Values of the training data
    gram_matrix : array-like, shape (N x N)
        The Gram matrix

    Returns
    ---------
    y_pred : float
        Predicted value for the target
    sigma_new : float
        Variance of target point prediction
    """
    if not train_data:
        return 10, 1    # TODO - make a better default value treatment
    else:
        k = [gp_kernel(target, yy) for yy in train_data]
        Sinv = np.linalg.pinv(gram_matrix)
        y_pred = np.dot(k, Sinv).dot(train_labels)     # y = K K^-1 y
        sigma_new = gp_kernel(target, target) - np.dot(k, Sinv).dot(k)
        return y_pred, sigma_new
