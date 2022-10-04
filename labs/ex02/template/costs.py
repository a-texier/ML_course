# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np




def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
<<<<<<< HEAD
    e = y - np.dot(tx,w)
    N = y.shape[0]
    return 1/(2*N) * np.dot(e.T,e)
=======
>>>>>>> 6786ed35ca89271dac99a74c25b70d0fc0c0de1f
    # TODO: compute loss by MSE
    # ***************************************************
    raise NotImplementedError
    