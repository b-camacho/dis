from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    preds = X.dot(W)
    cor = preds[np.arange(0, X.shape[0]), y]

    epreds = np.sum(np.exp(preds), 1)
    ecor = np.exp(cor)
    odds = ecor / epreds
    logodds = np.log(odds)
    loss = -np.sum(logodds) / X.shape[0]

    dneg = 1/(-X.shape[0])
    dlog = dneg * 1/odds
    doddsecor = dlog * 1/epreds
    doddsepreds = dlog * (-2) * ecor / np.square(epreds)
    decor = doddsecor * np.exp(cor)
    depreds = np.expand_dims(doddsepreds, 1) * np.exp(preds)
    dcorpreds = np.zeros_like(preds)
    dcorpreds[np.arange(0, X.shape[0]), y] = decor
    dpredsW = X.T @ (depreds + dcorpreds)
    dW = dpredsW + (reg * np.sum(W * W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    preds = X @ W
    cor = preds[np.arange(0, X.shape[0]), y]

    epreds = np.sum(np.exp(preds), 1)
    ecor = np.exp(cor)
    odds = ecor / epreds
    logodds = np.log(odds)
    loss = -np.sum(logodds) / X.shape[0]

    dneg = 1/(-X.shape[0])
    dlog = dneg * 1/odds
    doddsecor = dlog * 1/epreds
    doddsepreds = dlog * (-2) * ecor / np.square(epreds)
    decor = doddsecor * np.exp(cor)
    depreds = np.expand_dims(doddsepreds, 1) * np.exp(preds)
    dcorpreds = np.zeros_like(preds)
    dcorpreds[np.arange(0, X.shape[0]), y] = decor
    dpredsW = X.T @ (depreds + dcorpreds)
    dW = dpredsW + (reg * np.sum(W * W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
