from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    dots = np.zeros((num_train, num_classes))
    corscores = np.zeros(num_train)
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        dots[i] = scores
        corscores[i] = correct_class_score
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    addreg = 2*reg*np.sum(W)

    ddotclp = np.zeros((num_train, num_classes))
    dyclp = np.zeros(num_train)
    for i in range(num_train):
        for j in range(num_classes):
            if j != y[i]:
                if dots[i][j] - corscores[i] + 1 > 0:
                    ddotclp[i][j] = 1
                    ddotclp[i][y[i]] -= 1
    ddotclp /= num_train

    for i in range(num_train):
        dWi = np.dot(np.matrix(X[i]).T, np.matrix(ddotclp[i]))
        dW += dWi

    dW += addreg
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dot = X @ W # N, C
    cor = dot[np.arange(y.shape[0]), y] # N
    cor = np.expand_dims(cor, 1)
    # print(y[0])
    # print(dot[0])
    # print(dot[:, y][0])
    dotsubcor = dot - cor + 1
    dotclip = np.clip(dotsubcor, 0, None)
    dotsum = np.sum(dotclip)
    loss = dotsum - X.shape[0]
    loss /= X.shape[0]
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ddiv = 1/X.shape[0]

    dclipsubcor = np.ones(dotsubcor.shape)
    dclipsubcor[dotsubcor < 0] = 0
    dclipsubcor *= ddiv

    # dsubcordot = dclipsubcor
    ddotcor = np.ones(cor.shape)
    dsubcorcor = (ddotcor - 1) * dclipsubcor
    dcordot = np.zeros(dot.shape)
    dcordot[np.arange(y.shape[0]), y] = 1
    dsubcordot = (1 - dcordot) * dclipsubcor

    dcordot = dcordot * dsubcorcor

    ddotW = X.T @ dcordot + X.T @ dsubcordot

    dW = ddotW + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
