#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from scipy.sparse import lil_matrix
from stochastic_bb import svrg_bb, sgd_bb

"""
    An example showing how to use svrg_bb and sgd_bb
    The problem here is the regularized logistic regression
"""


if __name__ == '__main__':
    # problem size
    #n, d = 1000, 100
    n, d = 30, 10

    # randomly generate training data
    A = np.random.randn(n, d)
    x_true = np.random.randn(d)
    y = np.sign(np.dot(A, x_true) + 0.1 * np.random.randn(n))

    # generate test data
    A_test = np.random.randn(n, d)
    y_test = np.sign(np.dot(A_test, x_true))

    # preprocess data
    tmp = lil_matrix((n, n))
    tmp.setdiag(y)

    whole_data = tmp * A
    data = tf.placeholder(tf.float32, shape=[None,d],name="data")
    par = tf.Variable(np.zeros((d,1)), dtype=tf.float32)
    tild_par = tf.Variable(np.zeros((d,1)),dtype=tf.float32)
    l2 = 1e-2
    loss = tf.reduce_mean(tf.log(1 + tf.exp(-tf.matmul(data, par)))) + tf.reduce_sum(l2 / 2 * (par ** 2))
    grad = tf.gradients(loss, [par])[0]


    # test SVRG-BB
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x0 = np.random.rand(d)
    print('Begin to run SVRG-BB:')
    x = svrg_bb(grad, 1e-3, n, d, tensor_x=par,func=loss, sess = sess,par = par,whole_data=whole_data, max_epoch=50)
    y_predict = np.sign(np.dot(A_test, x))
    print('Test accuracy: %f' % (np.count_nonzero(y_test == y_predict)*1.0 / n))
    sess.close()


    # test SGD-BB
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print('\nBegin to run SGD-BB:')
    x = sgd_bb(grad, 1e-3, n, d, phi=lambda k: k, tensor_x=par, func=loss, sess = sess, par = par, whole_data=whole_data, beta=0.3, max_epoch=50)
    y_predict = np.sign(np.dot(A_test, x))
    print('Test accuracy: %f' % (np.count_nonzero(y_test == y_predict)*1.0 / n))
    sess.close()
