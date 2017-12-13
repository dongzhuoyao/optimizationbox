#!/usr/bin/env python

import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
"""
    Python implementation of SVRG-BB and SGD-BB methods from the following paper:
    "Barzilai-Borwein Step Size for Stochastic Gradient Descent".
    Conghui Tan, Shiqian Ma, Yu-Hong Dai, Yuqiu Qian. NIPS 2016.
"""

__license__ = 'MIT'
__author__ = 'Conghui Tan'
__email__ = 'tanconghui@gmail.com'






def svrg_bb(grad, init_step_size, n, d, sess, par, whole_data, max_epoch=100, m=0, tensor_x=None, func=None,
            verbose=True):
    """
        SVRG with Barzilai-Borwein step size for solving finite-sum problems

        grad: gradient function in the form of grad(x, idx), where idx is a list of induces
        init_step_size: initial step size
        n, d: size of the problem
        func: the full function, f(x) returning the function value at x
    """
    if not isinstance(m, int) or m <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')


    def get_loss(data):
        return sess.run([func], feed_dict={"data:0": data})[0]

    def get_grad(data):
        return sess.run([grad], feed_dict={"data:0": data})[0]

    def get_var(var):
        return sess.run([var])[0]

    def ass_var(var, value):
        assign_op = var.assign(value)
        sess.run(assign_op)


    def get_grad_on_var(data,par):
        ass_var(tensor_x, par)
        return get_grad(data)

    step_size = init_step_size
    for k in range(max_epoch):

        #full_grad = grad(x, range(n))
        full_grad = get_grad_on_var(whole_data,tensor_x)
        x_tilde = get_var(tensor_x)
        # estimate step size by BB method
        if k > 0:
            s = x_tilde - last_x_tilde
            y = full_grad - last_full_grad
            step_size = np.linalg.norm(s)**2 / np.dot(np.squeeze(s), np.squeeze(y)) / m

        last_full_grad = full_grad
        last_x_tilde = x_tilde
        if verbose:
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %.2e' % \
                     (k, step_size, np.linalg.norm(full_grad))
            if func is not None:
                output += ', Func. value: %e' % get_loss(whole_data)
            print(output)

        for i in range(m):
            idx = (random.randrange(n), )
            delta = step_size * (get_grad_on_var(whole_data[idx,:],tensor_x) - get_grad_on_var(whole_data[idx,:],x_tilde) + full_grad)
            ass_var(tensor_x, tensor_x - delta)
    return get_var(tensor_x)


def sgd_bb(grad, init_step_size, n, d, tensor_x, func, sess, par, whole_data, max_epoch=100, m=0, beta=0, phi=lambda k: k,
           verbose=True):
    """
        SGD with Barzilai-Borwein step size for solving finite-sum problems

        grad: gradient function in the form of grad(x, idx), where idx is a list of induces
        init_step_size: initial step size
        n, d: size of the problem
        m: step sie updating frequency
        beta: the averaging parameter
        phi: the smoothing function in the form of phi(k)
        func: the full function, f(x) returning the function value at x
    """
    if not isinstance(m, int) or m <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')

    if beta <= 0 or beta >= 1:
        beta = 10/m
        if verbose:
            print('Info: set beta=10/m by default')

    def get_grad(data):
        return sess.run([grad], feed_dict={"data:0": data})[0]

    def get_loss(data):
        return sess.run([func], feed_dict={"data:0": data})[0]

    def get_var(var):
        return sess.run([var])[0]

    def ass_var(var, value):
        assign_op = var.assign(value)
        sess.run(assign_op)

    step_size = init_step_size
    c = 1
    for k in range(max_epoch):
        x_tilde = get_var(tensor_x)
        # estimate step size by BB method
        if k > 1:
            s = x_tilde - last_x_tilde
            y = grad_hat - last_grad_hat
            step_size = np.linalg.norm(s)**2 / abs(np.dot(np.squeeze(s), np.squeeze(y))) / m
            # smoothing the step sizes
            if phi is not None:
                c = c ** ((k-2)/(k-1)) * (step_size*phi(k)) ** (1/(k-1))
                step_size = c / phi(k)

        if verbose:
            full_grad = get_grad(whole_data)
            output = 'Epoch.: {}, Step size: {}, Grad. norm: {}'.format(k, step_size, np.linalg.norm(full_grad))
            if func is not None:
                output += ', Func. value: %e' % get_loss(whole_data)
            print(output)

        if k > 0:
            last_grad_hat = grad_hat
            last_x_tilde = x_tilde

        if k == 0:
            grad_hat = np.zeros((d,1))

        #core logic
        for i in range(m):
            idx = (random.randrange(n), )
            g = get_grad(whole_data[idx,:])
            ass_var(tensor_x, tensor_x - step_size * g)
            # average the gradients
            grad_hat = beta*g + (1-beta)*grad_hat


    return tensor_x
