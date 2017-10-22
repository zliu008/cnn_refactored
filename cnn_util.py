# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 16:37:20 2017

@author: zaoliu
"""

import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.bn import batch_normalization


class LeNetConvLayer(object):
    def __init__(self, rng, input, filter_shape, 
                 image_shape, use_bn = 1):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        W_bound = numpy.sqrt(2. /(fan_in + fan_out))
        W_value = rng.normal(loc = 0., scale = W_bound, size = filter_shape)
        self.W = theano.shared(W_value, name = 'W', borrow = True)
        conv_out = conv2d(input = self.input, 
                   filters = self.W)        
        
#        pooled_out = pool.pool_2d(input = conv_out, 
#                                  ds=poolsize, ignore_border=True)                            
        
        b_bound = numpy.sqrt(2. /fan_out)
        b_value = rng.normal(loc = 0, scale = b_bound, size=(filter_shape[0],))
        self.b = theano.shared(b_value, name = 'b', borrow  = True)
        
        self.linear = conv_out + self.b.dimshuffle('x', 0, 'x','x')
        if use_bn == 1:
            self.gamma = theano.shared(value = numpy.ones((filter_shape[0],), dtype=theano.config.floatX), name='gamma')
            self.beta = theano.shared(value = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX), name='beta')
            self.linear_shuffle = self.linear.dimshuffle(0, 2, 3, 1)        
            self.linear_res = self.linear_shuffle.reshape( (self.linear.shape[0]*self.linear.shape[2]*self.linear.shape[3],  self.linear.shape[1]))
            bn_output = batch_normalization(inputs = self.linear_shuffle,
    			gamma = self.gamma, beta = self.beta, mean = self.linear_res.mean((0,), keepdims=True),
    			std = T.std(self.linear_res, axis=0), mode='high_mem')
                    
            self.output = T.nnet.relu( bn_output.dimshuffle(0, 3, 1, 2) )    
            self.params = [self.W, self.b, self.gamma, self.beta]
        else:
            self.output = T.nnet.relu(self.linear)
            self.params = [self.W, self.b]

    
        
class SoftMaxOutputLayer():
    def __init__(self, input, n_in, n_out):
        self.p_y_given_x = T.nnet.softmax(input)
        self.y_pred =  T.argmax(self.p_y_given_x, axis=1)
    
    def negative_log_likelihood(self, y):
        cost = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return cost
    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        else:
            return T.mean(T.neq(self.y_pred, y))
        
        