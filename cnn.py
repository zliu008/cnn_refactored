# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 16:37:20 2017

@author: zaoliu
"""

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, 
                 image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(2. /(fan_in + fan_out))
        W_value = rng.normal(loc = 0., scale = W_bound, size = filter_shape)
        self.W = theano.shared(W_value, name = 'W', borrow = True)
        conv_out = conv2d(input = self.input, 
                   filters = self.W)        
        
        pooled_out = pool.pool_2d(input = conv_out, 
                                  ds=poolsize, ignore_border=True)                            
        
        b_bound = numpy.sqrt(2. /fan_out)
        b_value = rng.normal(loc = 0, scale = b_bound, size=(filter_shape[0],))
        self.b = theano.shared(b_value, name = 'b', borrow  = True)
        
        #self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x','x')  )      
        self.params = [self.W, self.b]

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):
        self.input = input
        W_bound = numpy.sqrt( 2. /(n_in + n_out))
        W_value = rng.normal(0., W_bound, (n_in, n_out))
        self.W = theano.shared(W_value, name = 'W', borrow = True)
        
        b_bound = numpy.sqrt( 2. / n_out)
        b_value = rng.normal(loc = 0., scale = b_bound, size=(n_out,))
        self.b = theano.shared(b_value, name = 'b', borrow = True)
        
        lin_out = T.dot(input, self.W) + self.b
        self.p_y_given_x = activation(lin_out)
        self.params = [self.W, self.b]
#
class OutputLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, activation):
        HiddenLayer.__init__(self, rng, input, n_in, n_out, activation)
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
    
        
class LeNets(object):
    def __init__(self, rng, batch_size=500, nkerns=[20, 50], 
                         each_imag_shape = (28, 28), poolsize=(2, 2),
                         fH = 5, fW = 5):
        self.batch_size = batch_size
    
        # start-snippet-1
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels 
        
        img_shape_l1 = (batch_size, ) + (1,) + each_imag_shape
        conv_pool_l1_in = self.x.reshape( img_shape_l1 )
        self.conv_pool_l1 =  LeNetConvPoolLayer(
                        rng = rng,
                        input = conv_pool_l1_in,
                        image_shape = img_shape_l1,
                        filter_shape = (nkerns[0], 1, fH, fW),
                        poolsize= poolsize                   
                        )
        
        img_shape_l2 = (batch_size,) + (nkerns[0],) \
        + ((each_imag_shape[0] - fH + 1)/poolsize[0],  
           (each_imag_shape[0] - fW + 1)/poolsize[1])
        self.conv_pool_l2 =  LeNetConvPoolLayer(
                        rng = rng,
                        input = self.conv_pool_l1.output,
                        image_shape = img_shape_l2,
                        filter_shape = (nkerns[1], nkerns[0], fH, fW),
                        poolsize= poolsize                   
                        )
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        #conv_pool_l2.output was (500, 50, 4, 4)
        hid_input = self.conv_pool_l2.output.flatten(2)    
        self.hid_layer = HiddenLayer(
            rng,
            input = hid_input,
            n_in = nkerns[1] * (img_shape_l2[2] - fH + 1)/2 * (img_shape_l2[3] - fW + 1)/2,
            n_out = 500
        )
        
        self.out_layer = OutputLayer(
            rng,
            input = self.hid_layer.p_y_given_x, 
            n_in = 500, 
            n_out = 10, 
            activation = T.nnet.softmax
        )
        
        self.L1 = T.sum(T.abs_(self.conv_pool_l1.W)) + \
                 T.sum(T.abs_(self.conv_pool_l2.W)) + \
                 T.sum(T.abs_(self.hid_layer.W)) + \
                 T.sum(T.abs_(self.out_layer.W))
                 
    def cnn_setup_test(self, data_set_x, data_set_y, learning_rate=0.1):

        index = T.lscalar()  
        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [index],
            self.out_layer.errors(self.y),
            givens={
                self.x: data_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: data_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )
    def cnn_run_test(self, index):
        err =  self.test_model(index)        
        return err
        
    def cnn_setup_valid(self, data_set_x, data_set_y, learning_rate=0.1):

        index = T.lscalar()  
        # create a function to compute the mistakes that are made by the model
        self.valid_model = theano.function(
            [index],
            self.out_layer.errors(self.y),
            givens={
                self.x: data_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: data_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )
    def cnn_run_valid(self, index):
        err =  self.valid_model(index)        
        return err


    def cnn_setup_infer(self, data_set_x, data_set_y, learning_rate=0.1): 
        index = T.lscalar()
        self.infer_model = theano.function(
            [index],
            self.out_layer.y_pred,
            givens={
                self.x: data_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: data_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )
    def cnn_run_infer(self, index):
        pred = self.infer_model(index)
        return pred
        
    def cnn_setup_train(self, data_set_x, data_set_y, learning_rate=0.1, L1Reg = 0.0):  
        index = T.lscalar()
        # the cost we minimize during training is the NLL of the model
        cost = self.out_layer.negative_log_likelihood(self.y) +\
               L1Reg * self.L1
    
        # create a list of all model parameters to be fit by gradient descent
        self.params = self.out_layer.params + self.hid_layer.params \
        + self.conv_pool_l2.params + self.conv_pool_l1.params
    
        # create a list of gradients for all model parameters
        grads = T.grad(cost, self.params)
    
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]
    
        self.train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                self.x: data_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: data_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )
    def cnn_run_train(self, index):
        cost = self.train_model(index)
        return cost
        
        