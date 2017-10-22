# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:49:52 2017

@author: zaoliu
"""
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from cnn_util import LeNetConvLayer, SoftMaxOutputLayer



class LeNets(object):
    def __init__(self, rng, batch_size=500, nkerns=[20, 50, 70], poolsize=(2,2),
                         each_imag_shape = (28, 28),
                         fH = 5, fW = 5):
        self.batch_size = batch_size
    
        # start-snippet-1
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels 
        
        img_shape_l1 = (batch_size, ) + (1,) + each_imag_shape
        conv_l1_in = self.x.reshape( img_shape_l1 )
        self.conv_l1 =  LeNetConvLayer(
                        rng = rng,
                        input = conv_l1_in,
                        image_shape = img_shape_l1,
                        filter_shape = (nkerns[0], 1, fH, fW),
                        )
        img_shape_convl1_out = (batch_size,) + (nkerns[0],) \
        + ((each_imag_shape[0] - fH + 1),  \
           (each_imag_shape[0] - fW + 1))

        pooled_out_l1 = pool.pool_2d(input = self.conv_l1.output,
                                  ds=poolsize, ignore_border=True)                                   
        
        img_shape_l2 = (batch_size,) + (nkerns[0],) \
        + (img_shape_convl1_out[2]/poolsize[0], img_shape_convl1_out[3]/poolsize[1])
        
        self.conv_l2 = LeNetConvLayer(   
                        rng = rng,
                        input = pooled_out_l1,
                        image_shape = img_shape_l2,
                        filter_shape = (nkerns[1], nkerns[0], fH, fW)                 
                        )
        img_shape_convl2_out = (batch_size,) + (nkerns[1],) \
        + ((img_shape_l2[2] - fH + 1),  \
           (img_shape_l2[3] - fW + 1))
        
         #simple inception, 5x5 convolution + 1x1 convolution // max pool + 1x1 convolution
        self.conv_l3 = LeNetConvLayer(   
                        rng = rng,
                        input = self.conv_l2.output,
                        image_shape = img_shape_convl2_out,
                        filter_shape = (nkerns[2], nkerns[1], fH, fW)                 
                        )
        
        pooled_l3 = pool.pool_2d(input = self.conv_l2.output,
                                  ds=poolsize, ignore_border=True)   
        #1x1 convolution                                  
        self.conv_l3_1 = LeNetConvLayer(   
                rng = rng,
                input = self.conv_l3.output,
                image_shape = (batch_size, nkerns[2],img_shape_convl2_out[2] - fH + 1, \
                img_shape_convl2_out[3] - fW + 1),
                #filter_shape = (nkerns[2]/2, nkerns[2], 1, 1) 
                filter_shape = (nkerns[1]/2, nkerns[2], 1, 1)     # make output map number nkerns[1]/2                  
                )
        self.conv_l3_2 = LeNetConvLayer(   
                rng = rng,
                input = pooled_l3,
                image_shape = (batch_size, nkerns[1],img_shape_convl2_out[2]/2, \
                img_shape_convl2_out[3]/2),
                filter_shape = (nkerns[1]/2, nkerns[1], 1, 1)                 
                )
                                  
        inceptOut1 = T.concatenate([self.conv_l3_1.output, self.conv_l3_2.output], axis = 1)                         
        
        img_shape_l3 = (batch_size,) + (nkerns[1],) \
        + (img_shape_convl2_out[2]/poolsize[0], img_shape_convl2_out[3]/poolsize[1])                
                                               
        #simple inception, 3x3 convolution + 1x1 convolution // max poole + 1x1 convolution
        self.conv_l4 = LeNetConvLayer(   
                        rng = rng,
                        input = inceptOut1,
                        image_shape = img_shape_l3,
                        filter_shape = (nkerns[2], nkerns[1], 3, 3)                 
                        )
        pool_l4 =  pool.pool_2d(input = inceptOut1, ds = (2,2),
                        ignore_border = True, mode = 'average_exc_pad')  
                
        #1x1 convolution
        self.conv_l4_1 = LeNetConvLayer(   
                rng = rng,
                input = self.conv_l4.output,
                image_shape = (batch_size, nkerns[2],2,2),
                filter_shape = (6, nkerns[2], 1, 1)                 
                )
        self.conv_l4_2 = LeNetConvLayer(   
                rng = rng,
                input = pool_l4,
                image_shape = (batch_size, nkerns[1],2,2),
                filter_shape = (4, nkerns[1], 1, 1)                 
                )
        
        inceptOut2 = T.concatenate([self.conv_l4_1.output, self.conv_l4_2.output], axis = 1) 
        
        avgPool = pool.pool_2d(input = inceptOut2, ds = (2,2),
                        ignore_border = True, mode = 'average_exc_pad')
        self.out_layer = SoftMaxOutputLayer(
            input = avgPool.flatten(2),
            n_in = 10, 
            n_out = 10
        )
        
        self.L1 = T.sum(T.abs_(self.conv_l1.W)) + \
                 T.sum(T.abs_(self.conv_l2.W)) + \
                 T.sum(T.abs_(self.conv_l3.W)) +\
                 T.sum(T.abs_(self.conv_l3_1.W)) +\
                 T.sum(T.abs_(self.conv_l3_2.W)) +\
                 T.sum(T.abs_(self.conv_l4.W)) +\
                 T.sum(T.abs_(self.conv_l4_1.W)) +\
                 T.sum(T.abs_(self.conv_l4_2.W))
            
            
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
        self.params = self.conv_l4_2.params + self.conv_l4_1.params + self.conv_l4.params \
        + self.conv_l3.params + self.conv_l3_1.params + self.conv_l3_2.params\
        + self.conv_l2.params + self.conv_l1.params
    
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