# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:41:44 2017

@author: galad-loth
"""

import mxnet as mx
#import numpy as npy 
from custom_layers import SRLossLayer

def srcnn_symbol(n1,n2,n3):
    imgin = mx.sym.Variable("imgin")
    net = mx.sym.Convolution(data=imgin, kernel=(9, 9), stride=(1,1),
                             pad=(0, 0), num_filter=n1,name="conv0")
    net = mx.sym.Activation(data=net, act_type="relu", name="relu0")
    net = mx.sym.Convolution(data=net, kernel=(1, 1), stride=(1,1),
                             pad=(0, 0), num_filter=n2,name="conv1")   
    net = mx.sym.Activation(data=net, act_type="relu", name="relu1")
    net = mx.sym.Convolution(data=net, kernel=(5, 5), stride=(1,1),
                             pad=(0, 0), num_filter=n3,name="conv2")  
    loss=SRLossLayer()
    net=loss(conv_res=net,imglr=imgin, name="loss")
    return net 
    
if __name__=="__main__":
    net=srcnn_symbol(64,32,3)
    ex=net.simple_bind(ctx=mx.cpu(), imgin=(5,3,33,33),loss_imghr=(5,3,33,33))
    