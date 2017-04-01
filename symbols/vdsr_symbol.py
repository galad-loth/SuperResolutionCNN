# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 21:00:20 2017

@author: Fengjilan
"""
import mxnet as mx
#import numpy as npy 
from custom_layers import SRLossLayer

def vdsr_symbol(num_min_layer, num_filter):
    imgin = mx.sym.Variable("imgin")
    net = mx.sym.Convolution(data=imgin, kernel=(5, 5), stride=(1,1),
                             pad=(2, 2), num_filter=num_filter,name="conv0")
    net = mx.sym.Activation(data=net, act_type="relu", name="relu0")
    for i in range(num_min_layer):
        net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_filter,name="conv{}".format(i+1))
        net = mx.sym.Activation(data=net, act_type="relu", name="relu{}".format(i+1))
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=3,name="conv_out")  
    loss=SRLossLayer()
    net=loss(conv_res=net,imglr=imgin, name="loss")
    return net 
    
    
if __name__=="__main__":
    net=vdsr_symbol(8,32)
    ex=net.simple_bind(ctx=mx.cpu(), imgin=(5,3,33,33),loss_imghr=(5,3,33,33))