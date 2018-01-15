# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 07:14:31 2018

@author: galad-loth
"""
import mxnet as mx
from custom_layers import SRLossLayer 


def embed_net(data, num_feature):
    hn1 = mx.sym.Convolution(data=data, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_feature,name="embed_conv0")
    hn1 = mx.sym.Activation(data=hn1, act_type="relu", name="embed_relu0")
    hn0 = mx.sym.Convolution(data=hn1, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_feature,name="embed_conv1")
    hn0 = mx.sym.Activation(data=hn0, act_type="relu", name="embed_relu1")#embd result 
    return hn0

def recov_net(data, num_feature, num_channel, weights, bias):
    recov_res = mx.sym.Convolution(data=data, kernel=(3, 3), stride=(1,1),pad=(1, 1),
                              weight=weights[0],bias=bias[0],
                              num_filter=num_feature,name="recov_conv0")
    recov_res = mx.sym.Activation(data=recov_res, act_type="relu", name="recov_relu0")
    recov_res = mx.sym.Convolution(data=recov_res, kernel=(3, 3), stride=(1,1),pad=(1, 1), 
                             weight=weights[1],bias=bias[1],
                             num_filter=num_channel,name="recov_conv1")
    recov_res = mx.sym.Activation(data=recov_res, act_type="relu", name="recov_relu1")#embd result
    return recov_res
    
    
def get_drcn_symbol(num_recur_layer, num_feature, num_channel):
    imgin = mx.sym.Variable("imgin")
    # embed net
    embed_res=embed_net(imgin, num_feature)
    # (recursive) inference net 
    recur_weight=mx.sym.Variable('recur_weight')
    recur_bias=mx.sym.Variable('recur_bias')
    recov_weight=[]
    recov_bias=[]   
    for i in range(2):
        recov_weight.append(mx.sym.Variable('recov' + '_weight' + str(i)))
        recov_bias.append(mx.sym.Variable('recov' + '_bias' + str(i)))
    infer_res=mx.sym.Convolution(data=embed_res, kernel=(3, 3), stride=(1,1),pad=(1, 1), 
                             weight=recur_weight,bias=recur_bias,
                             num_filter=num_feature,name="recur_conv")
    infer_res = mx.sym.Activation(data=infer_res, act_type="relu", name="recur_relu")  
    recov_res=recov_net(infer_res,num_feature,num_channel, recov_weight, recov_bias)
    if num_recur_layer>1:
        for i in range(num_recur_layer-1):
             infer_res=mx.sym.Convolution(data=infer_res, kernel=(3, 3), 
                                          stride=(1,1),pad=(1, 1), 
                                         weight=recur_weight,bias=recur_bias,
                                         num_filter=num_feature,name="recur_conv")
             infer_res = mx.sym.Activation(data=infer_res, act_type="relu", name="recur_relu") 
             recov_res1=recov_net(infer_res,num_feature,num_channel, recov_weight, recov_bias)
             recov_res=recov_res+recov_res1
    return recov_res
            
if __name__=="__main__":
    net=get_drcn_symbol(6,64,3)
    ex=net.simple_bind(ctx=mx.cpu(), imgin=(1,3,64,64))            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            