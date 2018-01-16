# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 07:14:31 2018

@author: galad-loth
"""
import mxnet as mx
from custom_layers import SRLossLayer 

def recur_block(data, num_res_unit, conv_weights, conv_bias, num_feature):
    rb = mx.symbol.BatchNorm(data = data, eps = 0.001, name ='recur_batchnorm0')
    rb = mx.sym.Activation(data=rb, act_type="relu", name="recur_relu0")
    rb = mx.sym.Convolution(data=rb, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_feature,
                             weight=conv_weights[0], bias=conv_bias[0],name="recur_conv0")
    
    ru = mx.symbol.BatchNorm(data = rb, eps = 0.001, name ='recur_batchnorm1')
    ru = mx.sym.Activation(data=ru, act_type="relu", name="recur_relu1")
    ru = mx.sym.Convolution(data=ru, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_feature,
                             weight=conv_weights[1], bias=conv_bias[1],name="recur_conv1")
    ru = mx.symbol.BatchNorm(data = ru, eps = 0.001, name ='recur_batchnorm2')
    ru = mx.sym.Activation(data=ru, act_type="relu", name="recur_relu2")
    ru = mx.sym.Convolution(data=ru, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_feature,
                             weight=conv_weights[1], bias=conv_bias[1],name="recur_conv2")    
    ru=ru+rb
    if num_res_unit>1:
        for i in range(1,num_res_unit):
            ru = mx.symbol.BatchNorm(data = ru, eps = 0.001, name ='recur_batchnorm'+str(2*i+1))
            ru = mx.sym.Activation(data=ru, act_type="relu", name="recur_relu"+str(2*i+1))
            ru = mx.sym.Convolution(data=ru, kernel=(3, 3), stride=(1,1),
                                    pad=(1, 1), num_filter=num_feature,
                                    weight=conv_weights[2*i+1], bias=conv_bias[2*i+1],
                                    name="recur_conv"+str(2*i+1))
            ru = mx.symbol.BatchNorm(data = ru, eps = 0.001, name ='recur_batchnorm'+str(2*i+2))
            ru = mx.sym.Activation(data=ru, act_type="relu", name="recur_relu"+str(2*i+2))
            ru = mx.sym.Convolution(data=ru, kernel=(3, 3), stride=(1,1),
                                    pad=(1, 1), num_filter=num_feature,
                                    weight=conv_weights[2*i+2], bias=conv_bias[2*i+2],
                                    name="recur_conv"+str(2*i+2)) 
            ru=ru+rb
    return ru
    
def get_drrn_symbol(num_recur_block,num_res_unit,num_feature,num_channel):
    imgin = mx.sym.Variable("imgin")
    net=mx.sym.Convolution(data=imgin, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_feature,name="pre_conv0")
    conv_weight=[]
    conv_bias=[]  
    conv_weight.append(mx.sym.Variable('recur_weight0'))
    conv_bias.append(mx.sym.Variable('recur_bias0'))
    for i in range(num_res_unit):
        conv_weight.append(mx.sym.Variable('recur' + '_weight' + str(2*i+1)))
        conv_bias.append(mx.sym.Variable('recur' + '_bias' + str(2*i+1)))
        conv_weight.append(mx.sym.Variable('recur' + '_weight' + str(2*i+2)))
        conv_bias.append(mx.sym.Variable('recur' + '_bias' + str(2*i+2)))
    for i in range(num_recur_block):
        net=recur_block(net,num_res_unit,conv_weight,conv_bias,num_feature)
    
    net=mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_channel,name="post_conv0")
    net=net+imgin
    return net
    
if __name__=="__main__":
    net=get_drrn_symbol(4,4,128,3)
    ex=net.simple_bind(ctx=mx.cpu(), imgin=(1,3,64,64))  
    
    


 