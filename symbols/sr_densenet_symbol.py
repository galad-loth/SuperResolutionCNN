# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 07:14:31 2018

@author: galad-loth
"""
import mxnet as mx
from custom_layers import SRLossLayer 

def basic_block(data, growth_rate, drop_out, block_name):
    block_data=mx.symbol.BatchNorm(data=data,  fix_gamma=False, eps =1e-5, name =block_name+'bn')
    block_data=mx.sym.Activation(data=block_data, act_type='relu', name=block_name + '_relu')
    block_data = mx.sym.Convolution(data=block_data, num_filter=int(growth_rate), 
                                   kernel=(3,3), stride=(1,1), pad=(1,1),
                                   name=block_name + '_conv')
    if drop_out > 0:
        block_data = mx.symbol.Dropout(data=block_data, p=drop_out, name=block_name + '_dp')
    return block_data  
        
def dense_block(data, num_basic_block,growth_rate,drop_out,block_name):
    for i in range(num_basic_block):
        block_data=basic_block(data,growth_rate,drop_out,block_name+"_layer"+str(i))
        data= mx.symbol.Concat(data, block_data, name=block_name + '_concat'+str(i+1))
    return data
    
def get_srdn_h_symbol(num_channel,num_dense_block,num_basic_block, growth_rate,drop_out=0.0):
    imgin = mx.sym.Variable("imgin")
    #low-level feature learning
    net = mx.sym.Convolution(data=imgin, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=growth_rate,name="embed_conv0")
    net = mx.sym.Activation(data=net, act_type="relu", name="embed_relu0")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=growth_rate,name="embed_conv1")
    net = mx.sym.Activation(data=net, act_type="relu", name="embed_relu1")
    for i in range(num_dense_block):
        net=dense_block(net,num_basic_block,growth_rate,drop_out,block_name="dblock"+str(i))
    net=mx.symbol.Deconvolution(data=net,kernel=(4,4), stride=(2,2),
                                     pad=(1,1),num_filter=growth_rate,no_bias =True,
                                     name="deconv0")   
    net = mx.sym.Activation(data=net, act_type="relu", name="deconv_relu0")
    net=mx.symbol.Deconvolution(data=net,kernel=(4,4), stride=(2,2),
                                pad=(1,1),num_filter=growth_rate,no_bias =True,
                                name="deconv1")   
    net = mx.sym.Activation(data=net, act_type="relu", name="deconv_relu1")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_channel,name="recov_conv0")
    return net
  
def get_srdn_all_symbol(num_channel,num_dense_block,num_basic_block, growth_rate,drop_out=0.0):
    imgin = mx.sym.Variable("imgin")
    #low-level feature learning
    net = mx.sym.Convolution(data=imgin, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=growth_rate,name="embed_conv0")
    net = mx.sym.Activation(data=net, act_type="relu", name="embed_relu0")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=growth_rate,name="embed_conv1")
    net = mx.sym.Activation(data=net, act_type="relu", name="embed_relu1")
    netc=net
    for i in range(num_dense_block):
        net=dense_block(net,num_basic_block,growth_rate,drop_out,block_name="dblock"+str(i))
        netc=mx.symbol.Concat(netc, net, name='aggr_data'+str(i))
    net=mx.sym.Convolution(data=netc, kernel=(1, 1), stride=(1,1),
                             pad=(0, 0), num_filter=256,name="bottleneck_conv")
    net = mx.sym.Activation(data=net, act_type="relu", name="bottleneck_relu")   
    net=mx.symbol.Deconvolution(data=net,kernel=(4,4), stride=(2,2),
                                     pad=(1,1),num_filter=growth_rate,no_bias =True,
                                     name="deconv0")   
    net = mx.sym.Activation(data=net, act_type="relu", name="deconv_relu0")
    net=mx.symbol.Deconvolution(data=net,kernel=(4,4), stride=(2,2),
                                pad=(1,1),num_filter=growth_rate,no_bias =True,
                                name="deconv1")   
    net = mx.sym.Activation(data=net, act_type="relu", name="deconv_relu1")
    net = mx.sym.Convolution(data=net, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_channel,name="recov_conv0")
    return net  
    
if __name__=="__main__":
    net=get_srdn_all_symbol(3,5,8,16)
    ex=net.simple_bind(ctx=mx.cpu(), imgin=(1,3,64,64))  