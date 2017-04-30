# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 12:09:40 2017

@author: galad-loth
"""

import mxnet as mx
from custom_layers import CharbonnierLossLayer

def lapsrn_scale_op(img,feat,scale,num_conv_layer,num_feat):
    for k in range(num_conv_layer):
        feat = mx.symbol.Convolution(data=feat, kernel=(3, 3), stride=(1,1),
                             pad=(1, 1), num_filter=num_feat,
                             name="conv_s{}_n{}".format(scale,k))
        feat = mx.symbol.Activation(data=feat, act_type="relu",
                                 name="relu_s{}_n{}".format(scale,k))
    feat=mx.symbol.Deconvolution(data=feat,kernel=(4,4), stride=(2,2),
                                     pad=(1,1),num_filter=num_feat,no_bias =True,
                                     name="deconv_feat_s{}".format(scale))
    img_residual=mx.symbol.Convolution(data=feat, kernel=(3,3), stride=(1,1),
                                 pad=(1,1), num_filter=3,
                                 name="conv_res_s{}".format(scale))
    img_upscale=mx.symbol.Deconvolution(data=img,kernel=(4,4), stride=(2,2),
                                     pad=(1,1),num_filter=3,no_bias =True,
                                     name="deconv_img_s{}".format(scale)) 
    img_recover=img_residual+img_upscale
    return img_recover,feat   

def lapsrn_symbol(num_scales,num_conv_layer,num_feat):
    imglr=mx.symbol.Variable("imglr")
    imgrec,feat=lapsrn_scale_op(imglr,imglr,0,num_conv_layer,num_feat)
    loss_layer=CharbonnierLossLayer(1e-3)
    loss=loss_layer(imgrec=imgrec, name="loss_s0")
    if num_scales>1:
        for k in range(1,num_scales):
            imgrec,feat=lapsrn_scale_op(imgrec,feat,k,num_conv_layer,num_feat)
            loss_layer_k=CharbonnierLossLayer(1e-3)
            loss_k=loss_layer_k(imgrec=imgrec, name="loss_s{}".format(k))
            loss=mx.symbol.Group([loss,loss_k])
        
    return loss
        
if __name__=="__main__":
    net=lapsrn_symbol(3,5,64)
    ex=net.simple_bind(ctx=mx.cpu(),imglr=(1,3,64,64))
    
