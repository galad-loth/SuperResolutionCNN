# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:41:44 2017

@author: galad-loth
"""

import mxnet as mx
import numpy as npy 

class SRLossLayer(mx.operator.NumpyOp):
    def __init__(self):
        super(SRLossLayer, self).__init__(False)
        
    def list_arguments(self):
        return ['conv_res','imglr','imghr']
        
    def list_outputs(self):
        return ['imgout']
        
    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        imghr_shape=in_shape[1]
        return [data_shape, imghr_shape, imghr_shape],[data_shape]
        
    def forward(self, in_data, out_data):
        conv_res=in_data[0]
        imglr=in_data[1]        
        imgout=out_data[0]
        
        shapein=imglr.shape
        shapeout=imgout.shape
        cs=(shapein[-1]-shapeout[-1])/2
        rs=(shapein[-2]-shapeout[-2])/2  
        imgout[:]=conv_res+imglr[:,:,rs:rs+shapeout[-2],cs:cs+shapeout[-1]]
        
    def backward(self, out_grad, in_data, out_data, in_grad):        
        conv_res=in_data[0]
        imglr=in_data[1] 
        imghr=in_data[2]
        dx=in_grad[0]
        
        img_diff=imghr-imglr
        
        shapein=imglr.shape
        shapeout=conv_res.shape
        cs=(shapein[-1]-shapeout[-1])/2
        rs=(shapein[-2]-shapeout[-2])/2 
        img_diff_center=img_diff[:,:,rs:rs+shapeout[-2],cs:cs+shapeout[-1]]        
        
        d=conv_res-img_diff_center
        batch_size=conv_res.shape[0]
        dx[:]=d/batch_size


class CharbonnierLossLayer(mx.operator.NumpyOp):
    def __init__(self, epsilon):
        super(CharbonnierLossLayer, self).__init__(False)
        self.epsilon=epsilon
        
    def list_arguments(self):
        return ['imgrec','imggt']        
                
    def list_outputs(self):
        return ['imgout']
        
    def infer_shape(self, in_shape):
        data_shape=in_shape[0]
        return [data_shape, data_shape],[data_shape]
        
    def forward(self, in_data, out_data):
        imgrec=in_data[0]        
        imgout=out_data[0]
        imgout[:]=imgrec
    
    def backward(self, out_grad, in_data, out_data, in_grad):        
        imgrec=in_data[0]
        imggt=in_data[1] 
        dx=in_grad[0]
        
        diff=imgrec-imggt
        batch_size=imgrec.shape[0]
        diff_norm=npy.sqrt(diff*diff+self.epsilon*self.epsilon)
        dx[:]=diff/diff_norm/batch_size
        
        
