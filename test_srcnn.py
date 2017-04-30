# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:44:49 2017

@author:  galad-loth
"""
import numpy as npy
import mxnet as mx
import cv2
from utils.dataio import SRDataBatch

def get_low_res_img(img_hr, scale):
    img_shape=img_hr.shape
    img_ds=cv2.resize(img_hr,(img_shape[1]/scale, img_shape[0]/scale),
                      interpolation=cv2.INTER_CUBIC)  
    img_lr=cv2.resize(img_ds,(img_shape[1], img_shape[0]),
                      interpolation=cv2.INTER_CUBIC)  
    return img_lr  
    
    
def img_preprocess(img):
    img1=(img-128.0)/128.0         
    img1 = npy.swapaxes(img1, 0, 2)
    img1 = npy.swapaxes(img1, 1, 2)
    img1 = img1[npy.newaxis, :] 
    return img1
  
def img_recover(img):
    img1=img[0,:,:,:]
    img1 = npy.swapaxes(img1, 0, 2)
    img1 = npy.swapaxes(img1, 0, 1)
    img1=npy.maximum(-1,npy.minimum(1,img1))
    img1=img1*128.0+128.0    
#    img1=img1.astype(npy.uint8)
#    img1=cv2.cvtColor(img1,cv2.COLOR_YCR_CB2BGR)
    return img1
    
if __name__=="__main__":
    img=cv2.imread("E:\\DevProj\\Datasets\\SuperResolution\\Set14\\monarch.bmp",
                    cv2.IMREAD_COLOR)
    nh,nw,nc=img.shape
#    imghr=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)    
    imglr=get_low_res_img(img,3)  
    imghr=img.astype(npy.float32)
    imglr=imglr.astype(npy.float32)
    
    net, arg_params, aux_params = mx.model.load_checkpoint("checkpoint\\vdsr", 100)
    mod = mx.mod.Module(symbol=net, context=mx.gpu())
    mod.bind(for_training=False, 
             data_shapes=[('imgin', (1,nc,nh,nw))],
             label_shapes=[('loss_imghr',(1,nc,nh,nw))])
    mod.set_params(arg_params, aux_params,allow_missing=True)  
    
    imghr=img_preprocess(imghr)
    imglr=img_preprocess(imglr)
    one_batch=SRDataBatch(imglr,imghr)
    mod.forward(one_batch)
    imgsr=mod.get_outputs()[0].asnumpy()
    
    imgsr=img_recover(imgsr)
    imglr=img_recover(imglr)
    imghr=img_recover(imghr)
    cv2.imwrite("results\\imgsr.bmp",imgsr)
    cv2.imwrite("results\\imghr.bmp",imghr)
    cv2.imwrite("results\\imglr.bmp",imglr)
