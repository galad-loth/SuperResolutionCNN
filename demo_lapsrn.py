# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:45:31 2017

@author: galad-loth
"""

import logging
import sys
import numpy as npy
import cv2
import mxnet as mx

from symbols.lapsrn_symbol import lapsrn_symbol
from utils.dataio import get_lapsrn_iter,LapSRNDataBatch
from utils.filters import upsample_filt
from utils.evaluate_metric import psnr

logging.basicConfig(level=logging.INFO)

root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.INFO)

def train_lapsrn():
    net=lapsrn_symbol(2,8,64)
    ctx=mx.gpu()
    mod = mx.mod.Module(symbol=net, 
                    context=ctx,
                    data_names=['imglr'], 
                    label_names=['loss_s0_imggt','loss_s1_imggt'])
    
    optimizer = mx.optimizer.create(
        'adagrad',
        learning_rate =0.002,
        wd=0.0005,
        clip_gradient=0.005,
        lr_scheduler=mx.lr_scheduler.FactorScheduler(5000,0.6))
    
    initializer = mx.init.Xavier(rnd_type='gaussian', 
                                 factor_type="in",
                                 magnitude=2)  
    model_prefix="checkpoint\\lapsrn"
    checkpoint = mx.callback.do_checkpoint(model_prefix,period=100)
    
    datadir="E:\\DevProj\\Datasets\\SuperResolution\\SR_training_datasets\\General100"
    batch_size=64
    data_params={"batch_size":batch_size,"crop_size":80,"num_scales":2,
            "is_train":True,"num_train_img":100,"num_val_img":0,
            "img_type":[".jpg",".png"]}
    train_iter, _=get_lapsrn_iter(datadir,data_params)
    
    datadir1="E:\\DevProj\\Datasets\\SuperResolution\\SR_testing_datasets\\Set14\\GT"
    data_params["is_train"]=False
    val_iter=get_lapsrn_iter(datadir1,data_params)
    
    
    arg_names=net.list_arguments()
    arg_shapes, _, _ = net.infer_shape(imglr=train_iter.provide_data[0][1])
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) 
                            if x[0].find("deconv")!=-1])
    arg_params={}
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = npy.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        arg_params[k] = mx.nd.array(initw, ctx)
    
    
    mod.fit(train_iter,
              num_epoch=1000,
              eval_data=val_iter,
              eval_metric=psnr,
              optimizer=optimizer,
              initializer=initializer,  
              arg_params=arg_params,
              allow_missing=True,
              batch_end_callback = mx.callback.Speedometer(batch_size, 2000),
              epoch_end_callback=checkpoint)
 
def img_preprocess(img, num_scales):
    nh,nw,nc=img.shape
    scale_factor=npy.power(2,num_scales)
    nh_lr=nh/scale_factor
    nw_lr=nw/scale_factor
    nh_hr=nh_lr*scale_factor
    nw_hr=nw_lr*scale_factor
    img_crop=img[:nh_hr,:nw_hr,:]
    
    img_lr=npy.zeros((1,nc,nh_lr,nw_lr),dtype=npy.float32)
    img_pryd=[]
    nh_pryd=nh_hr
    nw_pryd=nw_hr
    for s in range(num_scales):
        img_temp=img_crop.astype(npy.float32)
        img_temp=(img_temp-128)/128.0
        img_temp = npy.swapaxes(img_temp, 0, 2)
        img_temp = npy.swapaxes(img_temp, 1, 2)
        img_pryd.append(img_temp[npy.newaxis,:,:,:])
        nh_pryd=nh_pryd/2
        nw_pryd=nw_pryd/2
        img_crop=cv2.resize(img_crop,(nw_pryd, nh_pryd),
                  interpolation=cv2.INTER_CUBIC) 
    img_temp=img_crop.astype(npy.float32)
    img_temp=(img_temp-128)/128.0
    img_temp = npy.swapaxes(img_temp, 0, 2)
    img_temp = npy.swapaxes(img_temp, 1, 2)
    img_lr[0,:,:,:]=img_temp 
    img_pryd=img_pryd[-1::-1]

    return img_lr,img_pryd
  
def img_recover(img):
    img1=img[0,:,:,:]
    img1 = npy.swapaxes(img1, 0, 2)
    img1 = npy.swapaxes(img1, 0, 1)
    img1=npy.maximum(-1,npy.minimum(1,img1))
    img1=img1*128.0+128.0    
    return img1
    
def test_lapsrn():
    img=cv2.imread(("E:\\DevProj\\Datasets\\SuperResolution\\SR_testing_datasets"
                    "\\Set14\\GT\\zebra.png"),cv2.IMREAD_COLOR)
    nh,nw,nc=img.shape
#    imghr=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)    
    img_lr, img_pryd=img_preprocess(img,2)
    one_batch=LapSRNDataBatch(img_lr,img_pryd)
    
    net, arg_params, aux_params = mx.model.load_checkpoint("checkpoint\\lapsrn", 100)
    mod = mx.mod.Module(symbol=net, context=mx.gpu())
    
    provide_data=[('imglr', img_lr.shape)]
    provide_label=[]
    for s in range(2):
        provide_label.append(("loss_s{}_imggt".format(s),img_pryd[s].shape))   
    mod.bind(for_training=False, 
             data_shapes=provide_data,
             label_shapes=provide_label)
    mod.set_params(arg_params, aux_params,allow_missing=True)  

    
    mod.forward(one_batch)
    img_sr=mod.get_outputs()
    
#    img_sr=img_recover(img_sr)
    img_lr=img_recover(img_lr)
    img_hr=img_recover(img_pryd[-1])
    cv2.imwrite("results\\lapsrn_imglr.bmp",img_lr)
    cv2.imwrite("results\\lapsrn_imghr.bmp",img_hr)
    for s in range(2):
        img_temp=img_recover(img_sr[s].asnumpy())
        cv2.imwrite("results\\lapsrn_imgsr{}.bmp".format(s),img_temp)
    
if __name__=="__main__":
#    train_lapsrn()
    test_lapsrn()
