# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 21:26:06 2017

@author: Fengjilan
"""

#import numpy as npy
import logging
import sys
import mxnet as mx

from symbols.vdsr_symbol import vdsr_symbol
from utils.dataio import get_sr_iter
from utils.evaluate_metric import psnr

logging.basicConfig(level=logging.INFO)

root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.INFO)

net=vdsr_symbol(8,32)
mod = mx.mod.Module(symbol=net, 
                    context=mx.gpu(),
                    data_names=['imgin'], 
                    label_names=['loss_imghr'])

#optimizer = mx.optimizer.create(
#        'sgd',
#        learning_rate =0.000001,
#        momentum = 0.9,
#        wd=0.002,
#        lr_scheduler=mx.lr_scheduler.FactorScheduler(9000,0.9))

optimizer = mx.optimizer.create(
        'adagrad',
        learning_rate =0.002,
        wd=0.0005,
        clip_gradient=0.005,
        lr_scheduler=mx.lr_scheduler.FactorScheduler(5000,0.6))

#lr_scale={}
#for arg_name in net.list_arguments():
#    if "conv0" in arg_name or "conv1" in arg_name:
#        lr_scale[arg_name] = 5
#optimizer.set_lr_mult(lr_scale)

initializer = mx.init.Xavier(rnd_type='gaussian', 
                             factor_type="in",
                             magnitude=2)  
model_prefix="checkpoint\\vdsr"
checkpoint = mx.callback.do_checkpoint(model_prefix,period=100)


datadir="E:\\DevProj\\Datasets\\SuperResolution\\SRCNN_Train"
batch_size=64
data_params={"batch_size":batch_size,"crop_size":81,"scale_factor":3,
        "is_train":True,"num_train_img":90,"num_val_img":0,
        "img_type":[".jpg",".bmp"]}
train_iter, _=get_sr_iter(datadir,data_params)

datadir1="E:\\DevProj\\Datasets\\SuperResolution\\Set14"
data_params["is_train"]=False
val_iter=get_sr_iter(datadir1,data_params)

mod.fit(train_iter,
          num_epoch=1000,
          eval_data=val_iter,
          eval_metric=psnr,
          optimizer=optimizer,
          initializer=initializer,          
          batch_end_callback = mx.callback.Speedometer(batch_size, 2000),
          epoch_end_callback=checkpoint)