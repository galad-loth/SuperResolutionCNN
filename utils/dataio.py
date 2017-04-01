# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 19:27:34 2017

@author: galad-loth
"""

import os


import numpy as npy
import cv2
import mxnet as mx
from matplotlib import pyplot as plt

class SRDataBatch(object):
    def __init__(self, img_lr, img_hr, pad=0):
        self.data =[mx.nd.array(img_lr)]
        self.label =[mx.nd.array(img_hr)]
        self.pad = pad     
        
class SRDataIter(mx.io.DataIter):
    def __init__(self,datadir,img_list,crop_num,crop_size, scale_factor):
        super(SRDataIter, self).__init__()
        self._datadir=datadir
        self._img_list = img_list
        self._crop_num = crop_num
        self._crop_size=crop_size
        self._scale_factor=scale_factor
        self.batch_num=len(img_list)
        self.cur_batch=0
        self._provide_data=zip(["imgin"],[(crop_num,3,crop_size,crop_size)])
        self._provide_label=zip(["loss_imghr"],[(crop_num,3,crop_size,crop_size)])
        
    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0        

    def __next__(self):
        return self.next()
  
    @property
    def provide_data(self):      
        return self._provide_data

    @property
    def provide_label(self):       
        return self._provide_label
    
    def next(self):
        nrow=0
        ncol=0
        crop_size=self._crop_size
        while (nrow<crop_size or ncol<crop_size) \
              and self.cur_batch < self.batch_num:                  
            img_path=os.path.join(self._datadir, self._img_list[self.cur_batch])      
            img=cv2.imread(img_path, cv2.IMREAD_COLOR)
#            img=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
            nrow,ncol=img.shape[0:2]
            self.cur_batch+=1
        if self.cur_batch < self.batch_num:
#            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_ds=cv2.resize(img,(ncol/3, nrow/3),interpolation=cv2.INTER_CUBIC)  
            img_lr=cv2.resize(img_ds,(ncol, nrow),interpolation=cv2.INTER_CUBIC)  
             
            img=img.astype(npy.float32)
            img_lr=img_lr.astype(npy.float32)
            
            sub_img_lr=npy.zeros(self._provide_data[0][1],dtype=npy.float32)
            sub_img_hr=npy.zeros(self._provide_label[0][1],dtype=npy.float32)
            for i in range(self._crop_num):
                nrow_start=npy.random.randint(0,nrow-crop_size)
                ncol_start=npy.random.randint(0,ncol-crop_size)
                img_crop=img_lr[nrow_start:nrow_start+crop_size,
                                ncol_start:ncol_start+crop_size,:]              
                img_crop=(img_crop-128) /128.0           
                img_crop = npy.swapaxes(img_crop, 0, 2)
                img_crop = npy.swapaxes(img_crop, 1, 2)
                sub_img_lr[i,:,:,:]=img_crop
                
                img_crop=img[nrow_start:nrow_start+crop_size,
                                ncol_start:ncol_start+crop_size,:]
                img_crop=(img_crop-128) /128.0                  
                img_crop = npy.swapaxes(img_crop, 0, 2)
                img_crop = npy.swapaxes(img_crop, 1, 2)
                sub_img_hr[i,:,:,:]=img_crop
            return SRDataBatch(sub_img_lr,sub_img_hr,0)
        else:
            raise StopIteration

def get_sr_iter(datadir,params):
    file_list=os.listdir(datadir)
    num_img=0
    img_list=[]
    for fname in file_list:
        for ftype in params["img_type"]:
            if fname.find(ftype)!=-1:
                num_img+=1
                img_list.append(fname)
                break
    if params["is_train"]:
        num_train_img=npy.minimum(num_img,params["num_train_img"])
        num_val_img=npy.minimum(num_img-num_train_img, params["num_val_img"])
        idx_rand=npy.random.permutation(num_img)
        train_img_list=[img_list[idx_rand[i]] for i in range(num_train_img)]
        val_img_list=[img_list[idx_rand[num_train_img+i]]
                        for i in range(num_val_img)]
        train_iter= SRDataIter(datadir,
                               train_img_list,
                               params["batch_size"],
                               params["crop_size"],
                               params["scale_factor"])  
        val_iter = SRDataIter(datadir,
                              val_img_list,
                              params["batch_size"],
                              params["crop_size"],
                              params["scale_factor"])    
        return train_iter,val_iter                 
    else:
        test_iter= SRDataIter(datadir,
                              img_list,
                              params["batch_size"],
                              params["crop_size"],
                              params["scale_factor"])
        return test_iter

    
    
if __name__=="__main__":
    datadir=("E:\\DevProj\\Datasets\\PascalVoc\\2012\\VOCdevkit\\VOC2012\\JPEGImages")
    params={"batch_size":50,"crop_size":81,"scale_factor":3,
            "is_train":True,"num_train_img":2000,"num_val_img":50,
            "img_type":[".jpg"]}
    train_iter, val_iter=get_sr_iter(datadir,params)
    
    one_batch=train_iter.next()
    img_lr=one_batch.data[0].asnumpy()
    img_hr=one_batch.label[0].asnumpy()
    img_lr=img_lr*128+128
    img_hr=img_hr*128+128    
    
    idx=49
    img1=img_lr[idx,:,:,:].astype(npy.uint8)
    img1 = npy.swapaxes(img1, 0, 2)
    img1 = npy.swapaxes(img1, 0, 1)
    img1=cv2.cvtColor(img1,cv2.COLOR_YCR_CB2RGB)
    img2=img_hr[idx,:,:,:].astype(npy.uint8)
    img2 = npy.swapaxes(img2, 0, 2)
    img2 = npy.swapaxes(img2, 0, 1)
    img2=cv2.cvtColor(img2,cv2.COLOR_YCR_CB2RGB)
    
    fig1=plt.figure(1,figsize=(4,2))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    plt.imshow(img2)