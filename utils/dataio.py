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
            img_ds=cv2.resize(img,(ncol/self._scale_factor, nrow/self._scale_factor),
                              interpolation=cv2.INTER_CUBIC)  
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

class LapSRNDataBatch(object):
    def __init__(self, img_lr, img_pryd, pad=0):
        self.data =[mx.nd.array(img_lr)]
        self.label =[]
        for img_gt in img_pryd:
            self.label.append(mx.nd.array(img_gt))
        self.pad = pad     
        
         
class LapSRNDataIter(mx.io.DataIter):
    def __init__(self,datadir,img_list,crop_num,crop_size, num_scales):
        super(LapSRNDataIter, self).__init__()
        self._datadir=datadir
        self._img_list = img_list
        self._crop_num = crop_num
#        self._crop_size=crop_size
        self._num_scales=num_scales
        self.batch_num=len(img_list)
        self.cur_batch=0
        scale_factor=npy.power(2,num_scales)
        imglr_size=crop_size/scale_factor
        imggt_size=imglr_size
        provide_label=[]
        for s in range(num_scales):
            imggt_size=imggt_size*2
            provide_label.append(("loss_s{}_imggt".format(s),
                                  (crop_num,3,imggt_size,imggt_size)))        
        self._provide_label=provide_label
        self._provide_data=zip(["imglr"],[(crop_num,3,imglr_size,imglr_size)])
        self._imglr_size=imglr_size
        self._crop_size=imggt_size
        
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
            sub_img_lr=npy.zeros(self._provide_data[0][1],dtype=npy.float32)
            sub_img_pryd=[]
            for item in self._provide_label:
                sub_img_pryd.append(npy.zeros(item[1],dtype=npy.float32))

            for i in range(self._crop_num):
                nrow_start=npy.random.randint(0,nrow-crop_size)
                ncol_start=npy.random.randint(0,ncol-crop_size)
                img_crop=img[nrow_start:nrow_start+crop_size,
                                ncol_start:ncol_start+crop_size,:] 
                imggt_size=crop_size                
                for s in range(self._num_scales):
                    img_temp=img_crop.astype(npy.float32)
                    img_temp=(img_temp-128)/128.0
                    img_temp = npy.swapaxes(img_temp, 0, 2)
                    img_temp = npy.swapaxes(img_temp, 1, 2)
                    sub_img_pryd[self._num_scales-s-1][i,:,:,:]=img_temp
                    imggt_size=imggt_size/2
                    img_crop=cv2.resize(img_crop,(imggt_size, imggt_size),
                              interpolation=cv2.INTER_CUBIC) 
                img_temp=img_crop.astype(npy.float32)
                img_temp=(img_temp-128)/128.0
                img_temp = npy.swapaxes(img_temp, 0, 2)
                img_temp = npy.swapaxes(img_temp, 1, 2)
                sub_img_lr[i,:,:,:]=img_temp              
                
            return LapSRNDataBatch(sub_img_lr,sub_img_pryd,0)
        else:
            raise StopIteration

def get_lapsrn_iter(datadir,params):
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
        train_iter= LapSRNDataIter(datadir,
                               train_img_list,
                               params["batch_size"],
                               params["crop_size"],
                               params["num_scales"])  
        val_iter = LapSRNDataIter(datadir,
                              val_img_list,
                              params["batch_size"],
                              params["crop_size"],
                              params["num_scales"])    
        return train_iter,val_iter                 
    else:
        test_iter= LapSRNDataIter(datadir,
                              img_list,
                              params["batch_size"],
                              params["crop_size"],
                              params["num_scales"])
        return test_iter
    
if __name__=="__main__":
    datadir=("E:\\DevProj\\Datasets\\SuperResolution\\SR_training_datasets\\General100")
    params={"batch_size":50,"crop_size":80,"num_scales":2,
            "is_train":True,"num_train_img":50,"num_val_img":50,
            "img_type":[".png"]}
    train_iter, val_iter=get_lapsrn_iter(datadir,params)
    
    one_batch=train_iter.next()
    img_lr=one_batch.data[0].asnumpy()
    img_mr=one_batch.label[0].asnumpy()
    img_hr=one_batch.label[1].asnumpy()
    img_lr=img_lr*128+128
    img_mr=img_mr*128+128
    img_hr=img_hr*128+128    
#    
    idx=21
    img1=img_lr[idx,:,:,:].astype(npy.uint8)
    img1 = npy.swapaxes(img1, 0, 2)
    img1 = npy.swapaxes(img1, 0, 1)
#    img1=cv2.cvtColor(img1,cv2.COLOR_YCR_CB2RGB)
    img2=img_mr[idx,:,:,:].astype(npy.uint8)
    img2 = npy.swapaxes(img2, 0, 2)
    img2 = npy.swapaxes(img2, 0, 1)
#    img2=cv2.cvtColor(img2,cv2.COLOR_YCR_CB2RGB)
    img3=img_hr[idx,:,:,:].astype(npy.uint8)
    img3 = npy.swapaxes(img3, 0, 2)
    img3 = npy.swapaxes(img3, 0, 1)
#    
    fig1=plt.figure(1,figsize=(6,2))
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.subplot(1,3,3)
    plt.imshow(img3)
