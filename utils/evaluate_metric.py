import numpy as npy
#import cv2

def psnr(hr_imgs, sr_imgs):
    sr_shape=sr_imgs.shape
    hr_shape=hr_imgs.shape
    cs=(hr_shape[-1]-sr_shape[-1])/2
    rs=(hr_shape[-2]-sr_shape[-2])/2
    hr_imgs=hr_imgs*128+128
    sr_imgs=sr_imgs*128+128
    
    hr_center_imgs=hr_imgs[:,:,rs:rs+sr_shape[-2],cs:cs+sr_shape[-1]]
#    diff=hr_center_imgs-sr_imgs
#    print sr_imgs[0,0,:,:]
    psnr=npy.zeros(hr_shape[0],dtype=npy.float32)
    for i in range(hr_shape[0]):
        img1=sr_imgs[i,:,:,:].astype(npy.uint8)
        img1 = npy.swapaxes(img1, 0, 2)
        img1 = npy.swapaxes(img1, 0, 1)
#        img1=cv2.cvtColor(img1,cv2.COLOR_YCR_CB2BGR)
        img2=hr_center_imgs[i,:,:,:].astype(npy.uint8)
        img2 = npy.swapaxes(img2, 0, 2)
        img2 = npy.swapaxes(img2, 0, 1)
#        img2=cv2.cvtColor(img2,cv2.COLOR_YCR_CB2BGR)
        diff=img2-img1
        mse=npy.mean(diff*diff)
        psnr[i]=10*npy.log10(255*255/mse)
#    print psnr
    return npy.mean(psnr)
    
         
    

        

