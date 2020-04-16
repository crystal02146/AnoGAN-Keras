# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:22:44 2020

@author: Tseng Yi-Yao
"""

from keras.utils. generic_utils import Progbar
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Flatten
from keras.layers.core import Activation
from keras.optimizers import Adam
import keras.backend as K
import math, cv2
import numpy as np
from os import listdir

import matplotlib.pyplot as plt

import tensorflow as tf
import time

import argparse

#限定GPU使用量

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from DCGAN import DCGAN
from AnoGAN import ANOGAN
from difference import *

#反正規化 0~1 變成 0~255之間

def denormalize(X):

    for i in range (0,X.shape[1]): #Height 
        for j in range (0,X.shape[0]): # Width
        
            if(X[i,j,0] < 0):
                X[i,j,0] = 0
            if(X[i,j,1] < 0):
                X[i,j,1] = 0
            if(X[i,j,2] < 0):
                X[i,j,2] = 0

    
    return (X * 255.0).astype(dtype=np.uint8)


#AnoGAN model 
class anogan_model(object):
    
    #K.clear_session()
    
    def __init__(self,model_path,input_dim,image_shape):
    
        anogan_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)#AnoGAN 優化器
        
        
        dcgan = DCGAN(input_dim, image_shape)#宣告Generator 與 descriminater 空model
        
        # load weights
        dcgan.load_weights(model_path+"Generator.h5",model_path+"Discriminator.h5")#載入權重

        self.anogan = ANOGAN(input_dim, dcgan.g)#輸入雜訊維度z 輸入生成器模型 
        self.anogan.compile(anogan_optim) #設定優化器 Loss值

    
    def detection(self,img,iterations):
        
        h,w,c = img.shape 
        
        img = img.reshape((h, w, c)) #重新調整維度大小
        img = img.astype('float32') / 255 #數值正規化0到1之間
        test_img = img[np.newaxis,:,:,:] #由於只有一筆資料 將(64,64,3)變成(1,64,64,3)維度

        anomaly_score, gen_img = self.anogan.compute_anomaly_score(test_img, iterations)#計算異常分數 跟 生成影像
        
        
        test_img = denormalize(test_img[0])  #數值反正規化將0~1的資料轉成0~255之間
        gen_img = denormalize(gen_img[0])#數值反正規化將0~1的資料轉成0~255之間
        

        
        return anomaly_score , test_img , gen_img
    
def show_image(test_img,gen_img):
    

    residual_img , threshold_img = difference(test_img,gen_img,50)#測試圖片 #生成圖片 #門檻值
    
    '''
    #OpenCV show image
    #h,w,c = test_img.shape 
    #test_img = cv2.resize(test_img,(h*4,w*4))
    #gen_img = cv2.resize(gen_img,(h*4,w*4))
    imgs = np.concatenate((test_img , gen_img,residual_img , threshold_img), axis=1) #合併圖片
    cv2.imshow('Windows', imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    imgs = [test_img , gen_img , residual_img , threshold_img]
    title = ["Test","Generated","Residual","Threshold"]
    fig = plt.figure(figsize=(12, 6))
    columns = 5
    rows = 1
    for i in range(rows,columns):
    
        fig.add_subplot(rows,columns, i)
        plt.title(title[i-1])
        plt.imshow(cv2.cvtColor(imgs[i-1], cv2.COLOR_BGR2RGB))
        
    plt.show()
    
def show_result(f,anomaly_score,tStart,tEnd):

    print("\n")
    print("File name : "+str(f))
    print("Anomaly Score : "+str(anomaly_score))
    print("Cost time : "+str(tEnd - tStart))
    
def write_text(f,label,anomaly_score):

    fp = open("anomaly_score.txt", "a")
    fp.write(f+" "+label+" "+str(anomaly_score)+"\n")
    fp.close()
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result', dest='result', default=0,
                        help='result : 1 show image result')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':

        w = 64
        h = 64
        c = 3 
        
        args = get_args()#外部輸入參數
        show_image_reuslt = args.result
        
        image_shape = ((h,w,c))
        input_dim = 200
        iterations = 150 #迭代次數
        
        
        AnoGAN = anogan_model("./model/",input_dim,image_shape)
        
        
        #show summury
        AnoGAN.anogan.model.summary()
        
        ########################abnormal###########################
        label = "abnormal" 
        file_path = "./dataset/test/"+label+"/"
        files = listdir(file_path)
        
        for f in files:
            
            img = cv2.imread(file_path+f)
            img = cv2.resize(img, (w, h))
            
            tStart = time.time()#計時開始

            anomaly_score , test_img , gen_img = AnoGAN.detection(img,iterations)

            tEnd = time.time()#計時結束
            
            show_result(f,anomaly_score,tStart,tEnd)
            
            if show_image_reuslt == "1":
                show_image(test_img,gen_img)
            
            write_text(f,label,anomaly_score)
        
        ########################normal###########################
        label = "normal" 
        file_path = "./dataset/test/"+label+"/"
        files = listdir(file_path)
        
        for f in files:
            
            img = cv2.imread(file_path+f)
            img = cv2.resize(img, (w, h))
            
            tStart = time.time()#計時開始

            anomaly_score , test_img , gen_img = AnoGAN.detection(img,iterations)

            tEnd = time.time()#計時結束
            
            show_result(f,anomaly_score,tStart,tEnd)
            
            if show_image_reuslt == "1":
                show_image(test_img,gen_img)
                
            write_text(f,label,anomaly_score)
        
