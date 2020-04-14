# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:22:44 2020

@author: Tseng Yi-Yao
"""


#引入模組
import os

def mkdir(path):
    #判斷目錄是否存在
    #存在：True
    #不存在：False
    folder = os.path.exists(path)

    #判斷結果
    if not folder:
        #如果不存在，則建立新目錄
        os.makedirs(path)
        print(path+'建立成功')

    else:
        #如果目錄已存在，則不建立，提示目錄已存在
        print(path+'目錄已存在')
 
mkdir("./dataset/train/normal/")
mkdir("./dataset/train/abnormal/")
mkdir("./dataset/test/normal/")
mkdir("./dataset/test/abnormal/")
mkdir("./model/")
mkdir("./result/")