 # AnoGAN-Keras

paper : https://arxiv.org/abs/1703.05921  

## 新建空白資料夾

```sh
python create_folder.py  
```

## 訓練階段

1.將訓練圖片資料放進./dataset/train/normal 資料夾當中  
2.訓練AI網路模型請執行
```sh
 python train.py  
```
3.訓練過程AI所生成的圖片會放進./result資料夾中  
4.訓練完成後會將模型放進./model資料夾中

## 測試階段
將測試圖片放進./dataset/test/abnormal/及./dataset/test/normal/資料夾當中  

只顯示結果
```sh
 python test.py    
```
顯示結果及圖片  
```sh
 python test.py --result=1
```
顯示異常分數直方圖
(請先確認anomaly_score.txt是否存在若不存在請先執行test.py) 
```sh
 python histogram.py
```
 


## 安裝套件 
```sh
 pip install -r requirements.txt  
```

## 需求套件及環境

python == 3.6 (版本3.0以上)  
numpy == 1.17.2 (不限定此版本)  
opencv-python == 4.1.1.26 (不限定此版本)  
tensorflow-gpu == 1.9.0  
keras == 2.2.5   
argparse  
matplotlib   
seaborn  

## 異常分數直方圖 

![image](https://github.com/crystal02146/image/blob/master/AnoGAN_histogram.png)

## 生成圖片及殘差圖片   

![image](https://github.com/crystal02146/image/blob/master/AnoGAN_result_normal1.PNG)
![image](https://github.com/crystal02146/image/blob/master/AnoGAN_result_normal2.PNG)
![image](https://github.com/crystal02146/image/blob/master/AnoGAN_result_abnormal1.PNG)
