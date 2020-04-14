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
1.將測試圖片放進./dataset/test/abnormal/資料夾當中  
2.測試圖片請執行 
```sh
 python test.py    
```
## 安裝套件 
```sh
 pip install -r requirements.txt  
```

## 需求套件及環境:

python == 3.6 (或以上)
numpy == 1.17.2 (不限定此版本)  
opencv-python == 4.1.1.26 (不限定此版本)  
tensorflow-gpu == 1.9.0  
keras == 2.2.5  
