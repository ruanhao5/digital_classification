# digital_classification

This is a repository for my homework.

Reference contest: [零基础入门CV - 街景字符编码识别](https://tianchi.aliyun.com/competition/entrance/531795/introduction)

## Prepare dataset

All download link of dataset in csv file: [mchar_data_list_0515.csv](https://aliyuntianchiresult.cn-hangzhou.oss.aliyun-inc.com/file/race/documents/531795/mchar_data_list_0515.csv?Expires=1625234966&OSSAccessKeyId=LTAI5tJYjgpnqJHcXFPFwvSi&Signature=1FGyVY8E5HDVC4s0z2KGQP3vWOk%3D&response-content-disposition=attachment%3B%20)

Download data and organise as follows:

```
# For SVHN Dataset
digital_classification
└─ input
   ├─ mchar_test_a
   ├─ mchar_train
   ├─ mchar_val
   ├─ mchar_sample_submit_A.csv
   ├─ mchar_train.json
   └─ mchar_val.json
```

## Basic installation

```bash
cd code
pip install -r requirements.txt 
```

## Train & Evaluate

Just open jupyter notebook `main.ipynb` and run it.