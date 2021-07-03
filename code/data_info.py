import glob
import json
import pandas as pd


# 统计train, val, test数据集的个数
def data_summary():
    train_list = glob.glob('../input/mchar_train/*.png')
    val_list = glob.glob('../input/mchar_val/*.png')
    test_list = glob.glob('../input/mchar_test_a/*.png')
    print('train image counts: %d' % len(train_list))
    print('val image counts: %d' % len(val_list))
    print('test image counts: %d' % len(test_list))

    print("-------------------------------------------")


# 统计图片分别含有数字的个数
def label_summary():
    marks = json.load(open('../input/mchar_train.json', 'r'))

    dicts = {}
    for img, mark in marks.items():
        if len(mark['label']) not in dicts:
            dicts[len(mark['label'])] = 0
        dicts[len(mark['label'])] += 1

    dicts = sorted(dicts.items(), key=lambda x: x[0])
    for k, v in dicts:
        print('%d个数字的图片数目: %d' % (k, v))

    print("-------------------------------------------")


# 看train数据集第一张的信息，长宽高等
def first_photo_info():
    train_json = json.load(open('../input/mchar_train.json'))
    print("First train photo info:")
    print(train_json['000000.png'])
    print("-------------------------------------------")


# 看需要输出文件的信息
def look_submit():
    df = pd.read_csv('../input/mchar_sample_submit_A.csv', sep=',')
    print(df.head(5))

    print("-------------------------------------------")


if __name__ == "__main__":
    data_summary()
    label_summary()
    first_photo_info()
    look_submit()
