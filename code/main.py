import glob, json

from PIL import Image
from PIL import ImageChops
from PIL import ImageFilter
import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data.dataset import Dataset
from tensorboardX import SummaryWriter   # tensorboard
import torchvision.transforms as transforms


print(torch.__version__)
print(torch.cuda.is_available())

train_path = glob.glob('../input/mchar_train/*.png')
train_path.sort()
train_json = json.load(open('../input/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]

val_path = glob.glob('../input/mchar_val/*.png')
val_path.sort()
val_json = json.load(open('../input/mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json]

# BATCH_SIZE = 10
BATCH_SIZE = 1000

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None, train=True, test=False):
        self.img_path = img_path
        self.img_label = img_label
        self.train = train
        self.test = test
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        
        if self.train and not self.test:
            img = img.resize((128, 64))
            img = ImageChops.offset(img, np.random.randint(-35, 35), np.random.randint(-10, 10))  # 平移
            if np.random.randint(2):
                # img = np.array(img)
                # img = util.random_noise(img, mode='gaussian')  # 添加噪声
                # img = np.uint8(img*255);
                # img = Image.fromarray(img)
                img = img.filter(ImageFilter.BLUR)  # 图像模糊

        if self.transform is not None:
            img = self.transform(img)
        
        if self.img_label == None:
            return img
        else:
            # 原始SVHN中类别10为填充的数字X, 最多字符为6
            lbl = np.array(self.img_label[index], dtype=np.int)
            lbl = list(lbl)  + (4 - len(lbl)) * [10]
            return img, torch.from_numpy(np.array(lbl[:4]))

    def __len__(self):
        return len(self.img_path)


train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                   transforms.Compose([
                       transforms.Resize((64, 128)),
                       transforms.RandomCrop((60, 120)),
                       transforms.ColorJitter(0.3, 0.3, 0.2),
                       transforms.RandomRotation(10),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),train=True, test=False), 
    batch_size=BATCH_SIZE, # 每批样本个数
    shuffle=True, # 是否打乱顺序
    num_workers=10, # 读取的线程个数
)

    
val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                    transforms.Compose([
#                        transforms.Resize((64, 128)),
                        transforms.Resize((60, 120)),
                    #    transforms.ColorJitter(0.3, 0.3, 0.2),
                    #    transforms.RandomRotation(5),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]), train=False, test=True), 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=10, 
)

# 定义模型
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
         # CNN，提取特征
        model_conv = models.resnet34(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)  # 更改了resnet中的avgpool层
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])  # children获得网络的子层，list封装一下，[:-1]遍历全部元素
        self.cnn = model_conv

        # 全连接网络，分类
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
#         self.fc5 = nn.Linear(512, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
#         c5 = self.fc5(feat)
        return c1, c2, c3, c4
    
def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    for (data, label) in tqdm(train_loader):
        data = data.cuda()
        label = label.cuda()
        c0, c1, c2, c3 = model(data)
#         sample weights
#         c0, c1, c2, c3, c4 = model(data)
#         loss = 2*criterion(c0, label[:, 0]) + \
#                 criterion(c1, label[:, 1]) + \
#                 2*criterion(c2, label[:, 2]) + \
#                 criterion(c3, label[:, 3])
        
        loss = criterion(c0, label[:, 0]) + \
                criterion(c1, label[:, 1]) + \
                criterion(c2, label[:, 2]) + \
                criterion(c3, label[:, 3]) 
        
        loss /= (4*BATCH_SIZE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    return np.mean(train_loss)


def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []
    val_predict = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for (data, label) in tqdm(val_loader):
            data = data.cuda()
            label = label.cuda()
            c0, c1, c2, c3 = model(data)

            loss = criterion(c0, label[:, 0]) + \
                    criterion(c1, label[:, 1]) + \
                    criterion(c2, label[:, 2]) + \
                    criterion(c3, label[:, 3]) 
            loss /= (4*BATCH_SIZE)
            val_loss.append(loss.item())
            
            # calculate validation accuracy
            output = np.concatenate([
                c0.data.cpu().numpy(),
                c1.data.cpu().numpy(),
                c2.data.cpu().numpy(),
                c3.data.cpu().numpy()], axis=1)  # output: shape(batch, 44)
            val_predict.append(output)        
        val_predict = np.vstack(val_predict)
        
        val_predict = np.vstack([
            val_predict[:, :11].argmax(1),
            val_predict[:, 11:22].argmax(1),
            val_predict[:, 22:33].argmax(1),
            val_predict[:, 33:44].argmax(1)]).T
        val_predict_label = list()
        for x in val_predict:
            val_predict_label.append(''.join(map(str, x[x!=10])))
        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_acc = np.mean(np.array(val_predict_label) == np.array(val_label))
    return np.mean(val_loss), val_acc

# load model
# model.load_state_dict(torch.load('model_end.pt'))

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

model = SVHN_Model1().cuda()
# 损失函数
criterion = nn.CrossEntropyLoss(size_average=False)
# 优化器
optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=0.0005)
best_loss = 1000.0


train_loss_plot = list()
val_loss_plot = list()
val_acc_plot = list()

# writer = SummaryWriter('log')
writer = SummaryWriter()

from datetime import datetime
now = datetime.now()

for epoch in range(40):
    print('\nEpoch: ', epoch)
    if epoch == 15: optimizer = torch.optim.Adam(model.parameters(), 0.0001, weight_decay=0.0005)
    if epoch == 25: optimizer = torch.optim.Adam(model.parameters(), 0.00001, weight_decay=0.0005)
#     if epoch == 35: optimizer = torch.optim.Adam(model.parameters(), 0.00001, weight_decay=0.0005)

    train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_loss, val_acc = validate(val_loader, model, criterion)# val_predict: shape(10000, 44)

    
    # 记录下验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './model_best.pt')
    
    train_loss_plot.append(train_loss)
    val_loss_plot.append(val_loss)
    val_acc_plot.append(val_acc)
    writer.add_scalar('Loss/train', float(train_loss), epoch)
    writer.add_scalar('Loss/validation', float(val_loss), epoch)
    writer.add_scalar('val_acc', float(val_acc), epoch)

writer.close()
torch.save(model.state_dict(), './model_end.pt')

print("\nruning time:", datetime.now() - now)

plt.plot(train_loss_plot)
plt.xlabel("Epoch")
plt.ylabel("train_loss")
plt.show()

plt.plot(val_loss_plot)
plt.xlabel("Epoch")
plt.ylabel("val_loss")
plt.show()

plt.plot(val_acc_plot)
plt.xlabel("Epoch")
plt.ylabel("val_acc")
plt.show()

print("\ntrain_loss:", train_loss_plot, "\n\nmax:", max(train_loss_plot))
print("\nval_loss:", val_loss_plot, "\n\nmax:", max(val_loss_plot))
print("\nval_acc:", val_acc_plot, "\n\nmax:", max(val_acc_plot))


test_path = glob.glob('../input/mchar_test_a/*.png')
test_path.sort()
test_label = None


test_loader = torch.utils.data.DataLoader(
        SVHNDataset(test_path, test_label,
                   transforms.Compose([
                       transforms.Resize((60, 120)),
                    #    transforms.ColorJitter(0.3, 0.3, 0.2),
                    #    transforms.RandomRotation(5),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]), train=False, test=True), 
    batch_size=BATCH_SIZE, # 每批样本个数
    shuffle=False, # 是否打乱顺序
    num_workers=10, # 读取的线程个数
)


def predict(test_loader, model):
    model.eval()

    test_pred = []
    with torch.no_grad():
        for input in tqdm(test_loader):
            input = input.cuda()

            c0, c1, c2, c3 = model(input)
            output = np.concatenate([
                c0.data.cpu().numpy(),
                c1.data.cpu().numpy(),
                c2.data.cpu().numpy(),
                c3.data.cpu().numpy()], axis = 1)
            test_pred.append(output)
        test_pred = np.vstack(test_pred)
        test_pred = np.vstack([
            test_pred[:, :11].argmax(1),
            test_pred[:, 11:22].argmax(1),
            test_pred[:, 22:33].argmax(1),
            test_pred[:, 33:44].argmax(1)]).T
        
        test_predict_label = list()        
        for x in test_pred:
            test_predict_label.append(''.join(map(str, x[x!=10])))
    return test_predict_label


# load model
model.load_state_dict(torch.load('model_best.pt'))
submit = predict(test_loader, model)


df_submit = pd.read_csv('../input/mchar_sample_submit_A.csv')
df_submit['file_code'] = submit
df_submit.to_csv('submit.csv', index=None)
print("save success!")