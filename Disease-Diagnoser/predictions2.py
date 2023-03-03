import torch
import torch.nn as nn

import numpy as np
import cv2

class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()

    self.cnn1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
    self.batchnorm1=nn.BatchNorm2d(32)
    self.relu=nn.ReLU()
    self.max=nn.MaxPool2d(kernel_size=2)
    self.cnn2=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
    self.batchnorm2=nn.BatchNorm2d(32)
    self.cnn3=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)
    self.batchnorm3=nn.BatchNorm2d(32)
    self.cnn4=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)
    self.batchnorm4=nn.BatchNorm2d(32)
    self.cnn5=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=7,stride=1,padding=3)
    self.batchnorm5=nn.BatchNorm2d(32)
    self.cnn6=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=7,stride=1,padding=3)
    self.batchnorm6=nn.BatchNorm2d(32)
    self.fc1=nn.Linear(512,250)
    self.drop=nn.Dropout(p=0.5)
    self.fc2=nn.Linear(250,2)

  def forward(self,x):

    out=self.cnn1(x)
    out=self.batchnorm1(out)
    out=self.relu(out)
    out=self.max(out)#128

    out=self.cnn2(out)
    out=self.batchnorm2(out)
    out=self.relu(out)
    out=self.max(out)#64

    out=self.cnn3(out)
    out=self.batchnorm3(out)
    out=self.relu(out)
    out=self.max(out)#32

    out=self.cnn4(out)
    out=self.batchnorm4(out)
    out=self.relu(out)
    out=self.max(out)#16

    out=self.cnn5(out)
    out=self.batchnorm5(out)
    out=self.relu(out)
    out=self.max(out)#8

    out=self.cnn6(out)
    out=self.batchnorm6(out)
    out=self.relu(out)
    out=self.max(out)#4

    out=out.view(-1,512)

    out=self.fc1(out)
    out=self.relu(out)
    #out=self.max(out)
    out=self.drop(out)
    out=self.fc2(out)

    return out

def prediction(path):
    x = []
    y = []
    class_names=["Normal","Heart Stroke"]

    trained_model = CNN()
    trained_model1 = CNN()
    trained_model2 = CNN()
    trained_model3 = CNN()
    trained_model4 = CNN()
    trained_model5 = CNN()

    trained_model.load_state_dict(torch.load('model/heartdisease/heart disease full.pth'), strict=False)
    trained_model1.load_state_dict(torch.load('model/heartdisease/heart disease1.pth'), strict=False)
    trained_model2.load_state_dict(torch.load('model/heartdisease/heart disease2.pth'), strict=False)
    trained_model3.load_state_dict(torch.load('model/heartdisease/heart disease3.pth'), strict=False)
    trained_model4.load_state_dict(torch.load('model/heartdisease/heart disease4.pth'), strict=False)
    trained_model5.load_state_dict(torch.load('model/heartdisease/heart disease5.pth'), strict=False)

    trained_model.eval()
    trained_model1.eval()
    trained_model2.eval()
    trained_model3.eval()
    trained_model4.eval()
    trained_model5.eval()

    a = []

    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.array(image, dtype=np.float32)
    image = torch.from_numpy(image)

    c = trained_model(image)
    c1 = trained_model1(image)
    c2 = trained_model2(image)
    c3 = trained_model3(image)
    c4 = trained_model4(image)
    c5 = trained_model5(image)

    a1 = torch.sigmoid(c1)
    a.append(a1[0].tolist())
    a2 = torch.sigmoid(c2)
    a.append(a2[0].tolist())
    a3 = torch.sigmoid(c3)
    a.append(a3[0].tolist())
    a4 = torch.sigmoid(c4)
    a.append(a4[0].tolist())
    a5 = torch.sigmoid(c5)
    a.append(a5[0].tolist())
    af = torch.sigmoid(c)
    a.append(af[0].tolist())

    perd = []
    perd.append(a[0].index(max(a[0])))
    perd.append(a[1].index(max(a[1])))
    perd.append(a[2].index(max(a[2])))
    perd.append(a[3].index(max(a[3])))
    perd.append(a[4].index(max(a[4])))
    perd.append(a[5].index(max(a[5])))

    def county(y):
        result = (y[0] * 0.15) + (y[1] * 0.15) + (y[2] * 0.15) + (y[3] * 0.15) + (y[4] * 0.15) + (y[5] * 0.25)
        if result > 0.5:
            result = 1
        else:
            result = 0
        return result
    return class_names[county(perd)]
