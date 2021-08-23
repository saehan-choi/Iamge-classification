import torch
import torch.nn as nn
from torchsummary import summary
import cv2
import numpy as np
import os

from google.colab import drive
from google.colab.patches import cv2_imshow
drive.mount('/content/drive')

class myconv(nn.Module):
    def __init__(self, init_weights: bool = True):
        super(myconv, self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3x64 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv64x64 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
        )
        self.conv64x128 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
        )
        self.conv128x128 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
        )
        self.conv128x256 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
        )
        self.conv256x256 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),            
        )        
        self.conv256x512 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
        )
        self.conv512x512 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),            
        )
        self.convnet = nn.Sequential(
            self.conv3x64,
            self.conv64x128,
            self.maxpool, # 112 -> 56  
            self.conv128x128,
            self.conv128x256,
            self.maxpool, # 56 -> 28
            self.conv256x256,
            self.conv256x512,
            self.maxpool, # 28 -> 14
            self.conv512x512,
            self.conv512x512,
            self.maxpool # 14 -> 7
        )


        self.fclayer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

    def forward(self, x:torch.Tensor):
        x = self.convnet(x)
        x = torch.flatten(x, 1)
        x = self.fclayer(x)
        return x

device = torch.device('cuda')
convnet = myconv()
convnet = convnet.to(device)
summary(convnet, (3, 224, 224))

PATH="/content/drive/MyDrive/aiplatform_project/"
convnet.load_state_dict(torch.load(PATH+'model.pt'))
testsetPATH = "/content/drive/MyDrive/aiplatform_project/test/"
classes =  ('cat','dog')

print(f'테스트 셋 개수 : {len(os.listdir(testsetPATH))}장')

for i in range(1,len(os.listdir(testsetPATH))+1):
  img = cv2.imread(f'{testsetPATH}{i}.jpg')
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  dst_ = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_AREA)
  dst = np.transpose(dst_, (2, 0, 1))
  tensor = torch.Tensor(dst)
  tensor = tensor.unsqueeze(0)
  images = tensor.cuda()
  outputs = convnet(images)
  _, predicted = torch.max(outputs, 1)
  text = classes[predicted.item()]

  img = cv2.cvtColor(dst_,cv2.COLOR_RGB2BGR)
  cv2_imshow(img)
  print(f'{text}입니다.')
  print("")
  print("")

for epoch in range(20):  # loop over the dataset multiple times
    train_loss = 0.0
    train_correct = 0
    val_loss = 0.0
    val_correct = 0
    True_positive, True_Negative, False_positive, False_Negative = [0 for _ in range(4)]
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs= vgg11(inputs)

        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (predicted == labels).cpu().sum()
        
        for i in range(batch_size_):
            label = labels[i]
            classes_ = classes[label]
            c = (predicted == labels).squeeze()

            if classes_ == "Dog":
                if classes_ == classes[c[i].item()]:
                    True_positive += 1
                else:
                  
                    True_Negative += 1

            elif classes_ != "Dog":
                if classes_ == classes[c[i].item()]:
                    False_positive += 1
                else:
                    False_Negative += 1