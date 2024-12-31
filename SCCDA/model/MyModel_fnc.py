import torch
import torch.nn as nn
import torch.nn.functional as F
# from grad_reverse import grad_reverse
from torchvision import models
from torchsummary import summary
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features=nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),
                    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),
                    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),
                    nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)),
                    nn.Conv2d(384, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
                )
        self.classifier = nn.Sequential(
               # nn.Dropout(),
               # nn.Linear(512*7*7, 2048),
               # nn.ReLU(inplace=True),
               # nn.Dropout(),
               # nn.Linear(2048, 1024),
               # nn.ReLU(inplace=True),
               # nn.Linear(1024, 1000))

               #nn.Dropout(),
               nn.ReLU(inplace=True),
               nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
               nn.BatchNorm2d(512),
               nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1)),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True),
               nn.Conv2d(512, 1024, kernel_size=(2, 2), stride=(1, 1)),
               nn.BatchNorm2d(1024),
               nn.ReLU(inplace=True),
               nn.Conv2d(1024, 1024, kernel_size=(2, 2), stride=(1, 1)),
               nn.BatchNorm2d(1024),
               nn.ReLU(inplace=True),
               nn.Conv2d(1024, 2, kernel_size=(1, 1), stride=(1, 1))
                )
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
# #        print(x)
# #        print(x.size())
#         x = self.classifier(x)
#         return x

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        load_capsule = MyModel()
        #load_capsule.load_state_dict(torch.load('G.pkl'))
        self.Feature = load_capsule.features

    def forward(self, x):
        x = self.Feature(x)
       # in_fc = x.view(x.size(0), -1)
        return x
        #        print(x)
        #        print(x.size())

# class Predictor(nn.Module):
#     def __init__(self):
#         super(Predictor, self).__init__()
#         self.fc1 = nn.Linear(512 * 7 * 7, 2048)
#         self.fc2 = nn.Linear(2048, 1024)
#         self.fc3 = nn.Linear(1024, 1000)
#         self.fc4 = nn.Linear(1000, 2)
#
#
#     def forward(self, x):
#         # if reverse:
#         #     x = grad_reverse(x, self.lambd)
#         x = F.dropout(x, training=self.training, p=0.5)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training, p=0.5)
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x, training=self.training, p=0.5)
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        load_capsule = MyModel()
        #load_capsule.load_state_dict(torch.load('C.pkl'))
        self.predictor = load_capsule.classifier
       # self.fc4 = nn.Linear(1000, 2)

    def forward(self, x):
        out = self.predictor(x)
       # x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.5)
       # out = self.fc4(x)
        return out

if __name__ == '__main__':
    # my_model = nn.Sequential(   Feature(),    Predictor()).cuda()
    my_model = Feature().cuda()
    summary(my_model, (3, 224, 224))