import torch
from torch import nn
from torchvision import models

class TimeWarp(nn.Module):
    def __init__(self, baseModel, method='squeeze'):
        super(TimeWarp, self).__init__()
        self.baseModel = baseModel
        self.method = method
 
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        if self.method == 'loop':
            output = []
            for i in range(time_steps):
                x_t = self.baseModel(x[:, i, :, :, :])
                x_t = x_t.view(x_t.size(0), -1)
                output.append(x_t)
            x = torch.stack(output, dim=0).transpose_(0, 1)
            output = None
            x_t = None
        else:
            x = x.contiguous().view(batch_size * time_steps, C, H, W)
            x = self.baseModel(x)
            x = x.view(x.size(0), -1)
            x = x.contiguous().view(batch_size , time_steps , x.size(-1))
        return x
    
class extractlastcell(nn.Module):
    def forward(self,x):
        out , _ = x
        return out[:, -1, :]
    
def create_model(num_features, rnn_hidden_size, rnn_num_layers, dr_rate, num_classes):
    ResNet50 = models.resnet50(pretrained=True)
    custom_RN50 = nn.Sequential()
    ct = 0
    for child in ResNet50.children():
        if ct < 5:
            custom_RN50.append(child)
            for param in child.parameters():
                param.requires_grad = False
        elif ct == 5:
            custom_RN50.append(child)
            for param in child.parameters():
                param.requires_grad = True
        else:
            break
        ct += 1
    custom_RN50.append(nn.AdaptiveAvgPool2d(output_size=(5, 5)))

    return nn.Sequential(
        TimeWarp(custom_RN50),
        nn.Dropout(dr_rate),
        nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True),
        extractlastcell(),
        nn.Linear(30, 256),
        nn.ReLU(),
        nn.Dropout(dr_rate),
        nn.Linear(256, num_classes)
    )