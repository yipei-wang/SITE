import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size = 3,
                               stride = stride,
                               padding = 1,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, 
                               planes, 
                               kernel_size = 3, 
                               stride = 1,
                               padding = 1,
                               bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          planes, 
                          kernel_size = 1,
                          stride = stride,
                          bias = False),
                nn.BatchNorm2d(planes))
            
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, 
                 BasicBlock, 
                 num_classes = 10, 
                 num_Blocks = [2,2,2,2]):
        
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(nn.Conv2d(3, 
                                             self.inchannel, 
                                             kernel_size = 3,
                                             stride = 1,
                                             padding = 1,
                                             bias = False),
                                   nn.BatchNorm2d(self.inchannel),
                                   nn.ReLU())
        self.layer1 = self.make_layer(BasicBlock, 
                                      2*self.inchannel, 
                                      num_Blocks[0], 
                                      stride = 1)
        self.layer2 = self.make_layer(BasicBlock, 
                                      2*self.inchannel, 
                                      num_Blocks[1], 
                                      stride = 2)
        self.layer3 = self.make_layer(BasicBlock, 
                                      2*self.inchannel, 
                                      num_Blocks[2], 
                                      stride = 2)
        self.layer4 = self.make_layer(BasicBlock, 
                                      2*self.inchannel, 
                                      num_Blocks[3], 
                                      stride = 2)
        self.conv2 = nn.Conv2d(256, 10, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(10)
        self.fc = nn.Linear(10, 10)
        
    def make_layer(self, BasicBlock, channels, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.inchannel,
                                        channels,
                                        stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn1(self.conv2(out))
        feature = torch.tanh(out)
        out = F.avg_pool2d(feature, 16)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return feature, out
    
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 10, 3, 2, 1, output_padding = 1),
            nn.BatchNorm2d(10),
            nn.Tanh(),
            nn.Flatten(),
        )
        
    def forward(self, x):
        return self.main(x)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(10, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.LeakyReLU()
        )
        
        self.generator = nn.ModuleList([Generator().to(device) for i in range(10)])
        
    def forward(self, x):
        feature = self.extractor(x)
        W = self.generator[0](feature).view(feature.shape[0], 1, 2560)
        Y = torch.matmul(W, x.view(feature.shape[0], 2560, 1))
        Y = Y.view(feature.shape[0], 1)
        
        for i in range(1, 10):
            w = self.generator[i](feature).view(feature.shape[0], 1, 2560)
            y = torch.matmul(w, x.view(feature.shape[0], 2560, 1))
            y = y.view(feature.shape[0], 1)
            
            W = torch.cat([W, w], dim = 1)
            Y = torch.cat([Y, y], dim = 1)

        W = W.view(feature.shape[0], 10, 10, 16, 16)
        
        return W, Y
    

class SITE(nn.Module):
    def __init__(self):
        super(SITE, self).__init__()
        self.backbone = ResNet(BasicBlock).to(device)
        self.site = Classifier().to(device)
        
    def forward(self, x):
        feature, _ = self.backbone(x)
        _, pred = self.site(feature)
        return pred
    
    def get_explanation(self, x, target_class):
        feature, _ = self.backbone(x)
        W, _ = self.site(feature)
        explanation = (feature.unsqueeze(1)*W).mean(2).squeeze()[target_class]
        explanation = F.interpolate(explanation[None, None], size = (128, 128), mode = 'bilinear', align_corners = True).squeeze()
        return explanation
    

class MNIST_Generator(nn.Module):
    def __init__(self):
        super(MNIST_Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 7, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
        )
        
    def forward(self, x):
        return self.main(x)
    
class MNIST_SITE(nn.Module):
    def __init__(self):
        super(MNIST_SITE, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 7, bias = False),
            nn.ReLU(),
            nn.Flatten()
        )
        

        self.generator = nn.ModuleList([MNIST_Generator().to(device) for i in range(10)])
        
    def forward(self, x):
        batch_size = x.shape[0]
        feature = self.extractor(x).view(-1, 64, 1, 1)
        y = torch.zeros(batch_size, 10).to(x.device)
        W = torch.zeros(batch_size, 10, 784).to(x.device)
        for i in range(10):
            W[:,i] = self.generator[i](feature)
            y[:,i] = torch.matmul(x.view(batch_size, 1, 784), 
                                  W[:,i].view(batch_size, 784, 1)).flatten()
        W = W.view(batch_size, 10, 28, 28)
        return y    
    
    def get_explanation(self, x):
        batch_size = x.shape[0]
        feature = self.extractor(x).view(-1, 64, 1, 1)
        y = torch.zeros(batch_size, 10).to(x.device)
        W = torch.zeros(batch_size, 10, 784).to(x.device)
        for i in range(10):
            W[:,i] = self.generator[i](feature)
            y[:,i] = torch.matmul(x.view(batch_size, 1, 784), 
                                  W[:,i].view(batch_size, 784, 1)).flatten()
        W = W.view(batch_size, 10, 28, 28)
        return W, y


if __name__ == '__main__':
    model = ResNet(BasicBlock).to(device)
    G = Classifier().to(device)
    site = SITE().to(device)
    site = MNIST_SITE().to(device)
    print('Models are built!')





