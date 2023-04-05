import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(ConvBlock,self).__init__()

        k = kernel
        s = stride

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=(k, k), stride=(s, s), 
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(output_dim)
        )

    def forward(self, x):
      return self.conv_block(x)

class DownSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownSample,self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1), stride=(2, 2), 
                      bias=False),
            nn.BatchNorm2d(output_dim)
        )
    
    def forward(self, x):
      return self.downsample(x)

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, skip_connection=False):
        super(BasicBlock,self).__init__()

        self.skip_connection = skip_connection
        
        self.conv1 = ConvBlock(input_dim, output_dim, kernel=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        if self.skip_connection:
            self.conv2 = ConvBlock(output_dim, output_dim, kernel=3, stride=2)
        else:
            self.conv2 = ConvBlock(output_dim, output_dim, kernel=3, stride=1)

        if self.skip_connection:
            self.downsample = DownSample(input_dim, output_dim)

    def forward(self, x):
      identity = x

      out = self.conv1(x)
      out = self.relu(out)
      out = self.conv2(out)

      if self.skip_connection:
          identity = self.downsample(x)
      
      out += identity
      out = self.relu(out)
      return out

class ResNet34(nn.Module):
    def __init__(self,input_dim=3, hid_dim = 64, output_dim = 128):
        super(ResNet34,self).__init__()

        self.conv_block = nn.Sequential(
            ConvBlock(input_dim, hid_dim, kernel=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, 
                         ceil_mode=False)
        )

        self.layer1 = nn.Sequential(
            BasicBlock(hid_dim,hid_dim),
            BasicBlock(hid_dim,hid_dim),
            BasicBlock(hid_dim,hid_dim)
        )
       
        self.layer2 = nn.Sequential(
            BasicBlock(hid_dim,hid_dim*2,skip_connection=True), # (64,128)
            BasicBlock(hid_dim*2,hid_dim*2),
            BasicBlock(hid_dim*2,hid_dim*2),
            BasicBlock(hid_dim*2,hid_dim*2)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(hid_dim*2,hid_dim*4,skip_connection=True),   # (128,256)
            BasicBlock(hid_dim*4,hid_dim*4),
            BasicBlock(hid_dim*4,hid_dim*4),
            BasicBlock(hid_dim*4,hid_dim*4),
            BasicBlock(hid_dim*4,hid_dim*4),
            BasicBlock(hid_dim*4,hid_dim*4),
        )

        self.layer4 = nn.Sequential(
            BasicBlock(hid_dim*4,hid_dim*8,skip_connection=True), # (256,512)
            BasicBlock(hid_dim*8,hid_dim*8),
            BasicBlock(hid_dim*8,hid_dim*8)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=hid_dim*8, out_features=output_dim, bias=True)
        
    def forward(self,x):

        x = self.conv_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, 1)
        x = self.fc(x)

        return x