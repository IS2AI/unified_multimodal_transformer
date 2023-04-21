from torchvision import models
import torch.nn as nn
import torch

from speaker_verification.models_handmade.resnet import ResNet34
import timm

class SelfAttentivePool2d(nn.Module):
    def __init__(self, input_dim=512):
        super(SelfAttentivePool2d, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def preprocess(self,x):
        """
            [batch size, n_mels, n_frames, hidden dim] --> [batch size, sequence length, hidden dim]
            [1, 3, 24, 512] --> [1, 72, 512]
        """
        return torch.flatten(x.permute(0, 2, 3, 1), start_dim=1, end_dim=2)
        
        
    def forward(self, x):
        """
        output = Softmax(W @ H) * H
        input:
            x : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        H = self.preprocess(x)

        attention_scores = self.softmax(self.W(H))
        utter_rep = torch.sum(attention_scores * H, dim=1)

        return utter_rep

# RESNET 34 pretrained
class ResNet(nn.Module):
    """
        Parameters
        ----------
        pretrained_weights : bool, default = "True"
            Ways of weights initialization. 
            If "False", it means random initialization and no pretrained weights,
            If "True" it means resnet34 pretrained weights are used.

        fine_tune: bool, default = "False"
            Allows to choose between two types of transfer learning: fine tuning and feature extraction.
            For more details of the description of each mode, 
            read https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

        embedding_size: int, default = 128
            Size of the embedding of the last layer
            
    """
    def __init__(self, pretrained_weights=True, fine_tune=False, embedding_size=128, modality = "rgb", filter_size="default", from_torch=True):
        super(ResNet, self).__init__()

        if from_torch:
            self.pretrained_weights = pretrained_weights
            self.fine_tune = fine_tune
            
            if self.pretrained_weights:
                pretrained_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                pretrained_model = models.resnet34(weights=None)
            
            if not self.fine_tune:
                for param in pretrained_model.parameters():
                    param.requires_grad = False

            last_layer_hidden_size = pretrained_model.fc.in_features
            
            if modality == "wav":
                pretrained_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                pretrained_model.avgpool = SelfAttentivePool2d(last_layer_hidden_size)

            pretrained_model.fc = nn.Linear(last_layer_hidden_size, embedding_size)

            self.model = pretrained_model
        else:
            if filter_size == "default":
                hid_dim = 64
            elif filter_size == "half":
                hid_dim = int(64/2)
            elif filter_size == "quarter":
                hid_dim = int(64/4)
            
            if modality == "wav":
                input_dim = 1
            else:
                input_dim = 3
            
            self.model = ResNet34(input_dim=input_dim, hid_dim = hid_dim, output_dim = embedding_size)


    def forward(self, x):
        x = self.model(x)
        return x


class Model(nn.Module):
    def __init__(self, 
                library="pytorch", 
                pretrained_weights=True, 
                fine_tune=False, 
                embedding_size=128, 
                modality = "rgb",
                model_name = "resnet34",
                pool="default"):

        super(Model, self).__init__()

        if modality == "wav":
            in_channels = 1
        else:
            in_channels = 3
    
        if library == "pytorch":
            if model_name == "resnet34":
                if pretrained_weights:
                    weights = models.ResNet34_Weights.DEFAULT
                else:
                    weights = None

                self.model = models.resnet34(weights=weights)
                if modality == "wav":
                    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)
                
                if pool == "SAP":
                    self.model.avgpool = SelfAttentivePool2d(self.model.fc.in_features)

            if fine_tune:
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.parameters():
                    param.requires_grad = False
                
                self.model.fc.weight.requires_grad = True
                self.model.fc.bias.requires_grad = True

        elif library == "timm":
            self.model = timm.create_model(model_name, pretrained=pretrained_weights, num_classes=embedding_size, in_chans=in_channels)

            if pool == "SAP":
                self.model.global_pool = SelfAttentivePool2d()
            if fine_tune:
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.parameters():
                    param.requires_grad = False

                self.model.get_classifier().weight.requires_grad = True
                self.model.get_classifier().bias.requires_grad = True
            


    def forward(self, x):
        x = self.model(x)
        return x

