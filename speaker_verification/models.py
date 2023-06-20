import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import timm
from transformers import AutoModelForAudioClassification
from transformers import WavLMForXVector


class Model(nn.Module):
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

    def __init__(self, 
                library,
                pretrained_weights,
                fine_tune,
                embedding_size,
                model_name,
                pool, 
                data_type):

        super(Model, self).__init__()

        self.library = library
        self.pretrained_weights = pretrained_weights
        self.fine_tune = fine_tune
        self.embedding_size = embedding_size
        self.model_name = model_name
        self.pool = pool
        self.data_type = data_type
        
        if len(data_type) == 1:

            if data_type[0] == "wav":
                print("wav data type")
                self.model = self.wav_model()
            elif data_type[0] == "rgb":
                print("rgb data type")
                self.model = self.image_model()
            elif data_type[0] == "thr":
                print("thr data type")
                self.model = self.image_model()

    def forward(self, x):
        if self.library == "huggingface":
            x = self.model(x).logits
        else:
            x = self.model(x)
        return x

    def image_model(self):

        if self.library == "huggingface":
            print("HuggingFace model is used.")
            pass
        elif self.library == "pytorch":
            print("pytorch model is used.")
            model = self.pytorch_model(in_channels = 3)
        elif self.library == "timm":
            print("timm model is used.")
            model = self.timm_model(in_channels = 3)

        return model
    
    def wav_model(self):

        if self.library == "huggingface":
            print("HuggingFace model is used.")
            model = self.huggingface_model()
        elif self.library == "pytorch":
            print("pytorch model is used.")
            model = self.pytorch_model(in_channels = 1)
        elif self.library == "timm":
            print("timm model is used.")
            model = self.timm_model(in_channels = 3)

        return model
    
    def huggingface_model(self):
        if self.model_name == "WavLM":
            print("WavLM model is used.")
            model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-sv')
            model.classifier = nn.Linear(model.classifier.in_features, self.embedding_size)

            if self.fine_tune:
                for param in model.parameters():
                    param.requires_grad = True
            else:
                for param in model.parameters():
                    param.requires_grad = False
                
                model.classifier.weight.requires_grad = True
                model.classifier.bias.requires_grad = True
        elif self.model_name == "AST":
            print("AST model is used.")
            model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            model.classifier.dense = nn.Linear(model.classifier.dense.in_features, self.embedding_size)

            if self.fine_tune:
                for param in model.parameters():
                    param.requires_grad = True
            else:
                for param in model.parameters():
                    param.requires_grad = False
                
                model.classifier.dense.weight.requires_grad = True
                model.classifier.dense.bias.requires_grad = True

        return model

    def timm_model(self, in_channels):
        model = timm.create_model(self.model_name, pretrained=self.pretrained_weights, num_classes=self.embedding_size, in_chans=in_channels)

        if self.pool == "SAP":
            model.global_pool = SelfAttentivePool2d()
        if self.fine_tune:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = False

            model.get_classifier().weight.requires_grad = True
            model.get_classifier().bias.requires_grad = True

        return model

    def pytorch_model(self, in_channels):
        if self.model_name == "resnet34":
            if self.pretrained_weights:
                weights = torchvision.models.ResNet34_Weights.DEFAULT
            else:
                weights = None

            model = torchvision.models.resnet34(weights=weights)
            
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.fc = nn.Linear(model.fc.in_features, self.embedding_size)
            
            if self.pool == "SAP":
                model.avgpool = SelfAttentivePool2d(model.fc.in_features)

        if self.fine_tune:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = False
            
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True

        return model


class SelfAttentivePool2d_2008(nn.Module):
    '''
    Based on this article: https://arxiv.org/pdf/2008.01077v1.pdf
    '''
    def __init__(self, input_dim=512):
        super(SelfAttentivePool2d_2008, self).__init__()
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


class SelfAttentivePool2d(nn.Module):
    '''
    Based on this article: https://www.isca-speech.org/archive/pdfs/odyssey_2018/cai18_odyssey.pdf
    '''
    def __init__(self, input_dim=512):
        super(SelfAttentivePool2d, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)

        self.u = nn.Parameter(torch.FloatTensor(input_dim, 1))
        nn.init.xavier_normal_(self.u)

    def preprocess(self,x):
        """
            x: [batch_size, n_channels, n_mels, number of frames] --> [batch_size, number of frames, n_channels]

            step 1: [batch_size, n_channels, n_mels, number of frames] --> [batch_size, n_channels, 1, number of frames]
            step 2: [batch_size, n_channels, 1, number of frames] --> [batch_size, number of frames, n_channels, 1]
            step 3: [batch_size, number of frames, n_channels, 1] --> [batch_size, number of frames, n_channels]
        """
        x = torch.mean(x, dim=2, keepdim=True) 
        x = x.permute(0,3,1,2)
        x = x.squeeze(-1)

        return x
        
    def forward(self, x):
        """
        h = tanh(Wx + b)
        w = Softmax(h @ u) * H
        e = sum(w*x)
        input:
            x : [batch_size, n_channels, n_mels, number of frames]
        
        return:
            e: size (batch_size, n_channels)
        """
        x = self.preprocess(x)

        h = torch.tanh(self.W(x))
        w = torch.matmul(h, self.u).squeeze(dim=2) # [batch_size, number of frames, n_channels=1] --> squeeze: [batch_size, number of frames]
        w = F.softmax(w, dim=1)
        w = w.view(x.size(0), x.size(1), 1) # [batch_size, number of frames] --> [batch_size, number of frames, n_channels=1]
        e = torch.sum(x * w, dim=1) # utterance level representation e
        e = e.view(e.size()[0], -1) # flatten
        return e