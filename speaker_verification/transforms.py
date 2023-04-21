import torch
import torchaudio
import torchvision

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from PIL import Image

class Image_Transforms:
    def __init__(self, 
                 model,
                 library="pytorch",
                 model_name = "resnet34",
                 resize=128):

        self.library = library
        self.model_name = model_name
        self.model = model

        if model_name == "resnet34":
            self.transform_image = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((resize,resize)),
            torchvision.transforms.ToTensor(), 
        ])

        if model_name == "vit_base_patch16_224":
            config = resolve_data_config({}, model=self.model)
            self.transform_image = create_transform(**config)

    def transform(self, image):
  
        if self.model_name == "vit_base_patch16_224":
            image = Image.fromarray(image)
        
        return self.transform_image(image)


class Audio_Transforms:
    def __init__(self, 
                sample_rate=16000,
                sample_duration=3, # seconds
                n_fft=512, # from Korean code
                win_length=400,
                hop_length=160,
                window_fn=torch.hamming_window,
                n_mels=40,
                ):

        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.n_mels = n_mels

        self.to_MelSpectrogram =  torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=torch.hamming_window,
            n_mels=self.n_mels
        )

    
    def transform(self, signal, sample_rate):

        # stereo --> mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # sample_rate --> 16000
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            signal = resampler(signal)

        # normalize duration --> 3 seconds (mean duration in dataset)
        sample_length_signal = self.sample_duration * self.sample_rate # sample length of the audio signal
        length_signal = signal.shape[1]
        if length_signal < sample_length_signal:
            num_missing_points = sample_length_signal - length_signal
            dim_padding = (0, num_missing_points) # (left_pad, right_pad)
            # ex: dim_padding = (1,2) --> [1,1,1] -> [0,1,1,1,0,0]
            signal = torch.nn.functional.pad(signal, dim_padding)
        elif length_signal > sample_length_signal:
            middle_of_the_signal = length_signal // 2
            left_edge = middle_of_the_signal - sample_length_signal // 2
            right_edge = middle_of_the_signal + sample_length_signal // 2
            signal = signal[:,left_edge:right_edge]
            
        # wav --> melspectrogram: 
        # [1, 44733] - [n_audio_channels, points] --> [1, 64, 88] - [n_channels, n_mels, number of frames]
        signal = self.to_MelSpectrogram(signal)

        return signal
                

# old code

def audio_transform(signal, sample_rate):

    # stereo --> mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    # sample_rate --> 16000
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        signal = resampler(signal)

    # normalize duration --> 3 seconds (mean duration in dataset)
    sample_duration = 3 # seconds
    sample_length_signal = sample_duration * 16000 # sample length of the audio signal
    length_signal = signal.shape[1]
    if length_signal < sample_length_signal:
        num_missing_points = sample_length_signal - length_signal
        dim_padding = (0, num_missing_points) # (left_pad, right_pad)
        # ex: dim_padding = (1,2) --> [1,1,1] -> [0,1,1,1,0,0]
        signal = torch.nn.functional.pad(signal, dim_padding)
    elif length_signal > sample_length_signal:
        middle_of_the_signal = length_signal // 2
        left_edge = middle_of_the_signal - sample_length_signal // 2
        right_edge = middle_of_the_signal + sample_length_signal // 2
        signal = signal[:,left_edge:right_edge]
        
    # wav --> melspectrogram: 
    # [1, 44733] - [n_audio_channels, points] --> [1, 64, 88] - [n_channels, n_mels, number of frames]
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    return transform(signal)
    
def image_transform(image, resize=128):

    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((resize,resize)),
            torchvision.transforms.ToTensor(), 
        ])

    return transform(image)
    
        

        