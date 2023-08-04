import torch
import torchaudio
import torchvision.transforms as T
from transformers import ASTFeatureExtractor
from transformers import Wav2Vec2FeatureExtractor


class Image_Transforms:
    def __init__(self, 
                 library,
                 model_name, modality):

        self.library = library
        self.model_name = model_name
        self.modality = modality
        
        if self.library == "huggingface":
            pass
        elif self.library == "timm":
            self.timm_init()
        elif self.library == "pytorch":
            self.pytorch_init()

    def timm_init(self):
        if self.model_name == "vit_base_patch16_224":
            # n_channels = 3
            # mean and std computed for SF dataset only
            if self.modality == "rgb":
                mean_val = [0.35843268, 0.27319421, 0.23963803]
                std_val = [0.15948673, 0.13425587, 0.1222331]
                
            elif self.modality == "thr":
                mean_val = [0.94060585, 0.74481036, 0.29649508]
                std_val = [0.15892989, 0.27409379, 0.26099585]
            
            self.transform_image=T.Compose([
                T.ToPILImage(),
                T.Resize(size=(224,224)),
                T.ToTensor(), 
                T.Normalize(mean=torch.tensor(mean_val), std=torch.tensor(std_val))  
            ])
        else:
            self.transform_image = T.Compose([
                T.ToPILImage(),
                T.Resize((128,128)),
                T.ToTensor(), 
            ])
        
    def pytorch_init(self):
        self.transform_image = T.Compose([
                T.ToPILImage(),
                T.Resize((128,128)),
                T.ToTensor(), 
            ])

    # MAIN TRANSFORM FUNCTION
    def transform(self, image):
            return self.transform_image(image)


    
class Audio_Transforms:
    def __init__(self, 
                sample_rate,
                sample_duration, # seconds
                n_fft, # from Korean code
                win_length,
                hop_length,
                window_fn,
                n_mels,
                model_name, 
                library
                ):

        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.n_mels = n_mels
        self.model_name = model_name
        self.library = library

        if self.library == "huggingface":
            self.huggingface_init()
        elif self.library == "timm":
            self.timm_init()
        elif self.library == "pytorch":
            self.pytorch_init()

    def huggingface_init(self):
        if self.model_name == "WavLM":
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
        elif self.model_name == "AST":
            self.feature_extractor = ASTFeatureExtractor()

    def timm_init(self):
        self.to_MelSpectrogram =  torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                window_fn=self.window_fn,
                n_mels=self.n_mels
            )

        if self.model_name == "vit_base_patch16_224":
            # n_channels = 1
            self.vit_transform=T.Compose([
                T.Resize(size=(224,224))
                #T.ToPILImage(),
                #T.Resize(size=(224,224), interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=None),
                #T.ToTensor(),
                #T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
            ])

    def pytorch_init(self):
        self.to_MelSpectrogram =  torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                window_fn=self.window_fn,
                n_mels=self.n_mels
            )

    # MAIN TRANSFORM FUNCTION
    def transform(self, signal, sample_rate):

        signal = self.basic_transform(signal, sample_rate)

        if self.library == "huggingface":
            inputs = self.huggingface_transform(signal)
        elif self.library == "timm":
            inputs = self.timm_transform(signal)
        elif self.library == "pytorch":
            inputs = self.pytorch_transform(signal)
        
        return inputs
    
    def basic_transform(self, signal, sample_rate):

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
            num_missing_points = int(sample_length_signal - length_signal)
            dim_padding = (0, num_missing_points) # (left_pad, right_pad)
            # ex: dim_padding = (1,2) --> [1,1,1] -> [0,1,1,1,0,0]
            signal = torch.nn.functional.pad(signal, dim_padding)
        elif length_signal > sample_length_signal:
            middle_of_the_signal = length_signal // 2
            left_edge = int(middle_of_the_signal - sample_length_signal // 2)
            right_edge = int(middle_of_the_signal + sample_length_signal // 2)
            signal = signal[:,left_edge:right_edge]

        return signal
    
    def huggingface_transform(self, audio):
        input = audio.squeeze()
        input = self.feature_extractor(input, sampling_rate=self.sample_rate, padding=True, return_tensors="pt")
        input = input.input_values.squeeze()
        return input

    def timm_transform(self, audio):
        input = self.to_MelSpectrogram(audio)
        if self.model_name == "vit_base_patch16_224":
            input = input.repeat(3, 1, 1)
            input = self.vit_transform(input)
        return input

    def pytorch_transform(self, audio):
        input = self.to_MelSpectrogram(audio)
        return input
