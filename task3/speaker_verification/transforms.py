import torch
import torchaudio
import torchvision


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
    
def image_transform(image):

    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.ToTensor(), 
        ])

    return transform(image)