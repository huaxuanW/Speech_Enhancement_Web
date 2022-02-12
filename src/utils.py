import torch
import numpy as np
import librosa
import soundfile as sf
import os
from .model_pipeline import SePipline, load_model

def snr_mixer2(clean, noise, snr):
    eps= 1e-8
    s_pwr = np.var(clean)
    noise = noise - np.mean(noise)
    n_var = s_pwr / (10**(snr / 10))
    noise = np.sqrt(n_var) * noise / (np.std(noise) + eps)
    noisyspeech = clean + noise

    if max(abs(noisyspeech)) > 1:
        noisyspeech /= max(abs(noisyspeech))

    return noisyspeech, noise

def load_audio(path, sr= 16_000):
    """
    读取audio
    return numpy array
    """

    waveform, sr = librosa.load(path, sr=sr)

    return waveform

def save_audio(wav, path, sr=16_000):

    sf.write(path, wav, sr)

    return

def generate_test_files(clean_path, noise_path, mixed_path):
    
    clean = load_audio("static/upload_file/" + clean_path)

    noise = load_audio("static/temp/" + noise_path) 

    mixed = load_audio("static/temp/" + mixed_path)
               
    return {'mixed': mixed, 'clean': clean, 'noise': noise}

def noise_mixer(clean_path, noise_path, snr, mixed_path):
    clean = load_audio('static/upload_file/' + clean_path)
    noise = torch.load(noise_path)

    noise_start = np.random.randint(0, noise.shape[0] - clean.shape[0] + 1)

    noise_snippet = noise[noise_start : noise_start + clean.shape[0]]

    mixed, noise = snr_mixer2(clean, noise_snippet, snr)

    save_audio(mixed, "static/temp/" + mixed_path)

    save_audio(noise, "static/temp/temp_noise.wav")

    return mixed_path


def prediction(clean_path, noise_path, mixed_path, output_path):

    pretrain_model_path = "src/pretrained_model/" + "1best_model.pth.tar"

    SE = SePipline(
        version="v11",
        n_fft=512, 
        hop_len=128, 
        win_len= 512, 
        window="hanning",
        device="cpu",
        chunk_size=128,
        transform_type = "logmag",
        target = "mask",
        cnn = "2d")

    SE = load_model(model= SE, 
              optimizer= None,
              action= 'predict',
              pretrained_model_path = pretrain_model_path)
    
    dt = generate_test_files(clean_path, noise_path, mixed_path)

    for key, value in dt.items():

        dt[key] = torch.tensor(value, device = 'cpu')
    
    with torch.no_grad():

        dt = SE(dt, train=False)

    save_audio(dt['pred_y'], "static/upload_file/" + output_path)

    return
