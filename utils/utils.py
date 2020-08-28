import numpy as np
from scipy.io.wavfile import read
import torch
import librosa


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def read_wav_np(path, sample_rate):
    sr, wav = read(path)
    if sr == sample_rate:

        if len(wav.shape) == 2:
            wav = wav[:, 0]

        if wav.dtype == np.int16:
            wav = wav / 32768.0
        elif wav.dtype == np.int32:
            wav = wav / 2147483648.0
        elif wav.dtype == np.uint8:
            wav = (wav - 128) / 128.0
    else:
        wav = librosa.load(path, sr=sample_rate)[0]

    wav = wav.astype(np.float32)

    return sr, wav


