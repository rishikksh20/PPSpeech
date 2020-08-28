import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.utils import read_wav_np
from utils.hparams import HParam

def main(args, hp):
    stft = TacotronSTFT(filter_length=hp.filter_length,
                        hop_length=hp.hop_length,
                        win_length=hp.win_length,
                        n_mel_channels=hp.n_mel_channels,
                        sampling_rate=hp.sampling_rate,
                        mel_fmin=hp.mel_fmin,
                        mel_fmax=hp.mel_fmax)

    wav_files = glob.glob(os.path.join(args.data_path, '**', '*.wav'), recursive=True)
    mel_path = hp.data_path
    os.makedirs(mel_path, exist_ok=True)
    print("Sample Rate : ", hp.sampling_rate)
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        sr, wav = read_wav_np(wavpath, hp.sampling_rate)
        wav = torch.from_numpy(wav).unsqueeze(0)      
        mel, mag = stft.mel_spectrogram(wav) # mel [1, 80, T]  mag [1, num_mag, T]
        mel = mel.squeeze(0)  # [num_mel, T]
        id = os.path.basename(wavpath).split(".")[0]
        np.save('{}/{}.npy'.format(mel_path,id), mel.numpy(), allow_pickle=False)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help="root directory of wav files")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)