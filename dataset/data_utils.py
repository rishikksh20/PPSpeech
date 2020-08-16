import random
import numpy as np
import torch
import torch.utils.data

from core import layers
from utils.utils import load_wav_to_torch, load_filepaths_and_text
from dataset.text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, pre_text, text, post_text = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3]
        pre_text = self.get_text(pre_text)
        text = self.get_text(text)
        post_text = self.get_text(post_text)
        # pre_mel, mel, post_mel = self.get_mel(audiopath)
        mel = self.get_mel(audiopath)
        return (pre_text, text, post_text, mel)

    def get_mel(self, filename):
        melspec = torch.from_numpy(np.load(filename.replace(".wav", ".npy")))
        # size_of_pre = torch.from_numpy(np.load(filename.replace(".wav", ".npy")))
        # size_of_current = torch.from_numpy(np.load(filename.replace(".wav", ".npy")))
        # size_of_post = torch.from_numpy(np.load(filename.replace(".wav", ".npy")))
        assert melspec.size(0) == self.stft.n_mel_channels, (
            'Mel dimension mismatch: given {}, expected {}'.format(
                melspec.size(0), self.stft.n_mel_channels))
        # pre_mel = melspec[:,:size_of_pre]
        # mel = melspec[:, size_of_pre:size_of_pre + size_of_current]
        # post_mel = melspec[:, size_of_pre + size_of_current:]
        # assert post_mel.size(1) == size_of_post
        # assert (pre_mel.size(1) + mel.size(1) + post_mel.size(1)) == melspec.size(1)
        # return pre_mel, mel, post_mel
        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """

        # Current Phrase
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[1]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][1]
            text_padded[i, :text.size(0)] = text

        # Pre Phrase
        pre_input_lengths, _ = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = pre_input_lengths[0]

        pre_text_padded = torch.LongTensor(len(batch), max_input_len)
        pre_text_padded.zero_()
        pre_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            pre_text_padded[i, :text.size(0)] = text
            pre_lengths[i] = text.size(0)


        # Post Phrase
        post_input_lengths, _ = torch.sort(
            torch.LongTensor([len(x[2]) for x in batch]),
            dim=0, descending=True)
        max_input_len = post_input_lengths[0]

        post_text_padded = torch.LongTensor(len(batch), max_input_len)
        post_text_padded.zero_()
        post_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][2]
            post_text_padded[i, :text.size(0)] = text
            post_lengths[i] = text.size(0)


        # Pre Mel

        # # Right zero-pad mel-spec
        # num_mels = batch[0][3].size(0)
        # max_target_len = max([x[3].size(1) for x in batch])
        #
        # # include mel padded and gate padded
        # pre_mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        # pre_mel_padded.zero_()
        # pre_output_lengths = torch.LongTensor(len(batch))
        # for i in range(len(ids_sorted_decreasing)):
        #     mel = batch[ids_sorted_decreasing[i]][3]
        #     pre_mel_padded[i, :, :mel.size(1)] = mel
        #     pre_output_lengths[i] = mel.size(1)

        # Current Mel
        # Right zero-pad mel-spec
        num_mels = batch[0][3].size(0)
        max_target_len = max([x[3].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][3]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        # Post Mel
        # # Right zero-pad mel-spec
        # num_mels = batch[0][3].size(0)
        # max_target_len = max([x[5].size(1) for x in batch])
        #
        # # include mel padded and gate padded
        # post_mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        # post_mel_padded.zero_()
        # post_output_lengths = torch.LongTensor(len(batch))
        # for i in range(len(ids_sorted_decreasing)):
        #     mel = batch[ids_sorted_decreasing[i]][5]
        #     post_mel_padded[i, :, :mel.size(1)] = mel
        #     post_output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, pre_text_padded, pre_lengths, post_text_padded, \
               post_lengths, mel_padded, gate_padded, output_lengths
