from math import sqrt
import torch
from torch import nn
from utils.utils import to_gpu, get_mask_from_lengths
from core.modules import Encoder, Decoder, Postnet
from core.gst import GST


class Tacotron2(nn.Module):
    def __init__(self, hparams, n_symbols):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        if hparams.with_gst:
            self.gst = GST(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        if hasattr(self, 'gst'):
            embedded_gst = self.gst(mels, output_lengths)
            embedded_gst = embedded_gst.repeat(1, encoder_outputs.size(1), 1)

            encoder_outputs = torch.cat(
                (encoder_outputs, embedded_gst), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, input, style_input):

        embedded_inputs = self.embedding(input).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        if hasattr(self, 'gst'):
            if isinstance(style_input, int):
                query = torch.zeros(1, 1, self.gst.encoder.ref_enc_gru_size).cuda()
                GST = torch.tanh(self.gst.stl.embed)
                key = GST[style_input].unsqueeze(0).expand(1, -1, -1)
                embedded_gst = self.gst.stl.attention(query, key)
            else:
                embedded_gst = self.gst(style_input)

            embedded_gst = embedded_gst.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat(
                (encoder_outputs, embedded_gst), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
