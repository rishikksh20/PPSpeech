from math import sqrt
import torch
from torch import nn
from utils.utils import to_gpu, get_mask_from_lengths
from core.modules import Encoder, Decoder, Postnet
from core.gst import GST


class PPSpeech(nn.Module):
    def __init__(self, hparams, n_symbols):
        super(PPSpeech, self).__init__()
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
        self.context_encoder = GST(hparams, True)
        self.acoustic_embed = GST(hparams)

    def parse_batch(self, batch):
        current_text_padded, input_lengths, pre_text_padded, pre_text_len, post_text_padded, post_text_len, \
        mel_padded, gate_padded, output_lengths = batch

        current_text_padded = to_gpu(current_text_padded).long()
        input_lengths = to_gpu(input_lengths).long()

        pre_text_padded = to_gpu(pre_text_padded).long()
        pre_text_len = to_gpu(pre_text_len).long()

        post_text_padded = to_gpu(post_text_padded).long()
        post_text_len = to_gpu(post_text_len).long()

        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return ((current_text_padded, input_lengths, pre_text_padded, pre_text_len, post_text_padded, post_text_len, \
             mel_padded, max_len, output_lengths), (mel_padded, gate_padded))

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
        current_text, input_lengths, pre_text, pre_text_len, post_text, post_text_len , mels, max_len, \
        output_lengths = inputs
        input_lengths, pre_text_len, post_text_len, output_lengths = \
            input_lengths.data, pre_text_len.data, post_text_len.data, output_lengths.data

        embedded_inputs = self.embedding(current_text).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        # Context embedding
        pre_embedded_inputs = self.embedding(pre_text).transpose(1, 2)
        post_embedded_inputs = self.embedding(post_text).transpose(1, 2)

        pre_embed = self.encoder(pre_embedded_inputs, pre_text_len)
        post_embed = self.encoder(post_embedded_inputs, post_text_len)

        context_embed = self.context_encoder(pre_embed, pre_text_len, post_embed, post_text_len)

        context_embed = context_embed.repeat(1, encoder_outputs.size(1), 1)

        acoustic_embed = self.acoustic_embed(mels, output_lengths)
        acoustic_embed = acoustic_embed.repeat(1, encoder_outputs.size(1), 1)

        # Context concat to encoder output
        encoder_outputs = torch.cat(
                (encoder_outputs, acoustic_embed, context_embed), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=input_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, input, prefix, postfix, style_input = None):

        embedded_inputs = self.embedding(input).transpose(1, 2)
        pre_embedded_inputs = self.embedding(prefix).transpose(1, 2)
        post_embedded_inputs = self.embedding(postfix).transpose(1, 2)


        encoder_outputs = self.encoder.inference(embedded_inputs)
        pre_embed = self.encoder.inference(pre_embedded_inputs)
        post_embed = self.encoder.inference(post_embedded_inputs)

        pre_context_embed = self.prefix_embed(pre_embed)
        post_context_embed = self.postfix_embed(post_embed)
        context_embed = torch.cat((pre_context_embed, post_context_embed), dim=2)
        context_embed = context_embed.repeat(1, encoder_outputs.size(1), 1)

        acoustic_embed = self.acoustic_embed(style_input)

        acoustic_embed = acoustic_embed.repeat(1, encoder_outputs.size(1), 1)
        encoder_outputs = torch.cat(
                (encoder_outputs, acoustic_embed, context_embed), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
