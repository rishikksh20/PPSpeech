import os
import time
import argparse
import tqdm
import torch
from dataset.text import symbols
from model import PPSpeech
from utils.hparams import HParam
from dataset.text import text_to_sequence
import numpy as np

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    #optimizer.load_state_dict(checkpoint_dict['optimizer'])
    #learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model

def main(checkpoint_path, hparams, reference_mel, pre_text, text, post_text ):
    model = PPSpeech(hparams, len(symbols)).cuda()
    model = load_checkpoint(checkpoint_path, model)
    model.eval()
    input_text = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners)).cuda().unsqueeze(0)
    pre_text = torch.LongTensor(text_to_sequence(pre_text, hparams.text_cleaners)).cuda().unsqueeze(0)
    post_text = torch.LongTensor(text_to_sequence(post_text, hparams.text_cleaners)).cuda().unsqueeze(0)
    reference = torch.from_numpy(np.load(reference_mel)).cuda().unsqueeze(0)
    with torch.no_grad():
        print("predicting")
        outs = model.inference(input_text, pre_text, post_text, style_input = reference)
        mel = outs[0]
        np.save("Mel.npy", mel.detach().cpu().numpy())
    print("Mel generated")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str,
                        help='Checkpoint path to read')
    parser.add_argument('-r', '--reference_mel', type=str,
                        help='Reference Mel for Acoustic Embedding')
    parser.add_argument('-p', '--pre_text', type=str,
                        help='Pre Text')
    parser.add_argument('-cu', '--text', type=str,
                        help='Current Text')
    parser.add_argument('-po', '--post_text', type=str,
                        help='Post Text')
    parser.add_argument('--config', type=str,
                        required=False, help='Config file')

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    torch.backends.cudnn.enabled = hp.cudnn_enabled
    torch.backends.cudnn.benchmark = hp.cudnn_benchmark

    print("cuDNN Enabled:", hp.cudnn_enabled)
    print("cuDNN Benchmark:", hp.cudnn_benchmark)

    main(args.checkpoint_path, hp, args.reference_mel, args.pre_text, args.text, args.post_text )
