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

def sentence_to_phrases(line, words=3):
    wordlist = line.split()
    phrases = []
    for i in range(0, len(wordlist), words):
        if i + words > len(wordlist):
            phrases[-1] = phrases[-1] +" "+' '.join(wordlist[i:i+words])
        else:
            phrases.append(' '.join(wordlist[i:i+words]))
    return phrases

def create_phrase_data(lines, words=4):
    #
    dataset=[]
    for line in lines:
        pre = []
        current = []
        post = []
    if len(line.split()) < 8:
        pre.append("^")
        current.append(line)
        post.append("~")
    else:
        phrases = sentence_to_phrases(line, words=words)
      #print(phrases)
        for i in range(0, len(phrases), 1):

            if i == 0:
                pre.append("^")
            else:
                pre.append(phrases[i-1])

            current.append(phrases[i])

            if i == len(phrases)-1:
                post.append("~")
                break
            else:
                post.append(phrases[i+1])
    return pre, current, post

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

def process_input(text, mode):
    dataset = []
    if mode == 0:
        pre = ["^"]
        post = ["~"]
        current = text.strip().split()
    if mode == 1:
        pre, current, post = create_phrase_data(text, 5)

    for i, j, k in zip(pre, current, post):
        dataset.append("{}|{}|{}".format(i,j,k))

    return dataset


def main(checkpoint_path, hparams, reference_mel, dataset, name ):
    model = PPSpeech(hparams, len(symbols)).cuda()
    model = load_checkpoint(checkpoint_path, model)
    model.eval()
    output = []
    for point in dataset:
        input_text = torch.LongTensor(text_to_sequence(point[1], hparams.text_cleaners)).cuda().unsqueeze(0)
        pre_text = torch.LongTensor(text_to_sequence(point[0], hparams.text_cleaners)).cuda().unsqueeze(0)
        post_text = torch.LongTensor(text_to_sequence(point[2], hparams.text_cleaners)).cuda().unsqueeze(0)
        reference = torch.from_numpy(np.load(reference_mel)).cuda().unsqueeze(0)
        with torch.no_grad():
            print("predicting")
            outs = model.inference(input_text, pre_text, post_text, style_input = reference)
            mel = outs[0]
            output.append(mel)
    if len(output)==1:
        output_mel = output[0]
    else:
        output_mel = np.concatenate(output, axis = 1)

    np.save(f"PPSpeech_{name}.npy", mel.detach().cpu().numpy())
    print(f"Mel generated with name PPSpeech_{name}.npy ")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str,
                        help='Checkpoint path to read')
    parser.add_argument('-r', '--reference_mel', type=str,
                        help='Reference Mel for Acoustic Embedding')
    parser.add_argument('-t', '--text', type=str,
                        help='Input Text')
    parser.add_argument('--config', type=str,
                        required=True, help='Config file')
    parser.add_argument('-n','--name', type=str,
                        required=True, help='Name of the file to be saved')
    parser.add_argument('-m','--mode', type=int,
                        required=True, help='Mode of inference: 0 - Sentence , 1 - Phrase ')

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    torch.backends.cudnn.enabled = hp.cudnn_enabled
    torch.backends.cudnn.benchmark = hp.cudnn_benchmark

    print("cuDNN Enabled:", hp.cudnn_enabled)
    print("cuDNN Benchmark:", hp.cudnn_benchmark)

    dataset = process_input(args.text, args.mode)
    main(args.checkpoint_path, hp, args.reference_mel, dataset , args.name)
