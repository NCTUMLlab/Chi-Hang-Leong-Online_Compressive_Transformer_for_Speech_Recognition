import time
from threading import Thread,Event
import numpy as np
import torch
import time
from espnet.asr.pytorch_backend.asr_init import load_trained_model
import torchaudio
import argparse
import pickle
from espnet.nets.pytorch_backend.nets_utils import pad_list
from online_inference import online_inference 

def load_model(model_src, parser):
    model, train_args = load_trained_model(model_src)
    is_sync = 'sync' in model_src
    online_model = online_inference(model, train_args, parser, is_sync)
    return online_model

def apply_cmvn(mat, stats):
    mean = stats[0, :-1] / stats[0, -1]
    variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
    return np.divide(np.subtract(mat, mean), np.sqrt(variance))

def normalization(feature):
    feature = torch.as_tensor(feature)
    std, mean = torch.std_mean(feature, dim=0)
    return (feature - mean) / std
def convert_fbank(wav_src):
    wavform, sample_frequency = torchaudio.load_wav(wav_src)
    feature = torchaudio.compliance.kaldi.fbank(wavform*32768, num_mel_bins=80, sample_frequency=sample_frequency, dither=1)
    return torch.as_tensor(feature)
def get_parser():
    parser = argparse.ArgumentParser()
    parser.ctc_weight = 0.0
    parser.beam_size = 5
    parser.penalty = 0.0
    parser.maxlenratio= 0.0
    parser.minlenratio= 0.0
    parser.nbest = 1
    torch.manual_seed(1)  
    torch.cuda.manual_seed_all(1)  
    return parser
def combine_all_speech(list_src):
    if len(list_src)==0: return
    all_feature = None
    for l in list_src:
        all_feature = torch.cat((all_feature,convert_fbank(l)),dim=0) if all_feature is not None else convert_fbank(l)
    return all_feature

if __name__ == "__main__":
    parser = get_parser()
    model_src_nopitch = "model/model.last5.avg.best35"
    no_pitch = load_model(model_src_nopitch, parser)
    all_feature = combine_all_speech(["demo/demo%03d.wav" % (i) for i in range(1,6)])
    
    std, mean = torch.std_mean(all_feature, dim=0)
    # std, mean = torch.std_mean(all_feature, dim=0)
    wavform, sample_frequency = torchaudio.load_wav("demo/demo002.wav")
    #0.015*16000=240
    no_pitch.setup()
    feature_all = None
    # feature_all = torchaudio.compliance.kaldi.fbank(wavform[:,0:(36*160+240)]*32768, num_mel_bins=80, sample_frequency=sample_frequency, dither=1)
    for i in range(0,14):
        feature_x = torchaudio.compliance.kaldi.fbank(wavform[:,36*i*160:36*160*(i+1)+3*160+240]*32768, num_mel_bins=80, sample_frequency=sample_frequency, dither=1)
        feature_all = torch.cat((feature_all,feature_x),dim=0) if feature_all is not None else feature_x
        feature_x = (feature_x-mean)/std
        no_pitch.get_inference_wav(feature_x)
    print("".join(no_pitch.text_l), sum(no_pitch.time_l))
    demo_no_pitch = (feature_all-mean)/std
    no_pitch.test_recognize_speed(demo_no_pitch)

    pass
