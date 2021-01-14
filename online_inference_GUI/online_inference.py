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

class online_inference():
    def __init__(self, model_src, mean=0, std=0):
        model, train_args = load_trained_model(model_src)
        is_sync = 'sync' in model_src
        self.sample_frequency = 16000
        self.model = model
        self.char_list = train_args.char_list
        self.parser = self.get_parser()
        self.hwsize = train_args.chunk_window_size
        self.hwsize = self.hwsize - train_args.chunk_overlapped if is_sync else self.hwsize
        self.overlapped = train_args.chunk_overlapped if is_sync else 0
        self.mean = mean
        self.std = std
        if "compressive" in train_args.model_module:
            if train_args.conv1d2decoder:
                self.name = "com2"
            else:
                self.name = "com1"
        else:
            self.name = "sync"
    
    def change_model(self, model_src):
        model, train_args = load_trained_model(model_src)
        self.model = model
    def change_cmvn(self, mean, std):
        if isinstance(mean,torch.Tensor) and isinstance(std,torch.Tensor):
            self.mean = mean
            self.std = std
            print("cmvn success")
        else:
            print("cmvn failed")

    def get_parser(self):
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

    def setup(self):
        self.model.online_recognize_setup(self.parser.beam_size)
        self.time_l = []
        self.text_l = []
        self.frame_l = []
        self.tmp = '>'

    def test_recognize_speed(self,feat):
        with torch.no_grad():
            self.setup()
            n_chunk = (len(feat))//(self.hwsize*4) #-self.overlapped*4
            self.get_inference(0,self.hwsize*4+self.overlapped*4+3,feat)
            for i in range(1,n_chunk-1):
                self.get_inference(i*(self.hwsize*4)+self.overlapped*4,(i+1)*(self.hwsize*4)+self.overlapped*4+3,feat)
            print(n_chunk, "".join(self.text_l), sum(self.time_l))

    def get_inference_wav(self,achunk_feat, end=0):
        a = time.time()
        nbest_hyps = self.model.online_recognize_each_chunk(achunk_feat, self.parser)
        b = time.time()
        self.model.update_commem()
        test = "".join([self.char_list[i] for i in nbest_hyps[0]["yseq"]])
        l = test.find(self.tmp,-4)
        self.tmp = test[-1]
        # self.time_l.append(b-a)
        # self.frame_l.append(end)
        # self.text_l.append(test[l+1:])
        return test, test[l+1:]
        # print(self.frame_l[-1], self.text_l[-1], b-a)

    def get_inference(self, left, right,feat):
        a = time.time()
        nbest_hyps = self.model.online_recognize_each_chunk(feat[left :right], self.parser)
        b = time.time()
        self.model.update_commem()
        #best = sorted(nbest_hyps, key=lambda x: x["score"], reverse=True)[0]
        test = "".join([self.char_list[i] for i in nbest_hyps[0]["yseq"]])
        l = test.find(self.tmp,-4)
        self.tmp = test[-1]
        #self.time_l.append(b-a)
        #self.frame_l.append(right)
        self.text_l.append(test[l+1:])
        #print(self.frame_l[-1], self.text_l[-1], b-a)

    def save_data(self):
        with open(self.name+".pickle","wb") as fp:
            pickle.dump(self.frame_l,fp)
            pickle.dump(self.text_l,fp)
            pickle.dump(self.time_l,fp)

    def fbank(self, wavform):
        wavform = torch.as_tensor(wavform, dtype=torch.float32).unsqueeze(0)
        fbank_array = torchaudio.compliance.kaldi.fbank(wavform, num_mel_bins=80,
                                     sample_frequency=self.sample_frequency, dither=1)
        return fbank_array

    def fbank_cmvn(self, wavform):
        fbank_array = self.fbank(wavform)
        fbank_array = (fbank_array - self.mean) / self.std
        return fbank_array
    
    def get_mean_std(self,wavform):
        fbank_array = self.fbank(wavform)
        std, mean = torch.std_mean(fbank_array, dim=0)
        return mean, std



def apply_cmvn(mat, stats):
    mean = stats[0, :-1] / stats[0, -1]
    variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
    return np.divide(np.subtract(mat, mean), np.sqrt(variance))

def normalization(self, feature):
    feature = torch.as_tensor(feature)
    std, mean = torch.std_mean(feature, dim=0)
    return (feature - mean) / std
def convert_fbank(wav_src):
    wavform, sample_frequency = torchaudio.load_wav(wav_src)
    feature = torchaudio.compliance.kaldi.fbank(wavform*32768, num_mel_bins=80, sample_frequency=sample_frequency, dither=1)
    return torch.as_tensor(feature)
def combine_all_speech(list_src):
    if len(list_src)==0: return
    all_feature = None
    for l in list_src:
        all_feature = torch.cat((all_feature,convert_fbank(l)),dim=0) if all_feature is not None else convert_fbank(l)
    return all_feature
if __name__ == "__main__":
    model_src_nopitch = "model/last5.avg.best35.model"
    all_feature = combine_all_speech(["demo/demo%03d.wav" % (i) for i in range(1,6)])
    #combine_all_speech(["demo/123.wav","demo/124.wav"]) #
    std, mean = torch.std_mean(all_feature, dim=0)
    no_pitch = online_inference(model_src_nopitch, std, mean)
    # with open("cmvn_new.pickle", 'wb') as fp:
    #     pickle.dump([mean, std], fp)
    
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
    for i in range(10):
        no_pitch.test_recognize_speed(demo_no_pitch)

    pass
