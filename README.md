# Introduction
This is ASR source code for Online Compressive Transformer from National Chiao Tung University, Taiwan.
The code also includes Synchronous transformer, it refer to the original paper[1].

## Requirements
- ESPnet
- Python 3.6.1+
- gcc 4.9+ for PyTorch1.0.0+

Optionally, GPU environment requires the following libraries:
- Cuda 8.0, 9.0, 9.1, 10.0 depending on each DNN library
- Cudnn 6+, 7+
- NCCL 2.0+ (for the use of multi-GPUs)

## Installation
The installation is the same with ESPnet verison 1. If you want to get more detail with installation, you can refer to [ESPnet tutorial](https://espnet.github.io/espnet/).

Please using this espnet version to install, the latest espnet maybe have something wrong to run our source code.
```sh
cd tools
make
```

## Train
Online compressive transformer doesn't use the language model, so we skip the stage 3.
```sh
cd egs/aishell/com_asr
# if you need prepare the dataset
./run_com.sh --stage -1  
```
```sh
cd egs/aishell/com_asr
# if you just train the model
./run_com.sh --stage 4 
```
## Inference 
tag is the model name in `egs/aishell/com_asr/exp/` folder.
```sh
cd egs/aishell/com_asr
./run_com.sh --stage 5 --recog_set "dev test" --tag "compressive_256GLU+CTC_.2_.3_.2_all_grad_ws9_ver2"
```

## model link
the trained models was put to gdrive, you can download in the following links and put in `egs/aishell/com_asr/exp/`.

### Aishell
- [Online compressive transformer version 1](https://drive.google.com/drive/folders/1wmG7B2Ld0k8B9aYqYJU7lhS9DiwNqNKq?usp=sharing)
- [Online compressive transformer version 2](https://drive.google.com/drive/folders/1Ex2qypmdnzLJDzH8LyXzbbfyuq4yVrhN?usp=sharing)
- [Synchronous transformer[1]](https://drive.google.com/drive/folders/1_OyG2p5EikSRP_T37odkH0-EHIdpDG-F?usp=sharing)
- [Transformer](https://drive.google.com/drive/folders/1pVspTH4ljY4ddYujlSo4PRKjDICC1reV?usp=sharing)

## Reference
[1] Tian, Z., Yi, J., Bai, Y., Tao, J., Zhang, S., & Wen, Z. (2020). Synchronous Transformers for end-to-end Speech Recognition. ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 7884-7888.

## Acknowledge
Online compressive Transformer uses [ESPNET1](https://github.com/espnet/espnet) framework.
