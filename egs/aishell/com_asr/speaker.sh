#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=1        
stop_stage=12

voxceleb1_trials=data/voxceleb1_test/trials

nnet_dir=/home/container_user/hang/espnet/egs/aishell/com_asr/exp/vector_nnet_1a_ver1/

feat_tr_dir=dump/train/deltafalse; mkdir -p ${feat_tr_dir}
feat_test_dir=dump/voxceleb1_test/deltafalse; mkdir -p ${feat_test_dir}
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    fbankdir=fbank
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
		data/voxceleb1_test exp/make_fbank/voxceleb1_test ${fbankdir}
	dump.sh --cmd "$train_cmd" --nj 32 --do_delta false \
			data/train/feats.scp data/train_sp/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
	dump.sh --cmd "$train_cmd" --nj 20 --do_delta false \
        data/voxceleb1_test/feats.scp data/train_sp/cmvn.ark exp/dump_feats/voxceleb1_test ${feat_test_dir}
fi

dir=$nnet_dir/xvectors_train/
data_train=/home/container_user/hang/espnet/egs/aishell/com_asr/data/train/
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	echo "$0: extracting vector from nnet"
	${train_cmd} JOB=1:32 ${dir}/log/extract.JOB.log \
		python extract_speaker_feat.py JOB "train" || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
	echo "$0: combining xvectors across jobs"
	for j in $(seq 32); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;

	echo "$0: computing mean of xvectors for each speaker"
	${train_cmd} $dir/log/speaker_mean.log \
		ivector-mean ark:$data_train/spk2utt scp:$dir/xvector.scp \
		ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp ark,t:$dir/num_utts.ark || exit 1;

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
	echo "stage 3 "

fi

dir=$nnet_dir/xvectors_voxceleb1_test
data_train=/home/container_user/hang/espnet/egs/aishell/com_asr/data/voxceleb1_test/
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
	echo "$0: extracting vector from nnet"
	${train_cmd} JOB=1:20 ${dir}/log/extract.JOB.log \
		python extract_speaker_feat.py JOB "test" || exit 1;
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
	echo "$0: combining xvectors across jobs"
	for j in $(seq 20); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;

	echo "$0: computing mean of xvectors for each speaker"
	${train_cmd} $dir/log/speaker_mean.log \
		ivector-mean ark:$data_train/spk2utt scp:$dir/xvector.scp \
		ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp ark,t:$dir/num_utts.ark || exit 1;
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
	echo "stage 6 "
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
	echo "stage 7 "

fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
	echo "stage 8 "

fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
	echo "stage 9 "
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $nnet_dir/xvectors_train/log/compute_mean.log \
    ivector-mean scp:$nnet_dir/xvectors_train/xvector.scp \
    $nnet_dir/xvectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $nnet_dir/xvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train/xvector.scp ark:- |" \
    ark:data/train/utt2spk $nnet_dir/xvectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd $nnet_dir/xvectors_train/log/plda.log \
    ivector-compute-plda ark:data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $nnet_dir/xvectors_train/plda || exit 1;
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  $train_cmd $nnet_dir/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $nnet_dir/scores_voxceleb1_test || exit 1;
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials $nnet_dir/scores_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 $nnet_dir/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $nnet_dir/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 3.128%
  # minDCF(p-target=0.01): 0.3258
  # minDCF(p-target=0.001): 0.5003
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.329%
  # minDCF(p-target=0.01): 0.4933
  # minDCF(p-target=0.001): 0.6168
fi
