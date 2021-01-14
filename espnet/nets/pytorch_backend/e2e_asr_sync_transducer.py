"""Transducer speech recognition model (pytorch)."""

from distutils.util import strtobool
import logging
import math

import chainer
from chainer import reporter
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask,transformerXL_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.pytorch_backend.sync_transducer.initializer import initializer
from espnet.nets.pytorch_backend.sync_transducer.loss import TransLoss
from espnet.nets.pytorch_backend.sync_transducer.rnn_decoders import decoder_for
from espnet.nets.pytorch_backend.sync_transducer.transformer_decoder import Decoder
from espnet.nets.pytorch_backend.sync_transducer.utils import prepare_loss_inputs
from espnet.nets.pytorch_backend.sync_transducer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


class Reporter(chainer.Chain):
    """A chainer reporter wrapper for transducer models."""

    def report(self, loss, cer, wer):
        """Instantiate reporter attributes."""
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        reporter.report({"loss": loss}, self)

        logging.info("loss:" + str(loss))


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (Namespace): argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Extend arguments for transducer models.

        Both Transformer and RNN modules are supported.
        General options encapsulate both modules options.

        """
        group = parser.add_argument_group("transformer model setting")

        # Encoder - general
        group.add_argument(
            "--etype",
            default="blstmp",
            type=str,
            choices=[
                "transformer",
                "lstm",
                "blstm",
                "lstmp",
                "blstmp",
                "vgglstmp",
                "vggblstmp",
                "vgglstm",
                "vggblstm",
                "gru",
                "bgru",
                "grup",
                "bgrup",
                "vgggrup",
                "vggbgrup",
                "vgggru",
                "vggbgru",
            ],
            help="Type of encoder network architecture",
        )
        group.add_argument(
            "--elayers",
            default=4,
            type=int,
            help="Number of encoder layers (for shared recognition part "
            "in multi-speaker asr mode)",
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=300,
            type=int,
            help="Number of encoder hidden units",
        )
        group.add_argument(
            "--dropout-rate",
            default=0.0,
            type=float,
            help="Dropout rate for the encoder",
        )
        # Encoder - RNN
        group.add_argument(
            "--eprojs", default=320, type=int, help="Number of encoder projection units"
        )
        group.add_argument(
            "--subsample",
            default="1",
            type=str,
            help="Subsample input frames x_y_z means subsample every x frame "
            "at 1st layer, every y frame at 2nd layer etc.",
        )
        # Attention - general
        group.add_argument(
            "--adim",
            default=320,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        group.add_argument(
            "--transformer-attn-dropout-rate-encoder",
            default=0.0,
            type=float,
            help="dropout in transformer decoder attention.",
        )
        group.add_argument(
            "--transformer-attn-dropout-rate-decoder",
            default=0.0,
            type=float,
            help="dropout in transformer decoder attention.",
        )
        # Attention - RNN
        group.add_argument(
            "--atype",
            default="location",
            type=str,
            choices=[
                "noatt",
                "dot",
                "add",
                "location",
                "coverage",
                "coverage_location",
                "location2d",
                "location_recurrent",
                "multi_head_dot",
                "multi_head_add",
                "multi_head_loc",
                "multi_head_multi_res_loc",
            ],
            help="Type of attention architecture",
        )
        group.add_argument(
            "--awin", default=5, type=int, help="Window size for location2d attention"
        )
        group.add_argument(
            "--aconv-chans",
            default=10,
            type=int,
            help="Number of attention convolution channels "
            "(negative value indicates no location-aware attention)",
        )
        group.add_argument(
            "--aconv-filts",
            default=100,
            type=int,
            help="Number of attention convolution filters "
            "(negative value indicates no location-aware attention)",
        )
        # Decoder - general
        group.add_argument(
            "--dtype",
            default="lstm",
            type=str,
            choices=["lstm", "gru", "transformer"],
            help="Type of decoder to use.",
        )
        group.add_argument(
            "--dlayers", default=1, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=320, type=int, help="Number of decoder hidden units"
        )
        group.add_argument(
            "--dropout-rate-decoder",
            default=0.0,
            type=float,
            help="Dropout rate for the decoder",
        )
        # Decoder - RNN
        group.add_argument(
            "--dec-embed-dim",
            default=320,
            type=int,
            help="Number of decoder embeddings dimensions",
        )
        group.add_argument(
            "--dropout-rate-embed-decoder",
            default=0.0,
            type=float,
            help="Dropout rate for the decoder embeddings",
        )
        # Transformer
        group.add_argument(
            "--transformer-warmup-steps",
            default=25000,
            type=int,
            help="optimizer warmup steps",
        )
        group.add_argument(
            "--transformer-init",
            type=str,
            default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
            ],
            help="how to initialize transformer parameters",
        )
        group.add_argument(
            "--transformer-input-layer",
            type=str,
            default="conv2d",
            choices=["conv2d", "vgg2l", "linear", "embed"],
            help="transformer encoder input layer type",
        )
        group.add_argument(
            "--transformer-dec-input-layer",
            type=str,
            default="embed",
            choices=["linear", "embed"],
            help="transformer decoder input layer type",
        )
        group.add_argument(
            "--transformer-lr",
            default=10.0,
            type=float,
            help="Initial value of learning rate",
        )
        # Transducer
        group.add_argument(
            "--trans-type",
            default="warp-transducer",
            type=str,
            choices=["warp-transducer"],
            help="Type of transducer implementation to calculate loss.",
        )
        group.add_argument(
            "--rnnt-mode",
            default="rnnt",
            type=str,
            choices=["rnnt", "rnnt-att"],
            help="Transducer mode for RNN decoder.",
        )
        group.add_argument(
            "--chunk-window-size",
            default=10,
            type=int,
            help="Number of encoder chunk window size",
        )
        group.add_argument(
            "--chunk-overlapped",
            default=3,
            type=int,
            help="Number of encoder chunk overlapped ",
        )
        group.add_argument(
            "--chunk-padding",
            type=strtobool,
            nargs="?",
            default=False,
            help="padding the zeros on chunk",
        )
        group.add_argument(
            "--score-norm-transducer",
            type=strtobool,
            nargs="?",
            default=True,
            help="Normalize transducer scores by length",
        )
        group.add_argument(
            "--load-pretrain",
            type=str,
            default=None,
            help="Normalize transducer scores by length",
        )
        
        group.add_argument(
            "--feedforwardGLU",
            default=False,
            type=strtobool,
            help="using GLU on feedforward layer",
        )
        group.add_argument(
            "--left_content",
            default=-1,
            type=int,
            help="only using left content",
        )

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0):
        """Construct an E2E object for transducer model.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        torch.nn.Module.__init__(self)

        if args.etype == "transformer":
            self.subsample = get_subsample(args, mode="asr", arch="transformer")

            self.encoder = Encoder(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                input_layer=args.transformer_input_layer,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate_encoder,
                feedforward_GLU=args.feedforwardGLU,
                # normalize_before=False #07/08 try not good
            )
        else:
            self.subsample = get_subsample(args, mode="asr", arch="rnn-t")

            self.enc = encoder_for(args, idim, self.subsample)

        if args.dtype == "transformer":
            self.decoder = Decoder(
                odim=odim,
                # jdim=args.joint_dim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                input_layer=args.transformer_dec_input_layer,
                dropout_rate=args.dropout_rate_decoder,
                positional_dropout_rate=args.dropout_rate_decoder,
                attention_dropout_rate=args.transformer_attn_dropout_rate_decoder,
                feedforward_GLU=args.feedforwardGLU,
            )
        else:
            if args.etype == "transformer":
                args.eprojs = args.adim

            if args.rnnt_mode == "rnnt-att":
                self.att = att_for(args)
                self.dec = decoder_for(args, odim, self.att)
            else:
                self.dec = decoder_for(args, odim)

        self.hwsize = args.chunk_window_size
        self.hb = self.hwsize - args.chunk_overlapped
        self.Padding = args.chunk_padding
        self.unfold = torch.nn.Unfold(kernel_size=(self.hwsize,args.adim),padding=0,stride=self.hb)

        self.etype = args.etype
        self.dtype = args.dtype
        self.rnnt_mode = args.rnnt_mode

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim
        self.adim = args.adim

        self.left_content = args.left_content


        self.reporter = Reporter()

        self.criterion = TransLoss(args.trans_type, self.blank_id)

        self.default_parameters(args)

        if args.report_cer or args.report_wer:
            from espnet.nets.e2e_asr_common import ErrorCalculatorTrans

            if self.dtype == "transformer":
                self.error_calculator = ErrorCalculatorTrans(self.decoder, args)
            else:
                self.error_calculator = ErrorCalculatorTrans(self.dec, args)
        else:
            self.error_calculator = None

        self.logzero = -10000000000.0
        self.loss = None
        self.rnnlm = None

    def default_parameters(self, args):
        """Initialize/reset parameters for transducer."""
        initializer(self, args)
        
        if args.load_pretrain is not None:
            path = args.load_pretrain
            model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(model_state_dict,strict=False)
            for k,v in model_state_dict.items():
                if k not in self.state_dict() or not torch.equal(v,self.state_dict()[k]):
                    logging.warning("weight not equal or not in this model: %s" % k)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): transducer loss value

        """
        # 1. encoder
        if self.etype == "transformer":
            xs_pad = xs_pad[:, : max(ilens)]
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(1)  # (batch, 1, ilen)
            if self.left_content !=-1:
                XLmask = transformerXL_mask(self.left_content,ilens.tolist()).to(xs_pad.device)
                src_mask = src_mask & XLmask
            hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)
        self.hs_pad = hs_pad
        #---------------New---------------
        # hs_pad_ = hs_pad.unsqueeze(2)
        # if((hs_pad_.size(1)-self.hwsize)%self.hb!=0 and self.Padding):
        #     hs_pad_ = torch.cat((hs_pad_,to_device(self,torch.zeros(hs_pad_.size(0),self.hb,1,self.adim))),dim=1)

        # reshape_hs = (); L_hs_pad=hs_pad_.size(1)
        # for i in range(self.hwsize):
        #     reshape_hs = reshape_hs + (hs_pad_[:,i:L_hs_pad-self.hwsize+i+1:self.hb,:],)
        # hs_pad_ = torch.cat(reshape_hs,2)
        # n_chunk = hs_pad_.size(1)


        hs_pad = hs_pad.unsqueeze(1)
        hs_pad_reshape = self.unfold(hs_pad)
        n_chunk = hs_pad_reshape.size(2)
        hs_pad_reshape = hs_pad_reshape.transpose(1,2)
        hs_pad_reshape = hs_pad_reshape.reshape(-1,n_chunk,self.hwsize,self.adim)

        # reshape_mask = (); L_hs_mask=hs_mask.size(-1)
        # for i in range(self.hwsize):
        #     reshape_mask = reshape_mask + (hs_mask[:,:,i:L_hs_mask-self.hwsize+i+1:self.hb],)
        # hs_mask_reshape = torch.cat(reshape_mask,1).transpose(-1,-2)
        #---------------New---------------

        # 1.5. transducer preparation related
        ys_in_pad, target, ys_mask, target_len = prepare_loss_inputs(ys_pad) # hs_mask_reshape[:,:,-1]

        pred_len = ((ilens-3)//4-self.hwsize)//self.hb+1
        pred_len = pred_len.to(ys_pad.device).type(torch.int32)

        # 2. decoder
        if self.dtype == "transformer":
            ys_in_pad = ys_in_pad.unsqueeze(1).expand(-1, n_chunk, -1)  #(batch_size, chunk, tgtsize)
            ys_mask = ys_mask.unsqueeze(1).expand(-1, n_chunk, -1, -1)  #(batch_size, chunk, tgtsize, tgtsize)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad_reshape, None) # None is hs_mask, using pred_len to mask

        self.pred_pad = pred_pad    # (batch_size,nchunk,nseq,tgtsize)
        # pred_pad = torch.log_softmax(pred_pad,dim=2) #log_softmax
        # 3. loss computation
        loss = self.criterion(pred_pad, target, pred_len, target_len)

        self.loss = loss
        loss_data = float(self.loss)

        # 4. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad_, ys_pad)

        if not math.isnan(loss_data):
            self.reporter.report(loss_data, cer, wer)
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def encode_transformer(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)

        """
        self.eval()

        x = to_device(self,torch.as_tensor(x).unsqueeze(0))
        if self.left_content == -1:
            enc_output, _ = self.encoder(x, None)
        else:
            mask = transformerXL_mask(self.left_content,[x.size(-2)])
            enc_output, _ = self.encoder(x,mask)


        return enc_output.squeeze(0)



    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            recog_args (namespace): argument Namespace containing options
            char_list (list): list of characters
            rnnlm (torch.nn.Module): language model module

        Returns:
            y (list): n-best decoding results

        """
        self.eval()
        with torch.no_grad():
            h = self.encode_transformer(x)

            #---------------New---------------
            # hs_pad_ = h.unsqueeze(0)
            # hs_pad_ = hs_pad_.unsqueeze(2)
            # if((hs_pad_.size(1)-self.hwsize)%self.hb!=0 and self.Padding):
            #     hs_pad_ = torch.cat((hs_pad_,to_device(self,torch.zeros(hs_pad_.size(0),self.hb,1,self.adim))),dim=1)
            # reshape_hs = (); L_hs_pad=hs_pad_.size(1)
            # for i in range(self.hwsize):
            #     reshape_hs = reshape_hs + (hs_pad_[:,i:L_hs_pad-self.hwsize+i+1:self.hb,:],)
            # hs_pad_ = torch.cat(reshape_hs,2)
            # n_chunk = hs_pad_.size(1)
            # hs_pad_ = hs_pad_.squeeze(0)

            hs_pad = h.unsqueeze(0)
            hs_pad = hs_pad.unsqueeze(1)
            hs_pad_reshape = self.unfold(hs_pad)
            n_chunk = hs_pad_reshape.size(2)
            hs_pad_reshape = hs_pad_reshape.transpose(1,2)
            hs_pad_reshape = hs_pad_reshape.reshape(-1,n_chunk,self.hwsize,self.adim)
            hs_pad_reshape = hs_pad_reshape.squeeze(0)
            #---------------New---------------
            recog_args.hwsize = self.hwsize
            recog_args.hb = self.hb
            recog_args.n_chunk = n_chunk
            params = [hs_pad_reshape, recog_args]

            if recog_args.beam_size == 1 or recog_args.beam_size==0:
                nbest_hyps = self.decoder.recognize(hs_pad_reshape, recog_args)
            else:
                #params.append(rnnlm)
                #nbest_hyps = self.decoder.recognize_beam(*params)
                nbest_hyps = self.decoder_recognize(hs_pad_reshape, recog_args)
            return nbest_hyps

    def online_recognize_setup(self, beam_size):
        if self.left_content !=-1:
            self.src_mask = to_device(self,transformerXL_mask(self.left_content//4,self.left_content//4+self.hb))
                            # because 2Layer conv2d, dim will divide 4
        else:
            pass #haven't thought it
            #self.src_mask = torch.tensor([True]*(self.hwsize*4+3)).reshape((1,1,-1))
        self.kv = None
        self.hi = 0
        self.hs_pad_temp = None
        if beam_size == 1 or beam_size==0:
            hyp = {"score": 0.0, "yseq": [self.blank_id]}
        else:
            hyp = {"score": 0.0, "yseq": torch.tensor([self.blank_id], dtype=torch.long)}
        self.hyps = [hyp]

    def online_recognize_each_chunk(self, x, recog_args):
        self.eval()
        x = to_device(self,torch.as_tensor(x).unsqueeze(0))
        d_x = (x.size(1)-3)//4 # conv2d divide 4
        if not self.kv:
            src_mask = self.src_mask[:,:d_x,:d_x]
            i = 0 # i is kv first node
        elif self.left_content//4 > self.kv[0].size(1):
            d_kv = self.kv[0].size(1)
            src_mask = self.src_mask[:, d_kv:d_kv+d_x, :d_kv+d_x]
            i = 0
        else:
            src_mask = self.src_mask[:, -d_x:, :]
            i = self.kv[0].size(1) - self.left_content//4 # i is kv first node

        hs_temp, _, self.kv= self.encoder.forward_one_step(x, src_mask, self.kv, i) 
                                              # batch, windows_size,adim   
        hs_temp = hs_temp.squeeze(0)                     
        self.hs_pad_temp = torch.cat((self.hs_pad_temp, hs_temp),dim=0) if self.hs_pad_temp is not None else hs_temp
        if self.hs_pad_temp.size(0)<self.hi+self.hb-1 or self.hs_pad_temp.size(0)< self.hwsize:
            return self.hyps
        hs = self.hs_pad_temp[self.hi:self.hi+self.hwsize, :]

        if recog_args.beam_size == 1 or recog_args.beam_size==0:
            self.hyps = self.hyps[0]
            self.hyps = self.decoder.recognize_each_chunk(self.hyps , hs)
            self.hyps = [self.hyps]
        else:
            self.hyps =self.decoder_each_chunk_beam_search(self.hyps, hs)
        self.hi += self.hb
        return self.hyps
    def update_commem(self):
        pass #Nothing needs update
    def decoder_recognize(self,h,recog_args): 
        # search parms
        beam = recog_args.beam_size
        nbest = recog_args.nbest

        #initialize hypothesis
        hyp = {"score": 0.0, "yseq": torch.tensor([self.blank_id], dtype=torch.long)}
        hyps = [hyp] 
        
        for i,hi in enumerate(h):
            hyps=self.decoder_each_chunk_beam_search(hyps,hi,beam=beam)
        nbest_hyps = sorted(hyps, key=lambda x: x["score"], reverse=True)[:nbest]
        return nbest_hyps

    def decoder_each_chunk_beam_search(self, hyps, hi, h_mask=None, beam=5, times=3):
        hyps_yseq = [h["yseq"] for h in hyps]
        hyps_len = [len(h["yseq"]) for h in hyps]
        hyps_score = torch.tensor([h["score"] for h in hyps])
        ys = to_device(self, pad_list(hyps_yseq, self.blank_id)).unsqueeze(1) #(batch,1, tgtsize)
        hi = hi.unsqueeze(0).unsqueeze(1).expand(ys.size(0),-1,-1,-1) # (batch,1,nwindow, adim)
        ys_mask = to_device(
            self, subsequent_mask(ys.size(-1)).unsqueeze(0).unsqueeze(0) #(1, 1, tgtsize, tgtsize)
        )
        scores=self.decoder.forward_one_step_forbatch(ys, ys_mask, hyps_len, hi, h_mask)
        n_tokens = scores.size(1)-1
        hyps_blank_score = hyps_score + scores[:,0]

        expan_blank_score, expan_hyps_yseq = [], []

        for ex_i in range(3): # means one chunk generate 2 word at most 
            if ex_i==0:
                score_expan = scores[:,1:].contiguous().view(-1)
                hyps_score_expan = hyps_score.unsqueeze(1).expand(-1,n_tokens).contiguous().view(-1) \
                                    + score_expan
                expan_scores, expan_ids = torch.topk(hyps_score_expan, beam)
                # Expansion 
                expan_hyps_yseq.append([torch.cat((hyps_yseq[expan_ids[i]//n_tokens],
                                            hyps_yseq[0].new([expan_ids[i]%n_tokens+1]))) 
                                            for i in range(beam)])
            else:
                score_expan = scores[:,1:].contiguous().view(-1)
                hyps_score_expan = expan_scores.unsqueeze(1).expand(-1,n_tokens).contiguous().view(-1) \
                                    + score_expan
                expan_scores, expan_ids = torch.topk(hyps_score_expan, beam)
                expan_hyps_yseq.append([torch.cat((expan_hyps_yseq[ex_i-1][expan_ids[i]//n_tokens],
                                            hyps_yseq[0].new([expan_ids[i]%n_tokens+1]))) 
                                            for i in range(beam)])
                
            hyps_lens_expan = [h.size(0) for h in expan_hyps_yseq[ex_i]]
            ys_expan = to_device(self, pad_list(expan_hyps_yseq[ex_i], self.blank_id)).unsqueeze(1) #(batch,1, tgtsize)
            ys_mask = to_device(
                self, subsequent_mask(ys_expan.size(-1)).unsqueeze(0).unsqueeze(0) #(1, 1, tgtsize, tgtsize)
            )
            hi = hi.expand(ys_expan.size(0),-1,-1,-1)
            scores=self.decoder.forward_one_step_forbatch(ys_expan, ys_mask, hyps_lens_expan, hi, h_mask)
            expan_blank_score.append(expan_scores + scores[:,0])

        final_score, final_ids = torch.topk(torch.cat((hyps_blank_score,torch.cat(expan_blank_score))),beam)
        hyps = []
        n_size = hyps_blank_score.size(0)
        for i in range(beam):
            ids = final_ids[i]
            if ids<hyps_blank_score.size(0): #means is hyps_blank_score
                hyp = {"score": hyps_blank_score[ids], "yseq": hyps_yseq[ids]}
            else:
                ids = ids - n_size
                hyp = {"score": expan_blank_score[ids//beam][ids % beam],
                        "yseq": expan_hyps_yseq[ids//beam][ids % beam]}
            hyps.append(hyp)
        return hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax)

        Returns:
            ret (ndarray): attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).

        """
        if (
            self.etype == "transformer"
            and self.dtype != "transformer"
            and self.rnnt_mode == "rnnt-att"
        ):
            raise NotImplementedError(
                "Transformer encoder with rnn attention decoder" "is not supported yet."
            )
        elif self.etype != "transformer" and self.dtype != "transformer":
            if self.rnnt_mode == "rnnt":
                return []
            else:
                with torch.no_grad():
                    hs_pad, hlens = xs_pad, ilens
                    hpad, hlens, _ = self.enc(hs_pad, hlens)

                    ret = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)
        else:
            with torch.no_grad():
                self.forward(xs_pad, ilens, ys_pad)

                ret = dict()
                for name, m in self.named_modules():
                    if isinstance(m, MultiHeadedAttention):
                        ret[name] = m.attn.cpu().numpy()

        return ret

    def plot_predict(self, xs_pad, ilens, ys_pad,token=None,name=None):
        self.eval()
        xs_pad = to_device(self,torch.as_tensor(xs_pad).unsqueeze(0))
        for i in range(len(ys_pad)):
            ys_pad[i] = int(ys_pad[i])
        ys_pad = to_device(self,torch.as_tensor(ys_pad).unsqueeze(0))
        ilens = to_device(self,torch.as_tensor(ilens))
        with torch.no_grad():
            # 1. encoder
            if self.etype == "transformer":
                xs_pad = xs_pad[:, : max(ilens)]
                src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

                hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
            else:
                hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)
            self.hs_pad = hs_pad

            #---------------New--------------
            hs_pad = hs_pad.unsqueeze(1)
            hs_pad_reshape = self.unfold(hs_pad)
            n_chunk = hs_pad_reshape.size(2)
            hs_pad_reshape = hs_pad_reshape.transpose(1,2)
            hs_pad_reshape = hs_pad_reshape.reshape(-1,n_chunk,self.hwsize,self.adim)
            #---------------New---------------
            ys_pad = ys_pad.unsqueeze(0)
            # 1.5. transducer preparation related
            ys_in_pad, target, pred_len, target_len = prepare_loss_inputs(ys_pad, torch.ones(hs_pad_reshape.size(0),n_chunk))

            
            # 2. decoder
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            ys_in_pad = ys_in_pad.unsqueeze(1).expand(-1, n_chunk, -1)  #(batch_size, chunk, tgtsize)
            ys_mask = ys_mask.unsqueeze(1).expand(-1, n_chunk, -1, -1)  #(batch_size, chunk, tgtsize, tgtsize)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad_reshape, None) # None is hs_mask, if input hs_mask will error

            pred_pad = torch.log_softmax(pred_pad,-1)
        
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn import preprocessing
        from scipy.special import softmax

        norm = preprocessing.MinMaxScaler()
        p = pred_pad.squeeze(0).cpu().numpy()
        y = ys_in_pad[0,0,:].cpu().numpy()
        output = []
        for i in range(p.shape[0]):
            temp =[]
            for j in range(p.shape[1]):
                p_norm = p[i,j,:] #.reshape(-1,1)
                temp.append(p_norm[y])
            output.append(softmax(np.array(temp)))
        output_np = np.array(output)

        link ="photo/%s/" %(name)
        import os
        if not os.path.exists(link):
            os.mkdir(link)
        token.insert(0,'<blk>')
        final = np.zeros_like(output_np[0])
        for i in range(len(output_np)):
            plt.imshow(output_np[i].T.astype(np.float32), aspect="auto")
            plt.title("chunk %d" % i)
            plt.xlabel("input")
            plt.ylabel("ouput")
            plt.xticks(np.arange(0,len(token)),token)
            plt.yticks(np.arange(0,len(token)),token)

            plt.savefig("%s/a%02d.png" %(link,i))
            final = final + output_np[i].T.astype(np.float32)


        plt.imshow(final, aspect="auto")
        plt.title("Sum of chunks")
        plt.xlabel("input")
        plt.ylabel("ouput")
        plt.xticks(np.arange(0,len(token)),token)
        plt.yticks(np.arange(0,len(token)),token)
        plt.savefig("%s/final.png" %(link))



