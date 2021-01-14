"""Transducer speech recognition model (pytorch)."""

from distutils.util import strtobool
import logging
import math

import chainer
from chainer import reporter
import torch
import numpy

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask,transformerXL_mask
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.pytorch_backend.compressive_transducer.initializer import initializer
from espnet.nets.pytorch_backend.compressive_transducer.loss import TransLoss
from espnet.nets.pytorch_backend.compressive_transducer.rnn_decoders import decoder_for
from espnet.nets.pytorch_backend.compressive_transducer.transformer_decoder import Decoder
from espnet.nets.pytorch_backend.compressive_transducer.utils import prepare_loss_inputs
from espnet.nets.pytorch_backend.compressive_transducer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.ctc import CTC

class Reporter(chainer.Chain):
    """A chainer reporter wrapper for transducer models."""

    def report(self, loss ,loss_rnnt, loss_ctc, loss_mem, loss_spk, loss_attndec_data):
        """Instantiate reporter attributes."""
        reporter.report({"loss": loss}, self)
        reporter.report({"loss_rnnt": loss_rnnt}, self)
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"loss_mem": loss_mem}, self)
        reporter.report({"loss_spk": loss_spk}, self)
        reporter.report({"loss_attndec": loss_attndec_data}, self)

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
            "--left-content",
            default=-1,
            type=int,
            help="only using left content",
        )
        group.add_argument(
            "--compressive-rate",
            default=3,
            type=int,
            help="the compressive_rate like 1d conv",
        )
        group.add_argument(
            "--memspeech-size",
            default=10,
            type=int,
            help="the memory speech size",
        )
        group.add_argument(
            "--memspeaker-size",
            default=3,
            type=int,
            help="the memory speaker size",
        )
        group.add_argument(
            "--spkodim",
            default=0,
            type=int,
            help="when use speaker feature using it",
        )
        group.add_argument(
            "--conv1d2decoder",
            default=False,
            type=strtobool,
            help="conv1d2decoder",
        )
        group.add_argument(
            "--speaker2decoder",
            default=False,
            type=strtobool,
            help="speaker2decoder",
        )
        group.add_argument(
            "--memattnloss_decoder",
            default=False,
            type=strtobool,
            help="memattnloss_decoder",
        )
        group.add_argument(
            "--usespk_version2",
            default=False,
            type=strtobool,
            help="usespk_version2",
        )
        
        group.add_argument(
            "--spkadim",
            default=256,
            type=int,
            help="spk linear transform dim",
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
                compressive_rate=args.compressive_rate,
                memspeech_size=args.memspeech_size,
                # normalize_before=False #07/08 try
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
        self.ctc = CTC(
            odim, args.adim, args.dropout_rate, reduce=True
        )
        self.hwsize = args.chunk_window_size
        # self.hb = self.hwsize - args.chunk_overlapped
        # self.Padding = args.chunk_padding
        self.unfold = torch.nn.Unfold(kernel_size=(self.hwsize*4+3,idim),padding=0,stride=self.hwsize*4) #,stride=self.hb*4

        self.alllayer2ctc = None # torch.nn.Linear(args.adim*args.elayers,args.adim,bias=False) don't work
        self.conv1d2decoder = None
        if args.conv1d2decoder:
            self.conv1d2decoder = torch.nn.Conv1d(in_channels=args.adim,out_channels=args.adim,
                                                kernel_size=args.compressive_rate,stride=args.compressive_rate)
        self.speaker2decoder = None
        if args.speaker2decoder:
            if not args.usespk_version2:
                self.speaker2decoder = torch.nn.Conv1d(in_channels=args.adim,out_channels=args.adim,
                                                kernel_size=args.compressive_rate,stride=args.compressive_rate)
            if args.spkodim ==0:
                logging.error("if use the speaker feature pls setting spkodim")
            self.spklinear = torch.nn.Sequential(torch.nn.Linear(args.adim,args.spkadim),
                                                    torch.nn.Linear(args.spkadim,args.spkodim))
            self.spkloss = torch.nn.CrossEntropyLoss()
            self.spkodim = args.spkodim
        if args.usespk_version2 and args.speaker2decoder:
            self.soft_linear = torch.nn.Linear(args.adim,args.adim*2)
            self.spklinear2em = torch.nn.Linear(args.adim,args.adim)
            self.shlinear2em = torch.nn.Linear(args.adim,args.adim)

        self.memattnloss_decoder = args.memattnloss_decoder
        self.usespk2decoder = args.speaker2decoder
        self.usespk_version2 = args.usespk_version2
        self.etype = args.etype
        self.dtype = args.dtype
        self.rnnt_mode = args.rnnt_mode

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.idim = idim
        self.odim = odim
        self.adim = args.adim

        self.left_content = args.left_content

        self.memspeaker_size = args.memspeaker_size
        self.memspeech_size = args.memspeech_size
        self.compressive_rate = args.compressive_rate

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
            # self.load_state_dict(model_state_dict,strict=False)
            logging.warning("the pretrain model is %s" % path)
            for k,v in model_state_dict.items():
                if k in self.state_dict() and v.shape == self.state_dict()[k].shape:
                    self.state_dict()[k].copy_(v)
                if k not in self.state_dict() or not torch.equal(v,self.state_dict()[k]):
                    logging.warning("weight not equal or not in this model: %s" % k)

    def reshapetochunk(self,xs_pad):
        xs_pad = xs_pad.unsqueeze(1)
        xs_pad_reshape = self.unfold(xs_pad)
        n_chunk = xs_pad_reshape.size(2)
        xs_pad_reshape = xs_pad_reshape.transpose(1,2)
        xs_pad_reshape = xs_pad_reshape.reshape(-1,n_chunk,self.hwsize*4+3,self.idim)
        
        return xs_pad_reshape

    def initial_state(self,batch_size,memspeech_size,layer_len):
        memsh = [to_device(self, torch.zeros(batch_size,memspeech_size,self.adim)) for _ in range(layer_len)]
        memsh_mask = to_device(self, torch.zeros(batch_size,1,memspeech_size))   #(batch,  1, memspeech_size)
        return memsh,memsh_mask


    def forward(self, xs_pad, ilens, ys_pad, spkid=None):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): transducer loss value

        """
        # 1. encoder
        xs_pad = xs_pad[:, : max(ilens)]

        # pred_len = ((ilens-3)//4-self.hwsize)//self.hb+1  #because two conv, will be downsampling
        pred_len = (ilens-3)//4//self.hwsize
        pred_len = pred_len.to(xs_pad.device).type(torch.int32)

        src_mask = make_non_pad_mask(pred_len.tolist()).to(xs_pad.device)  # (batch, nchunk)
        src_mask_expand = src_mask.unsqueeze(-1).unsqueeze(-2).expand(-1,-1,-1,self.hwsize*4+3) # (batch, nchunk, 1,hwsize*4+3)
        xs_pad_reshape = self.reshapetochunk(xs_pad)
        n_chunk = xs_pad_reshape.size(1)

        hs_pad = tuple()
        hs_pad_ctc = tuple()
        self.whole_hs = tuple() #analysis
        memsh, memsh_mask = self.initial_state(xs_pad.size(0),self.memspeech_size,len(self.encoder.encoders))
        memsh_mask = memsh_mask.type(src_mask_expand.dtype)
        mem_loss_total = None
        last_memsh = to_device(self, torch.zeros(xs_pad.size(0),self.memspeech_size,self.adim))
        memspk = to_device(self, torch.zeros(xs_pad.size(0),self.memspeaker_size,self.adim))
        memspk_mask = to_device(self, torch.zeros(xs_pad.size(0),1,self.memspeaker_size)).type(src_mask_expand.dtype)
        hs_pad_mask = []
        self.plot_enc_attn = []
        x_mems = []
        commems = []
        for i in range(n_chunk):
            hs_pad_temp, memsh_mask, memsh, mem_loss = self.encoder(xs_pad_reshape[:,i,:,:], src_mask_expand[:,i,:,:],
                                                            memsh, memsh_mask)  # batch, windows_size,adim
            # ---plot attn---------
            # plot_attn_chunk = []
            # for i,m in enumerate(self.encoder.encoders):
            #     plot_attn_chunk.append(
            #             m.self_attn.attn[0].cpu().detach().numpy()
            #         )
            # self.plot_enc_attn.append(plot_attn_chunk)
            # --------------------
            hs_pad_temp1 = hs_pad_temp
            mem_size = self.hwsize//self.compressive_rate
            if self.conv1d2decoder is not None and not self.usespk_version2:
                hs_pad_temp_mask = torch.ones(hs_pad_temp.size(0),1,hs_pad_temp.size(1)).type(memsh_mask.dtype).to(memsh_mask.device)
                hs_pad_mask.append(torch.cat((memsh_mask[:,:,-self.memspeech_size-mem_size:-mem_size]
                                    ,hs_pad_temp_mask),dim=2))
                hs_pad_temp = torch.cat((last_memsh[:,-self.memspeech_size:,:],hs_pad_temp),dim=1)  # batch, mem+windows_size, adim
                memsph_tmp = self.conv1d2decoder(hs_pad_temp1.transpose(-1,-2)).transpose(-1,-2)
                last_memsh = torch.cat((last_memsh,memsph_tmp),dim=1)
                # analysis
                self.whole_hs = self.whole_hs + (hs_pad_temp1,)
            elif self.alllayer2ctc is None and self.conv1d2decoder is None:
                hs_pad_ctc = hs_pad_ctc + (hs_pad_temp1,)
            if self.usespk2decoder:
                if self.usespk_version2 and self.conv1d2decoder is not None:
                    hs_pad_temp_mask = torch.ones(hs_pad_temp.size(0),1,hs_pad_temp.size(1)).type(memsh_mask.dtype).to(memsh_mask.device)
                    hs_pad_mask.append(torch.cat((memspk_mask[:,:,-self.memspeaker_size:]
                                                ,memsh_mask[:,:,-self.memspeech_size-mem_size:-mem_size]
                                                ,hs_pad_temp_mask),dim=2))
                    hs_pad_temp = torch.cat((memspk[:,-self.memspeaker_size:,:],last_memsh[:,-self.memspeech_size:,:],hs_pad_temp),dim=1)

                    mem_whole = self.conv1d2decoder(hs_pad_temp1.transpose(-1,-2)).transpose(-1,-2)
                    mem_softlin = self.soft_linear(mem_whole).view(mem_whole.size(0),mem_whole.size(1),2,self.adim)
                    mem_softmax = torch.nn.functional.softmax(mem_softlin,dim=2)
                    memspk_tmp = self.spklinear2em(mem_softmax[:,:,0,:]*mem_whole)
                    memsph_tmp = self.shlinear2em(mem_softmax[:,:,1,:]*mem_whole)
                    last_memsh = torch.cat((last_memsh,memsph_tmp),dim=1)
                    memspk = torch.cat((memspk, memspk_tmp) ,dim=1)
                    memspk_mask = torch.cat((memspk_mask
                                    , torch.ones(xs_pad.size(0),1,mem_size).type(memspk_mask.dtype).to(memspk_mask.device)),dim=2)
                else:
                    hs_pad_mask[-1] = torch.cat((memspk_mask[:,:,-self.memspeaker_size:],hs_pad_mask[-1]),dim=2)
                    hs_pad_temp = torch.cat((memspk[:,-self.memspeaker_size:,:],hs_pad_temp),dim=1)

                    memspk_mask = torch.cat((memspk_mask
                                    , torch.ones(xs_pad.size(0),1,mem_size).type(memspk_mask.dtype).to(memspk_mask.device)),dim=2)
                    memspk_tmp = self.speaker2decoder(hs_pad_temp1.transpose(-1,-2)).transpose(-1,-2)
                    # memspk_tmp = torch.mean(memspk_tmp, dim=1).unsqueeze(1)
                    memspk = torch.cat((memspk, memspk_tmp) ,dim=1)
            if self.memattnloss_decoder:
                x_mems.append(self.encoder.encoders[-1].x_mem.detach()) #use detach no grad
                commems.append(torch.cat((memspk_tmp,memsph_tmp),dim=1))

            hs_pad = hs_pad + (hs_pad_temp.unsqueeze(1),)     # batch, 1, windows_size, adim
            mem_loss_total = mem_loss_total + mem_loss if mem_loss_total is not None else mem_loss 
        
        # self.plot_encoders_attn(plot_enc_attn)  # plot attn

        if self.conv1d2decoder is not None:
            self.whole_hs = torch.cat(self.whole_hs,dim=1)
            memsh_final = self.memsh_final = last_memsh[:,self.memspeech_size:,:] #anaylsis
            hs_pad_mask = torch.stack(hs_pad_mask,dim=1)
            hs_len = memsh_mask.view(memsh_final.size(0), -1).sum(1)
        elif self.alllayer2ctc is None :
            # memsh_final = memsh[-1][:,self.memspeech_size:,:]
            # hs_len = memsh_mask.view(memsh_final.size(0), -1).sum(1)
            memsh_final = torch.cat(hs_pad_ctc,dim=1) # batch, all_window, adim
            hs_len = pred_len*self.hwsize
            # memsh_final = memsh[-1][:,self.memspeech_size:,:] # torch.mean(memsh,dim=0)[:,self.memspeech_size:,:] last layer
        else:
            memsh_final = torch.cat(memsh,dim=-1)[:,self.memspeech_size:,:]  #  batch, mem, dim
            memsh_final = self.alllayer2ctc(memsh_final)
            hs_len = memsh_mask.view(memsh_final.size(0), -1).sum(1)
        if self.memattnloss_decoder:
            x_mems = torch.stack(x_mems,dim=1)
            commems = torch.stack(commems,dim=1)

        self.mem_loss_total = mem_loss_total
        hs_pad = torch.cat(hs_pad,dim=1)    # batch, n_chunk, windows_size, adim
        self.hs_pad = hs_pad
        self.memspk = memspk
        # 1.5. transducer preparation related
        ys_in_pad, target, ys_mask , target_len = prepare_loss_inputs(ys_pad) # hs_mask_reshape[:,:,-1]

        # 2. decoder
        if self.dtype == "transformer":

            ys_in_pad = ys_in_pad.unsqueeze(1).expand(-1, n_chunk, -1)  #(batch_size, chunk, tgtsize)
            ys_mask = ys_mask.unsqueeze(1).expand(-1, n_chunk, -1, -1)  #(batch_size, chunk, tgtsize, tgtsize)
            hs_mask = hs_pad_mask if len(hs_pad_mask) else None         # hs_pad_mask for conv1d2decoder
            x_mems = x_mems if len(x_mems) else None
            commems = commems if len(commems) else None
            pred_pad, loss_attndec = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask, x_mems,commems) # None is hs_mask, using pred_len to mask

        # plot dec attn
        # self.plot_dec_attn = []
        # for m in self.decoder.decoders:
        #     self.plot_dec_attn.append(
        #             m.src_attn.attn[0].cpu().detach().numpy()  #(nchunk, head, nenc, ndec)
        #         )

        # self.plot_decoders_attn(self.plot_dec_attn)
        
        self.pred_pad = pred_pad    # (batch_size,nchunk,nseq,tgtsize)
        # pred_pad = torch.log_softmax(pred_pad,dim=2) #log_softmax
        # 3. loss computation

        loss_ctc = self.ctc(memsh_final, hs_len, ys_pad)
        loss = self.criterion(pred_pad, target, pred_len, target_len)
        self.loss = loss  + self.mem_loss_total + loss_ctc #0.3*
        if self.usespk2decoder:
            pred_spk = self.spklinear(memspk[:,self.memspeaker_size:,:])
            real_spk = spkid.unsqueeze(1).expand(-1,pred_spk.size(1)).reshape(-1)
            loss_spk = self.spkloss(pred_spk.reshape(-1,self.spkodim),real_spk)
            self.loss += loss_spk 
            loss_spk_data = float(loss_spk)
        else:
            loss_spk_data = None
        
        if loss_attndec is not None:
            self.loss += loss_attndec
            loss_attndec_data = float(loss_attndec)
        else:
            loss_attndec_data = None
        loss_rnnt_data = float(loss)
        loss_ctc_data = float(loss_ctc)
        loss_mem_data = float(self.mem_loss_total)
        # if(not torch.equal(torch.tensor([ilens[0]]*ilens.size(0)),ilens)):
        #     logging.warning(str(("ilens: ",ilens,"pred_lens: ",pred_len)))
        if math.isnan(float(self.loss)):
            print("outof mem")
            pass
        if not math.isnan(loss_rnnt_data):
            self.reporter.report(float(self.loss), loss_rnnt_data, loss_ctc_data, loss_mem_data,loss_spk_data,loss_attndec_data)
        else:
            logging.warning("loss (=%f) is not correct", loss_rnnt_data)

        return self.loss


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
        # logging.INFO("beam_size: ",recog_args.beam_size)
        with torch.no_grad():
            h = self.encode_transformer(x)  #h, memsh =
            h = h.squeeze(0)
            n_chunk = h.size(0)
            # memsh = torch.stack(memsh,dim=0)
            # memsh = torch.mean(memsh,dim=0)[:,self.memspeech_size:,:]

            recog_args.hwsize = self.hwsize
            recog_args.n_chunk = n_chunk

            params = [h, self.hs_pad_mask, recog_args]
            if recog_args.beam_size == 1 or recog_args.beam_size==0:
                nbest_hyps = self.decoder.recognize(*params)
            else:
                nbest_hyps = self.decoder_recognize(h,recog_args)
                # params.append(rnnlm)
                # nbest_hyps = self.decoder.recognize_beam(*params)

        return nbest_hyps

    def online_recognize_setup(self, beam_size):
        self.src_mask = torch.tensor([True]*(self.hwsize*4+3)).reshape((1,1,-1))
        self.memsh, self.memsh_mask = self.initial_state(1,self.memspeech_size,len(self.encoder.encoders))
        self.memsh_mask = self.memsh_mask.type(self.src_mask.dtype)
        self.last_memsh = None
        if self.conv1d2decoder is not None:
            self.last_memsh = to_device(self, torch.zeros(1,self.memspeech_size,self.adim))
        if beam_size == 1 or beam_size==0:
            hyp = {"score": 0.0, "yseq": [self.blank_id]}
        else:
            hyp = {"score": 0.0, "yseq": torch.tensor([self.blank_id], dtype=torch.long)}
        self.hyps = [hyp]

    def online_recognize_each_chunk(self, x, recog_args):
        self.eval()
        x = to_device(self,torch.as_tensor(x).unsqueeze(0))
        self.hs_pad_temp, _, _, _ = self.encoder(x, self.src_mask ,self.memsh,
                                             self.memsh_mask,is_inference=True,is_compress=False) 
                                              # batch, windows_size,adim   
        if self.conv1d2decoder is not None:
            hs = torch.cat((self.last_memsh[:,-self.memspeech_size:,:],self.hs_pad_temp),dim=1)  # batch, mem+windows_size, adim
            hs_mask_tmp=torch.ones(1,1,self.hwsize).type(self.memsh_mask.dtype).to(self.memsh_mask.device)
            hs_mask = torch.cat((self.memsh_mask[:,:,-self.memspeech_size:],hs_mask_tmp),dim=2)
        else:
            hs = self.hs_pad_temp
            hs_mask = None
        hs = hs.squeeze(0)
        if recog_args.beam_size == 1 or recog_args.beam_size==0:
            self.hyps = self.hyps[0]
            self.hyps = self.decoder.recognize_each_chunk(self.hyps , hs, h_mask=hs_mask)
            self.hyps = [self.hyps]
        else:
            if hs_mask is not None:
                hs_mask = hs_mask.squeeze(0)
            self.hyps =self.decoder_each_chunk_beam_search(self.hyps, hs, h_mask=hs_mask)
        return self.hyps

    def update_commem(self):  #for online
        self.memsh=self.encoder.update_commem(self.memsh)
        cp_size = self.hwsize//self.compressive_rate
        mask_tmp = torch.ones(1,1,cp_size).type(self.memsh_mask.dtype).to(self.memsh_mask.device)
        self.memsh_mask = torch.cat((self.memsh_mask,mask_tmp),dim=2)
        if self.conv1d2decoder is not None:
            mem_tmp = self.conv1d2decoder(self.hs_pad_temp.transpose(-1,-2)).transpose(-1,-2)
            self.last_memsh = torch.cat((self.last_memsh,mem_tmp),dim=1)

    def encode_transformer(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)

        """
        self.eval()

        x = to_device(self,torch.as_tensor(x).unsqueeze(0))
        x_reshape = self.reshapetochunk(x)
        n_chunk = x_reshape.size(1)
        src_mask = make_non_pad_mask([n_chunk]).to(x.device)  # (batch, nchunk)
        src_mask_expand = src_mask.unsqueeze(-1).unsqueeze(-2).expand(-1,-1,-1,self.hwsize*4+3)  # (batch, nchunk, 1,hwsize*4+3)
        memsh, memsh_mask = self.initial_state(x_reshape.size(0),self.memspeech_size,len(self.encoder.encoders))
        memsh_mask = memsh_mask.type(src_mask_expand.dtype)
        
        memspk = to_device(self, torch.zeros(x_reshape.size(0),self.memspeaker_size,self.adim))
        memspk_mask = to_device(self, torch.zeros(x_reshape.size(0),1,self.memspeaker_size)).type(src_mask_expand.dtype)
        if self.conv1d2decoder is not None:
            last_memsh = to_device(self, torch.zeros(x_reshape.size(0),self.memspeech_size,self.adim))
        hs_pad = tuple()
        hs_pad_ctc = tuple()
        hs_pad_mask = []
        for i in range(n_chunk):
            hs_pad_temp, memsh_mask, memsh, _ = self.encoder(x_reshape[:,i,:,:], src_mask_expand[:,i,:,:] ,
                                                            memsh, memsh_mask,is_inference=True)  # batch, windows_size,adim   
            hs_pad_temp1 = hs_pad_temp
            mem_size = self.hwsize//self.compressive_rate
            if self.conv1d2decoder is not None and not self.usespk_version2:
                hs_pad_temp_mask = torch.ones(hs_pad_temp.size(0),1,hs_pad_temp.size(1)).type(memsh_mask.dtype).to(memsh_mask.device)
                hs_pad_mask.append(torch.cat((memsh_mask[:,:,-self.memspeech_size-mem_size:-mem_size]
                                    ,hs_pad_temp_mask),dim=2))
                hs_pad_temp = torch.cat((last_memsh[:,-self.memspeech_size:,:],hs_pad_temp),dim=1)  # batch, mem+windows_size, adim
                memsph_tmp = self.conv1d2decoder(hs_pad_temp1.transpose(-1,-2)).transpose(-1,-2)
                last_memsh = torch.cat((last_memsh,memsph_tmp),dim=1)
            elif self.alllayer2ctc is None and self.conv1d2decoder is None:
                hs_pad_ctc = hs_pad_ctc + (hs_pad_temp1,)
            if self.usespk2decoder:
                if self.usespk_version2 and self.conv1d2decoder is not None:
                    hs_pad_temp_mask = torch.ones(hs_pad_temp.size(0),1,hs_pad_temp.size(1)).type(memsh_mask.dtype).to(memsh_mask.device)
                    hs_pad_mask.append(torch.cat((memspk_mask[:,:,-self.memspeaker_size:]
                                                ,memsh_mask[:,:,-self.memspeech_size-mem_size:-mem_size]
                                                ,hs_pad_temp_mask),dim=2))
                    hs_pad_temp = torch.cat((memspk[:,-self.memspeaker_size:,:],last_memsh[:,-self.memspeech_size:,:],hs_pad_temp),dim=1)
                    
                    mem_whole = self.conv1d2decoder(hs_pad_temp1.transpose(-1,-2)).transpose(-1,-2)
                    mem_softlin = self.soft_linear(mem_whole).view(mem_whole.size(0),mem_whole.size(1),2,self.adim)
                    mem_softmax = torch.nn.functional.softmax(mem_softlin,dim=2)
                    memspk_tmp = self.spklinear2em(mem_softmax[:,:,0,:]*mem_whole)
                    memsph_tmp = self.shlinear2em(mem_softmax[:,:,1,:]*mem_whole)
                    last_memsh = torch.cat((last_memsh,memsph_tmp),dim=1)
                    memspk = torch.cat((memspk, memspk_tmp) ,dim=1)
                    memspk_mask = torch.cat((memspk_mask
                                    , torch.ones(x_reshape.size(0),1,mem_size).type(memspk_mask.dtype).to(memspk_mask.device)),dim=2)
                else:
                    hs_pad_mask[-1] = torch.cat((memspk_mask[:,:,-self.memspeaker_size:],hs_pad_mask[-1]),dim=2)
                    hs_pad_temp = torch.cat((memspk[:,-self.memspeaker_size:,:],hs_pad_temp),dim=1)

                    memspk_mask = torch.cat((memspk_mask
                                    , torch.ones(x_reshape.size(0),1,mem_size).type(memspk_mask.dtype).to(memspk_mask.device)),dim=2)
                    memspk_tmp = self.speaker2decoder(hs_pad_temp1.transpose(-1,-2)).transpose(-1,-2)
                    # memspk_tmp = torch.mean(memspk_tmp, dim=1).unsqueeze(1)
                    memspk = torch.cat((memspk, memspk_tmp) ,dim=1)

            hs_pad = hs_pad + (hs_pad_temp.unsqueeze(1),)     
        
        if self.conv1d2decoder is not None:
            self.memsh_final = last_memsh[:,self.memspeech_size:,:]
            self.hs_pad_mask = torch.stack(hs_pad_mask,dim=1)
        elif self.alllayer2ctc is None:
            
            self.memsh_final = torch.cat(hs_pad_ctc,dim=1) # batch, all_window, adim
            # self.memsh_final = memsh[-1][:,self.memspeech_size:,:]
             # torch.mean(memsh,dim=0)[:,self.memspeech_size:,:] use last layer for now
            self.hs_pad_mask = None
        else:
            memsh_final = torch.cat(memsh,dim=-1)[:,self.memspeech_size:,:]  #  batch, mem, dim
            self.memsh_final = self.alllayer2ctc(memsh_final)
            self.hs_pad_mask = None
        if self.usespk2decoder is not None :    
            self.memspk = memspk[:,self.memspeaker_size:,:]
        hs_pad = torch.cat(hs_pad,dim=1)    # batch, n_chunk, windows_size, adim
        
        return hs_pad #hs_pad,memsh

    def decoder_recognize(self,h,recog_args): 
        # search parms
        beam = recog_args.beam_size
        nbest = recog_args.nbest
        ctc_weight = recog_args.ctc_weight
        if ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(self.memsh_final)
            lpz = lpz.squeeze(0)
        else:
            lpz = None
        
        #initialize hypothesis
        hyp = {"score": 0.0, "yseq": torch.tensor([self.blank_id], dtype=torch.long)}
        if lpz is not None:
            self.ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = self.ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam)) #beam is enough, not beam * CTC_SCORING_RATIO
            else:
                ctc_beam = lpz.shape[-1]
        else:
            ctc_beam = 0
        hyps = [hyp] 
        
        for i,hi in enumerate(h):
            h_mask = self.hs_pad_mask[0,i] if self.hs_pad_mask is not None else None
            hyps = self.decoder_each_chunk_beam_search(hyps, hi, h_mask=h_mask, beam=beam,
                    ctc_beam=ctc_beam,ctc_weight=ctc_weight)
        nbest_hyps = sorted(hyps, key=lambda x: x["score"], reverse=True)[:nbest]
        return nbest_hyps

    def decoder_each_chunk_beam_search(self, hyps, hi, h_mask=None,
                                        beam=5, times=3,ctc_beam=0, ctc_weight=0):
        hyps_yseq = [h["yseq"] for h in hyps]
        hyps_len = [len(h["yseq"]) for h in hyps]
        hyps_score = torch.tensor([h["score"] for h in hyps])
        
        ys = to_device(self, pad_list(hyps_yseq, self.blank_id)).unsqueeze(1) #(batch,1, tgtsize)
        hi = hi.unsqueeze(0).unsqueeze(1).expand(ys.size(0),-1,-1,-1) # (batch,1,nwindow, adim)
        ys_mask = to_device(
            self, subsequent_mask(ys.size(-1)).unsqueeze(0).unsqueeze(0) #(1, 1, tgtsize, tgtsize)
        )
        h_mask = h_mask.unsqueeze(0).unsqueeze(0) if h_mask is not None else None
        scores=self.decoder.forward_one_step_forbatch(ys, ys_mask, hyps_len, hi, h_mask)
        n_tokens = scores.size(1)-1
        hyps_blank_score = hyps_score + scores[:,0]

        expan_blank_score, expan_hyps_yseq = [], []
        if ctc_beam>0:
            expan_ctc_score, expan_ctc_status = [], [] 
            hyps_ctc_score = [h["ctc_score_prev"] for h in hyps]
            hyps_ctc_state = [h["ctc_state_prev"] for h in hyps]

        for ex_i in range(times): # means one chunk generate 2 word at most 
            if ex_i==0:
                if ctc_beam>0:
                    ctc_expan_scores, ctc_expan_ids = torch.topk(scores[:,1:], ctc_beam, dim=1)
                    ctc_scores_list, ctc_states_list = [], []
                    ctc_local_scores_list = []
                    n_size = ctc_expan_scores.size(0)
                    for k in range(n_size):
                        ctc_scores, ctc_states = self.ctc_prefix_score(
                            hyps_yseq[k], ctc_expan_ids[k]+1, hyps_ctc_state[k]
                        )
                        ctc_scores_list.append(ctc_scores)
                        ctc_states_list.append(ctc_states)

                        local_scores = hyps_score[k] + (1.0 - ctc_weight) * ctc_expan_scores[k]  + ctc_weight * torch.from_numpy(ctc_scores - hyps_ctc_score[k])
                        ctc_local_scores_list.append(local_scores)
                    expan_scores, expan_ids = torch.topk(torch.cat(ctc_local_scores_list), beam)
                    token_ids = ctc_expan_ids.view(-1)[expan_ids]

                    # Expansion 
                    expan_hyps_yseq.append([torch.cat((hyps_yseq[expan_ids[i]//ctc_beam],
                                                hyps_yseq[0].new([token_ids[i]+1]))) 
                                                for i in range(beam)])
                    expan_ctc_score.append(numpy.concatenate(ctc_scores_list)[expan_ids])
                    expan_ctc_status.append(numpy.concatenate(ctc_states_list,axis=0)[expan_ids,:,:])
                else:
                    score_expan = scores[:,1:].contiguous().view(-1)
                    hyps_score_expan = hyps_score.unsqueeze(1).expand(-1,n_tokens).contiguous().view(-1) \
                                        + score_expan
                    expan_scores, expan_ids = torch.topk(hyps_score_expan, beam)
                    # Expansion 
                    expan_hyps_yseq.append([torch.cat((hyps_yseq[expan_ids[i]//n_tokens],
                                                hyps_yseq[0].new([expan_ids[i]%n_tokens+1]))) 
                                                for i in range(beam)])
            else:
                if ctc_beam>0:
                    ctc_expan_scores, ctc_expan_ids = torch.topk(scores[:,1:], ctc_beam, dim=1)
                    ctc_scores_list, ctc_states_list = [], []
                    ctc_local_scores_list = []
                    n_size = ctc_expan_scores.size(0)
                    for k in range(n_size):
                        ctc_scores, ctc_states = self.ctc_prefix_score(
                            expan_hyps_yseq[ex_i-1][k], ctc_expan_ids[k]+1, expan_ctc_status[ex_i-1][k,:,:]
                        )
                        ctc_scores_list.append(ctc_scores)
                        ctc_states_list.append(ctc_states)

                        local_scores = expan_scores[k] + (1.0 - ctc_weight) * ctc_expan_scores[k]  + ctc_weight * torch.from_numpy(
                            ctc_scores - expan_ctc_score[ex_i-1][k]
                        )
                        ctc_local_scores_list.append(local_scores)
                    expan_scores, expan_ids = torch.topk(torch.cat(ctc_local_scores_list), beam)
                    token_ids = ctc_expan_ids.view(-1)[expan_ids]
                    
                    expan_hyps_yseq.append([torch.cat((expan_hyps_yseq[ex_i-1][expan_ids[i]//ctc_beam],
                                                hyps_yseq[0].new([token_ids[i]+1]))) 
                                                for i in range(beam)])

                    expan_ctc_score.append(numpy.concatenate(ctc_scores_list)[expan_ids])
                    expan_ctc_status.append(numpy.concatenate(ctc_states_list,axis=0)[expan_ids,:,:])
                else:
                    score_expan = scores[:,1:].contiguous().view(-1)
                    hyps_score_expan = expan_scores.unsqueeze(1).expand(-1,n_tokens).contiguous().view(-1) \
                                        + score_expan
                    expan_scores, expan_ids = torch.topk(hyps_score_expan, beam)
                    expan_hyps_yseq.append([torch.cat((expan_hyps_yseq[ex_i-1][expan_ids[i]//n_tokens],
                                                hyps_yseq[0].new([expan_ids[i]%n_tokens+1]))) 
                                                for i in range(beam)])
            # whatever ctc or normal, just produce the 'beam size' output      
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
                if ctc_beam>0:
                    hyp["ctc_state_prev"] = hyps_ctc_state[ids]
                    hyp["ctc_score_prev"] = hyps_ctc_score[ids]
            else:
                ids = ids - n_size
                hyp = {"score": expan_blank_score[ids//beam][ids % beam],
                        "yseq": expan_hyps_yseq[ids//beam][ids % beam]}
                if ctc_beam>0:
                    hyp["ctc_state_prev"] = expan_ctc_status[ids//beam][ids % beam]
                    hyp["ctc_score_prev"] = expan_ctc_score[ids//beam][ids % beam]
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

    def plot_predict(self, xs_pad, ilens, ys_pad,token=None,name='None'):
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
        if token is not None:
            token.insert(0,'<blk>')
        final = np.zeros_like(output_np[0])
        for i in range(len(output_np)):
            plt.imshow(output_np[i].T.astype(np.float32), aspect="auto")
            plt.title("chunk %d" % i)
            plt.xlabel("input")
            plt.ylabel("ouput")
            if token is not None:
                plt.xticks(np.arange(0,len(token)),token)
                plt.yticks(np.arange(0,len(token)),token)

            plt.savefig("%s/a%02d.png" %(link,i))
            final = final + output_np[i].T.astype(np.float32)


        plt.imshow(final, aspect="auto")
        plt.title("Sum of chunks")
        plt.xlabel("input")
        plt.ylabel("ouput")
        if token is not None:
            plt.xticks(np.arange(0,len(token)),token)
            plt.yticks(np.arange(0,len(token)),token)
        plt.savefig("%s/final.png" %(link))

    def plot_encoders_attn(self, plot_attn):
        link ="photo/encoder_attn/"
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn import preprocessing
        from scipy.special import softmax
        if not os.path.exists(link):
            os.mkdir(link)
        attn_score = np.zeros_like(plot_attn[0][0][0])
        for i in range(len(plot_attn)):
            for j in range(len(plot_attn[i])):
                plt.figure(1,(25,10))
                for k in range(8):
                    attn_score+=plot_attn[i][j][k,:,:].astype(np.float32)
                #     plt.subplot(2,4,k+1)
                #     plt.imshow(plot_attn[i][j][k,:,:].astype(np.float32), aspect="auto")
                #     plt.title("att chunk %d layer %d" % (i,j))
                #     plt.xlabel("input mem/seq(x)")
                #     plt.ylabel("ouput seq(x)")
                #     plt.xticks(np.arange(0,self.hwsize+self.memspeech_size),np.concatenate((np.arange(0,self.memspeech_size),np.arange(0,self.hwsize)), axis=0)) #
                #     plt.yticks(np.arange(0,self.hwsize))
                # plt.savefig("%s/nchunk_%02d_layer_%d.png" %(link,i,j))
        score = np.sum(attn_score,axis=0)
        min_max_scaler = preprocessing.MinMaxScaler()
        score = min_max_scaler.fit_transform(score.reshape(-1,1)).reshape(-1)
        # score = softmax(score)
        plt.figure(2,(10,10))
        plt.bar(np.arange(0,score.shape[0]),score)
        plt.xticks(np.arange(0,self.hwsize+self.memspeech_size),np.concatenate((np.arange(0,self.memspeech_size),np.arange(0,self.hwsize)), axis=0)) #
        plt.xlabel("input mem/seq(x)")
        plt.ylabel("attn weight")
        plt.savefig("%s/attn_score.png" %(link))

    def plot_decoders_attn(self, plot_attn):    #plot_attn[layer]->(nchunk, head, ndec, nenc)
        link = "photo/decoder_attn/"
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn import preprocessing
        from scipy.special import softmax
        if not os.path.exists(link):
            os.mkdir(link)
        attn_score = np.zeros_like(plot_attn[0][0,0])
        for i in range(len(plot_attn)):
            for j in range(plot_attn[0].shape[0]):
                plt.figure(1,(25,10))
                for k in range(8):
                    attn_score+=plot_attn[i][j,k,:,:].astype(np.float32)
                    plt.subplot(2,4,k+1)
                    plt.imshow(plot_attn[i][j,k,:,:].astype(np.float32), aspect="auto")
                    plt.title("att chunk %d layer %d" % (j,i))
                    plt.xlabel("encoder hidden state(x)")
                    plt.ylabel("decoder(x)")
                    plt.xticks(np.arange(0,self.hwsize+self.memspeech_size),np.concatenate((np.arange(0,self.memspeech_size),np.arange(0,self.hwsize)), axis=0)) #
                    plt.yticks(np.arange(0,plot_attn[0].shape[2]))
                plt.savefig("%s/nchunk_%02d_layer_%d.png" %(link,i,j))
        
        score = np.sum(attn_score,axis=0)
        min_max_scaler = preprocessing.MinMaxScaler()
        score = min_max_scaler.fit_transform(score.reshape(-1,1)).reshape(-1)
        # score = softmax(score)
        plt.figure(2,(10,10))
        plt.bar(np.arange(0,score.shape[0]),score)
        plt.xticks(np.arange(0,self.hwsize+self.memspeech_size),np.concatenate((np.arange(0,self.memspeech_size),np.arange(0,self.hwsize)), axis=0)) #
        plt.xlabel("input mem/seq(x)")
        plt.ylabel("attn weight")
        plt.savefig("%s/attn_score.png" %(link))




