"""Decoder definition for transformer-transducer models."""

import six
import torch

from espnet.nets.pytorch_backend.nets_utils import to_device

from espnet.nets.pytorch_backend.sync_transducer.transformer_decoder_layer import (
    DecoderLayer,  # noqa: H301
)

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
    MultiHeadedAttention4dim, #(batch, chunk, time1, size)
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    PositionalEncodingChunk
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
    PositionwiseFeedForwardGLU,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class Decoder(torch.nn.Module):
    """Decoder module for transformer-transducer models.

    Args:
        odim (int): dimension of outputs
        jdim (int): dimension of joint-space
        attention_dim (int): dimension of attention
        attention_heads (int): number of heads in multi-head attention
        linear_units (int): number of units in position-wise feed forward
        num_blocks (int): number of decoder blocks
        dropout_rate (float): dropout rate for decoder
        positional_dropout_rate (float): dropout rate for positional encoding
        attention_dropout_rate (float): dropout rate for attention
        input_layer (str or torch.nn.Module): input layer type
        padding_idx (int): padding value for embedding
        pos_enc_class (class): PositionalEncoding or ScaledPositionalEncoding
        blank (int): blank symbol ID

    """

    def __init__(
        self,
        odim,
        # jdim,
        attention_dim=512,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncodingChunk,
        normalize_before=True,
        concat_after=False,
        feedforward_GLU=False,
        blank=0,
    ):
        """Construct a Decoder object for transformer-transducer models."""
        torch.nn.Module.__init__(self)

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        
        self.normalize_before=normalize_before

        if feedforward_GLU:
            positionwise_layer = PositionwiseFeedForwardGLU
        else:
            positionwise_layer = PositionwiseFeedForward

        self.decoders = repeat(
            num_blocks,
            lambda: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, attention_dropout_rate
                ),
                positionwise_layer(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

        self.attention_dim = attention_dim
        self.odim = odim

        self.blank = blank

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward transformer decoder.

        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
                                if input_layer == "embed"
                                input tensor
                                (batch, maxlen_out, #mels) in the other cases
            tgt_mask (torch.Tensor): input token mask,  (batch, maxlen_out)
                                     dtype=torch.uint8 in PyTorch 1.2-
                                     dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory (torch.Tensor): encoded memory, float32  (batch, maxlen_in, feat)

        Return:
            z (torch.Tensor): joint output (batch, maxlen_in, maxlen_out, odim)
            tgt_mask (torch.Tensor): score mask before softmax (batch, maxlen_out)

        """
        tgt = self.embed(tgt)

        tgt, tgt_mask, memory, memory_mask = self.decoders(
            tgt, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            tgt = self.after_norm(tgt)
        if self.output_layer is not None:
            tgt = self.output_layer(tgt)

        return tgt, tgt_mask


    def forward_one_step(self, tgt, tgt_mask,memory, cache=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
                                if input_layer == "embed"
                                input tensor (batch, maxlen_out, #mels)
                                in the other cases
            tgt_mask (torch.Tensor): input token mask,  (batch, Tmax)
                                     dtype=torch.uint8 in PyTorch 1.2-
                                     dtype=torch.bool in PyTorch 1.2+ (include 1.2)

        """
        tgt = self.embed(tgt)

        # if cache is None:
        #     cache = self.init_state()
        new_cache = []

        for decoder in self.decoders:
            tgt, tgt_mask, memory, _ = decoder(tgt, tgt_mask, memory, None, None) #memory_mask will error, use None
            # new_cache.append(tgt)

        tgt = tgt.squeeze(0)  #(batch, nchunk, nwords,adim)->(nchunk, nwords,adim)
        tgt = self.after_norm(tgt[:, -1])

        return tgt, new_cache

    def init_state(self, x=None):
        """Get an initial state for decoding."""
        return [None for i in range(len(self.decoders))]

    def recognize(self, h, recog_args):
        """Greedy search implementation for transformer-transducer.

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options

        Returns:
            hyp (list of dicts): 1-best decoding results

        """
        hyp = {"score": 0.0, "yseq": [self.blank]}

        # ys = to_device(self, torch.tensor(hyp["yseq"], dtype=torch.long)).unsqueeze(0).unsqueeze(0) #(1, 1, tgtsize)
        # ys_mask = to_device(self, subsequent_mask(1).unsqueeze(0).unsqueeze(1)) #(1, 1, tgtsize, tgtsize)
        # y, c = self.forward_one_step(ys, ys_mask, h[0,:,:].unsqueeze(0), None)
        for i,hi in enumerate(h):
            hyp = self.recognize_each_chunk(hyp, h[i], None)
        return [hyp]

    def recognize_each_chunk(self, hyp, hi, h_mask=None, n_times=3):
        times=0
        while(times<n_times):
            ys = to_device(self, torch.tensor(hyp["yseq"], dtype=torch.long)).unsqueeze(0).unsqueeze(1) #(1, 1, tgtsize)
            ys_mask = to_device(
                self, subsequent_mask(len(hyp["yseq"])).unsqueeze(0).unsqueeze(1) #(1, 1, tgtsize, tgtsize)
            )
            y, _ = self.forward_one_step(ys, ys_mask, hi.unsqueeze(0).unsqueeze(1), h_mask)

            ytu = torch.log_softmax(self.output_layer(y[0]), dim=0) #self.joint(hi, y[0])
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank and times<n_times:
                hyp["yseq"].append(int(pred))
                hyp["score"] += float(logp)
                times+=1
                # c = new_c
            else:
                return hyp
        return hyp
        
    def recognize_beam(self, h, recog_args, rnnlm=None):
        """Beam search implementation for transformer-transducer.

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language model module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        k_range = min(beam, self.odim)
        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        if rnnlm:
            kept_hyps = [
                {"score": 0.0, "yseq": [self.blank], "lm_state": None} # "cache": None,
            ]
        else:
            kept_hyps = [{"score": 0.0, "yseq": [self.blank]}] #"cache": None, <-- cancel it

        for i, hi in enumerate(h):
            hyps = kept_hyps
            kept_hyps = []
            j=0
            while True:
                new_hyp = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyp)

                ys = to_device(self, torch.tensor(new_hyp["yseq"]).unsqueeze(0).unsqueeze(1))#(1, 1, tgtsize)
                ys_mask = to_device(
                    self, subsequent_mask(len(new_hyp["yseq"])).unsqueeze(0).unsqueeze(1) #(1, 1, tgtsize, tgtsize)
                )
                y, _ = self.forward_one_step(ys, ys_mask, h[i,:,:].unsqueeze(0), None) #new_hyp["cache"]
                ytu = torch.log_softmax(self.output_layer(y[0]), dim=0) #self.joint(hi, y[0]) , dim=0
                # logp, pred = torch.max(ytu, dim=1)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(
                        new_hyp["lm_state"], ys.squeeze(0)[:, -1]
                    )
                for k in six.moves.range(self.odim):
                    beam_hyp = {
                        "score": new_hyp["score"] + float(ytu[k]),
                        "yseq": new_hyp["yseq"][:],
                        # "cache": new_hyp["cache"],
                    }

                    if rnnlm:
                        beam_hyp["lm_state"] = new_hyp["lm_state"]

                    if k == self.blank:
                            kept_hyps.append(beam_hyp)
                            if len(kept_hyps) >= k_range:
                                break
                    else:
                        beam_hyp["yseq"].append(int(k))
                        # beam_hyp["cache"] = c

                        if rnnlm:
                            beam_hyp["lm_state"] = rnnlm_state
                            beam_hyp["score"] += (
                                recog_args.lm_weight * rnnlm_scores[0][k]
                            )   

                        hyps.append(beam_hyp)
                if len(kept_hyps) >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] /(len(x["yseq"])), reverse=True
            )[:nbest]
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True)[
                :nbest
            ]

        return nbest_hyps



    def forward_one_step_forbatch(self, tgt_f, tgt_mask, hyps_len, memory, memory_mask=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
                                if input_layer == "embed"
                                input tensor (batch, maxlen_out, #mels)
                                in the other cases
            tgt_mask (torch.Tensor): input token mask,  (batch, Tmax)
                                     dtype=torch.uint8 in PyTorch 1.2-
                                     dtype=torch.bool in PyTorch 1.2+ (include 1.2)

        """
        tgt = self.embed(tgt_f)


        for decoder in self.decoders:
            tgt, tgt_mask, memory, memory_mask = decoder(tgt, tgt_mask, memory, memory_mask, None) 

        tgt = tgt.squeeze(1)  #(batch, 1, nwords,adim)->(batch, nwords,adim)
        tgt = [self.after_norm(tgt[i,l-1,:])for i,l in enumerate(hyps_len)]
        tgt = torch.stack(tgt,dim=0)
        tgt = torch.log_softmax(self.output_layer(tgt),dim=-1)

        return tgt