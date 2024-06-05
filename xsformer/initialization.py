# coding: utf-8
# 修改过
"""
Implements custom initialization
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out


def xavier_uniform_n_(w: Tensor, gain: float = 1.0, n: int = 4) -> None:
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.

    :param w: parameter
    :param gain: default 1
    :param n: default 4
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out //= n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


# pylint: disable=too-many-branches
def initialize_model(model) -> None:
    """
    This initializes a model based on the provided config.

    All initializer configuration is part of the `model` section of the
    configuration file.
    For an example, see e.g. `https://github.com/joeynmt/joeynmt/
    blob/master/configs/iwslt_envi_xnmt.yaml#L47`

    The main initializer is set using the `initializer` key.
    Possible values are `xavier`, `uniform`, `normal` or `zeros`.
    (`xavier` is the default).

    When an initializer is set to `uniform`, then `init_weight` sets the
    range for the values (-init_weight, init_weight).

    When an initializer is set to `normal`, then `init_weight` sets the
    standard deviation for the weights (with mean 0).

    The word embedding initializer is set using `embed_initializer` and takes
    the same values. The default is `normal` with `embed_init_weight = 0.01`.

    Biases are initialized separately using `bias_initializer`.
    The default is `zeros`, but you can use the same initializers as
    the main initializer.

    Set `init_rnn_orthogonal` to True if you want RNN orthogonal initialization
    (for recurrent matrices). Default is False.

    `lstm_forget_gate` controls how the LSTM forget gate is initialized.
    Default is `1`.

    :param model: model to initialize
    :param cfg: the model configuration
    :param txt_padding_idx: index of spoken language text padding token
    """

    # defaults: xavier, embeddings: normal 0.01, biases: zeros, no orthogonal
    gain = 1.0  # for xavier
    init = "xavier"
    init_weight = 0.01

    # embed_init = "xavier"
    # embed_init_weight = 0.01
    # embed_gain = 1.0  # for xavier

    bias_init = "zeros"
    bias_init_weight = 0.01

    # pylint: disable=unnecessary-lambda, no-else-return
    def _parse_init(s, scale, _gain):
        scale = float(scale)
        assert scale > 0.0, "incorrect init_weight"
        if s.lower() == "xavier":
            return lambda p: nn.init.xavier_uniform_(p, gain=_gain)
        elif s.lower() == "uniform":
            return lambda p: nn.init.uniform_(p, a=-scale, b=scale)
        elif s.lower() == "normal":
            return lambda p: nn.init.normal_(p, mean=0.0, std=scale)
        elif s.lower() == "zeros":
            return lambda p: nn.init.zeros_(p)
        else:
            raise ValueError("unknown initializer")

    init_fn_ = _parse_init(init, init_weight, gain)
    # embed_init_fn_ = _parse_init(embed_init, embed_init_weight, embed_gain)
    bias_init_fn_ = _parse_init(bias_init, bias_init_weight, gain)

    with torch.no_grad():
        for name, p in model.named_parameters():

            # if "lut" in name:  # for translation
            #     embed_init_fn_(p)

            if "bias" in name:
                print('zero init: {}'.format(name))
                bias_init_fn_(p)

            elif len(p.size()) > 1:
                print('xavier init: {}'.format(name))
                init_fn_(p)

        # zero out paddings
        # if model.txt_embed is not None:  # for translation
        #     model.lut.weight.data[0].zero_()
