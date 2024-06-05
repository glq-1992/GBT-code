import torch
import torch.nn as nn
from torch import Tensor
from transformer_layers import TransformerEncoderLayer, PositionalEncoding, MainTransformerEncoderLayer, \
    CoTransformerEncoderLayer
# pylint: disable=abstract-method


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


class Encoder(nn.Module):
    """
    Base encoder class
    """

    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = self.pe(embed_src)  # add position encoding to word embeddings
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].src_src_att.num_heads,
        )


class DoubleTransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(DoubleTransformerEncoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers - 1)  # 去掉最后一层改成CVA
            ]
        )

        self.co_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers - 1)  # 去掉最后一层改成CVA
            ]
        )

        # 是否需要存疑
        self.co_pe = PositionalEncoding(hidden_size)
        self.co_emb_dropout = nn.Dropout(p=emb_dropout)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size

        # 增加CVA相关层
        self.cva_main = MainTransformerEncoderLayer(
                            size=hidden_size,
                            ff_size=ff_size,
                            num_heads=num_heads,
                            dropout=dropout,
                        )
        self.cva_co = CoTransformerEncoderLayer(
                          size=hidden_size,
                          ff_size=ff_size,
                          num_heads=num_heads,
                          dropout=dropout,
                      )

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor, co_src: Tensor, co_mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param co_mask:
        :param co_src:
        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = self.pe(embed_src)  # add position encoding to word embeddings
        x = self.emb_dropout(x)
        for layer in self.layers:
            x = layer(x, mask)

        # 副view
        co_x = self.co_pe(co_src)  # add position encoding to word embeddings
        co_x = self.co_emb_dropout(co_x)
        for layer in self.co_layers:
            co_x = layer(co_x, co_mask)
        co_x, co_k, co_v = self.cva_co(co_x, co_mask)

        x = self.cva_main(x, mask, co_k, co_v)

        return self.layer_norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].src_src_att.num_heads,
        )


class TripleTransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TripleTransformerEncoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers - 1)  # 去掉最后一层(main layer)
            ]
        )

        self.co_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers - 2)  # 中间层，去掉最后两层，一层为main layer, 一层co layer
            ]
        )

        self.co2_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers - 2)  # 最小层，去掉最后两层(少一层，最后一层为co layer
            ]
        )

        # 是否需要存疑
        self.co_pe = PositionalEncoding(hidden_size)
        self.co_emb_dropout = nn.Dropout(p=emb_dropout)
        self.co2_pe = PositionalEncoding(hidden_size)
        self.co2_emb_dropout = nn.Dropout(p=emb_dropout)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size

        # 增加CVA相关层
        self.cva_main = MainTransformerEncoderLayer(
                            size=hidden_size,
                            ff_size=ff_size,
                            num_heads=num_heads,
                            dropout=dropout,
                        )
        self.cva_co_main = MainTransformerEncoderLayer(
                          size=hidden_size,
                          ff_size=ff_size,
                          num_heads=num_heads,
                          dropout=dropout,
                      )
        self.cva_co_co = CoTransformerEncoderLayer(
            size=hidden_size,
            ff_size=ff_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cva_co2_co = CoTransformerEncoderLayer(
            size=hidden_size,
            ff_size=ff_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor, co_src: Tensor, co_mask: Tensor, co2_src: Tensor,
            co2_mask: Tensor,
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param co2_mask:
        :param co2_src:
        :param co_mask:
        :param co_src:
        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = self.pe(embed_src)  # add position encoding to word embeddings
        x = self.emb_dropout(x)
        for layer in self.layers:
            x = layer(x, mask)

        # 副1至倒数第三层
        co_x = self.co_pe(co_src)  # add position encoding to word embeddings
        co_x = self.co_emb_dropout(co_x)
        for layer in self.co_layers:
            co_x = layer(co_x, co_mask)

        # 副2的k,v -> 副1
        co2_x = self.co2_pe(co2_src)  # add position encoding to word embeddings
        co2_x = self.co2_emb_dropout(co2_x)
        for layer in self.co2_layers:
            co2_x = layer(co2_x, co2_mask)
        co2_x, co2_k, co2_v = self.cva_co2_co(co2_x, co2_mask)
        # 副1 倒数第二层
        co_x = self.cva_co_main(co_x, co_mask, co2_k, co2_v)
        # 副1 最后一层
        co_x, co_k, co_v = self.cva_co_co(co_x, co_mask)
        # 主 最后一层
        x = self.cva_main(x, mask, co_k, co_v)

        return self.layer_norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].src_src_att.num_heads,
        )


class TripleTransformerEncoder2(Encoder):  # 两个副直接给主
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        **kwargs
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TripleTransformerEncoder2, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers - 2)  # 去掉最后两层(main layer)
            ]
        )

        self.co_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers - 1)  # 副1，在倒一层融合
            ]
        )

        self.co2_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers - 2)  # 副2，在倒二层融合
            ]
        )

        # 是否需要存疑
        self.co_pe = PositionalEncoding(hidden_size)
        self.co_emb_dropout = nn.Dropout(p=emb_dropout)
        self.co2_pe = PositionalEncoding(hidden_size)
        self.co2_emb_dropout = nn.Dropout(p=emb_dropout)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size

        # 增加CVA相关层
        self.cva_main = MainTransformerEncoderLayer(
                            size=hidden_size,
                            ff_size=ff_size,
                            num_heads=num_heads,
                            dropout=dropout,
                        )
        self.cva_main2 = MainTransformerEncoderLayer(
                          size=hidden_size,
                          ff_size=ff_size,
                          num_heads=num_heads,
                          dropout=dropout,
                      )
        self.cva_co = CoTransformerEncoderLayer(
            size=hidden_size,
            ff_size=ff_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cva_co2 = CoTransformerEncoderLayer(
            size=hidden_size,
            ff_size=ff_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor, co_src: Tensor, co_mask: Tensor, co2_src: Tensor,
            co2_mask: Tensor,
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param co2_mask:
        :param co2_src:
        :param co_mask:
        :param co_src:
        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        # 主至倒数第三层
        x = self.pe(embed_src)  # add position encoding to word embeddings
        x = self.emb_dropout(x)
        for layer in self.layers:
            x = layer(x, mask)

        # 副2至倒数第三层
        co2_x = self.co2_pe(co2_src)  # add position encoding to word embeddings
        co2_x = self.co2_emb_dropout(co2_x)
        for layer in self.co2_layers:
            co2_x = layer(co2_x, co2_mask)

        # 倒数第二层，副2的k,v -> 主
        co2_x, co2_k, co2_v = self.cva_co2(co2_x, co2_mask)
        x = self.cva_main(x, mask, co2_k, co2_v)

        # 副1至倒数第二层
        co_x = self.co_pe(co_src)  # add position encoding to word embeddings
        co_x = self.co_emb_dropout(co_x)
        for layer in self.co_layers:
            co_x = layer(co_x, co_mask)

        # 最后一层 副1的k,v -> 主
        co_x, co_k, co_v = self.cva_co(co_x, co_mask)
        x = self.cva_main2(x, mask, co_k, co_v)

        return self.layer_norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.layers),
            self.layers[0].src_src_att.num_heads,
        )