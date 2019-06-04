import mxnet as mx
import numpy as np
from mxnet.gluon import Block, HybridBlock
from mxnet.gluon import nn
from gluonnlp.attention_cell import DotProductAttentionCell, MultiHeadAttentionCell, _masked_softmax


class GPT2SelfAttentionLayer(HybridBlock):
    def __init__(self, units, num_heads, dropout=0.0,
                 weight_initializer=mx.init.Normal(0.02), bias_initializer='zeros', prefix=None, params=None):
        """

        Parameters
        ----------
        units : int
        num_heads : int
        dropout : float
        prefix : str, default None
            Prefix for name of `Block`s
            (and name of weight if params is `None`).
        params : Parameter or None, default None
            Container for weight sharing between cells.
            Created if `None`.
        """
        super(GPT2SelfAttentionLayer, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        assert units % num_heads == 0
        with self.name_scope():
            self._multi_head_qkv_proj = nn.Dense(units=units * 3, flatten=False, use_bias=True,
                                                 weight_initializer=weight_initializer,
                                                 bias_initializer=bias_initializer)
            self._base_attn_cell = DotProductAttentionCell(scaled=True, dropout=dropout)
            self._dropout_layer = nn.Dropout(dropout)
            self._out_proj = nn.Dense(units=units, flatten=False, use_bias=True,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer)

    def forward(self, data, states, mask=None):
        batch_size = data.shape[0]
        seq_length = data.shape[1]
        prev_key, _ = states
        prev_seq_length = prev_key.shape[1]
        data_pos = mx.nd.arange(prev_seq_length, prev_seq_length + seq_length, ctx=data.context, dtype=data.dtype)
        all_pos = mx.nd.arange(seq_length + prev_seq_length, ctx=data.context, dtype=data.dtype)
        if mask is None:
            mask = mx.nd.broadcast_lesser_equal(all_pos.reshape((1, -1)), data_pos.reshape((-1, 1)))
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0), axis=0, size=batch_size)
        return super(GPT2SelfAttentionLayer, self).forward(data, states, mask)


    def hybrid_forward(self, F, data, states, mask):
        """

        Parameters
        ----------
        F
        data : mx.nd.NDarray or mx.sym.Symbol
            The input data, should have shape (batch_size, seq_len, in_dim)
        states : list of NDArray or list of Symbol
            The states, contains the previous encoded key/values
            prev_key (batch_size * num_heads, ele_units, past_seq_len),
            prev_value (batch_size * num_heads, ele_units, past_seq_len)
        mask : NDArray or Symbol
            (batch_size, seq_len, past_seq_len + seq_len)
        Returns
        -------
        out : mx.nd.NDArray or mx.sym.Symbol
        new_states : list
        """
        prev_key, prev_value = states
        qkv = self._multi_head_qkv_proj(data)  # Shape (batch_size, seq_len, 3 * units)
        qkv = F.swapaxes(qkv, 1, 2)  # Shape (batch_size, 3 * units, seq_len)
        query, key, value = F.split(qkv, num_outputs=3, axis=1) # Each has shape (batch_size, units, seq_len)
        # Map each to have shape (batch_size * num_head, ele_units, seq_len)
        query = query.reshape(shape=(0, -4, self._num_heads, -1, 0)).reshape(shape=(-1, 0, 0), reverse=True)
        key = key.reshape(shape=(0, -4, self._num_heads, -1, 0)).reshape(shape=(-1, 0, 0), reverse=True)
        value = value.reshape(shape=(0, -4, self._num_heads, -1, 0)).reshape(shape=(-1, 0, 0), reverse=True)
        query = F.contrib.div_sqrt_dim(F.swapaxes(query, 1, 2))  # Shape(batch_size * num_heads, seq_len, ele_units)
        key = F.concat(prev_key, key, dim=2)   # Shape (batch_size * num_heads, all_len, ele_units)
        value = F.concat(prev_value, value, dim=2)
        att_score = F.batch_dot(query, key, transpose_b=True)  # Shape(batch_size * num_heads, seq_len, all_len)
        att_weights = self._dropout_layer(_masked_softmax(F, att_score, mask, np.float32))
        multi_head_out = F.batch_dot(att_weights, value)  # Shape(batch_size * num_heads, seq_len, ele_units)
        multi_head_out = multi_head_out.reshape((-1, self._num_heads, 0, 0), reverse=True)
        multi_head_out = F.transpose(multi_head_out, axes=(0, 2, 1, 3)).reshape((0, 0, -1))
        out = self._out_proj(multi_head_out)
        return out, [key, value]


class GPT2FFNLayer(HybridBlock):
    def __init__(self, units, hidden_size, dropout=0.0,
                 weight_initializer=mx.init.Normal(0.02), bias_initializer='zeros', prefix=None, params=None):
        super(GPT2FFNLayer, self).__init__(prefix=prefix, params=params)
        


class GPT2Model(Block):
    def __init__(self, units, embed_dim, vocab_size, max_seq_len, num_layers, num_heads, dropout=0.0,
                 prefix=None, params=None):
        """

        Parameters
        ----------
        units : int
        embed_dim: int
        vocab_size : int
        max_seq_len : int
            The maximum sequence length
        num_layers : int
        num_heads: int
        dropout : float
        prefix : str, default None
            Prefix for name of `Block`s
            (and name of weight if params is `None`).
        params : Parameter or None, default None
            Container for weight sharing between cells.
            Created if `None`.
        """
        super(GPT2Model, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._embed_dim = embed_dim
        self._max_seq_len = max_seq_len
        self._num_layers = num_layers
        self._num_heads = num_heads
        with self.name_scope():
            self._pos_embed = nn.Embedding(input_dim=max_seq_len, output_dim=embed_dim, name='pos_embed_')
            self._embed = nn.Embedding(input_dim=vocab_size, output_dim=embed_dim, name='embed_')
            self._self_attention_layers = nn.HybridSequential()
            self._
            for i in range(num_layers):
                self._self_attention_layers.add(GPT2SelfAttentionLayer(units=units, num_heads=num_heads,
                                                                       dropout=dropout,
                                                                       prefix='self_attn_l{}_'.format(i)))
