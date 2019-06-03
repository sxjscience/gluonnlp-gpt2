from mxnet.gluon import Block
from mxnet.gluon import nn

class GPT2Model(Block):
    def __init__(self, units, embed_dim, context_size, num_layers, num_heads, prefix=None, params=None):
        """

        Parameters
        ----------
        units : int
        embed_dim: int
        context_size : int
        num_layers : int
        num_heads: int
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
        self._context_size = context_size
        self._num_layers = num_layers
        self._num_heads = num_heads
        pass


