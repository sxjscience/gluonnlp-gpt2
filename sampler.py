import mxnet as mx
import numpy as np
import gluonnlp as nlp
from model import load_pretrained_GPT2

def _expand_to_beam_size(data, beam_size, batch_size, state_info=None):
    """Tile all the states to have batch_size * beam_size on the batch axis.

    Parameters
    ----------
    data : A single NDArray/Symbol or nested container with NDArrays/Symbol
        Each NDArray/Symbol should have shape (N, ...) when state_info is None,
        or same as the layout in state_info when it's not None.
    beam_size : int
        Beam size
    batch_size : int
        Batch size
    state_info : Nested structure of dictionary, default None.
        Descriptors for states, usually from decoder's ``state_info()``.
        When None, this method assumes that the batch axis is the first dimension.
    Returns
    -------
    new_states : Object that contains NDArrays/Symbols
        Each NDArray/Symbol should have shape batch_size * beam_size on the batch axis.
    """
    assert not state_info or isinstance(state_info, (type(data), dict)), \
            'data and state_info doesn\'t match, ' \
            'got: {} vs {}.'.format(type(state_info), type(data))
    if isinstance(data, list):
        if not state_info:
            state_info = [None] * len(data)
        return [_expand_to_beam_size(d, beam_size, batch_size, s)
                for d, s in zip(data, state_info)]
    elif isinstance(data, tuple):
        if not state_info:
            state_info = [None] * len(data)
            state_info = tuple(state_info)
        return tuple(_expand_to_beam_size(d, beam_size, batch_size, s)
                     for d, s in zip(data, state_info))
    elif isinstance(data, dict):
        if not state_info:
            state_info = {k: None for k in data.keys()}
        return {k: _expand_to_beam_size(v, beam_size, batch_size, state_info[k])
                for k, v in data.items()}
    elif isinstance(data, mx.nd.NDArray):
        if not state_info:
            batch_axis = 0
        else:
            batch_axis = state_info['__layout__'].find('N')
        if data.shape[batch_axis] != batch_size:
            raise ValueError('The batch dimension of all the inner elements in states must be '
                             '{}, Found shape={}'.format(batch_size, data.shape))
        new_shape = list(data.shape)
        new_shape[batch_axis] = batch_size * beam_size
        new_shape = tuple(new_shape)
        return data.expand_dims(batch_axis+1)\
                   .broadcast_axes(axis=batch_axis+1, size=beam_size)\
                   .reshape(new_shape)
    elif isinstance(data, mx.sym.Symbol):
        if not state_info:
            batch_axis = 0
        else:
            batch_axis = state_info['__layout__'].find('N')
        new_shape = (0, ) * batch_axis + (-3, -2)
        return data.expand_dims(batch_axis+1)\
                   .broadcast_axes(axis=batch_axis+1, size=beam_size)\
                   .reshape(new_shape)
    elif data is None:
        return None
    else:
        raise NotImplementedError

class GPT2Decoder(object):
    def __init__(self, gpt2_model):
        self._gpt2_model = gpt2_model

    def __call__(self, inputs, states):
        inputs = mx.nd.expand_dims(inputs, axis=1)
        return self._gpt2_model(inputs, states)


ctx = mx.gpu()
nlp.model.sequence_sampler._expand_to_beam_size = _expand_to_beam_size
from gluonnlp.model.sequence_sampler import SequenceSampler

model, vocab, tokenizer, detokenizer = load_pretrained_GPT2('117M', ctx=ctx)
decoder = GPT2Decoder(model)
eos_id = vocab[vocab.eos_token]
sampler = SequenceSampler(beam_size=1, max_length=1024, eos_id=eos_id, decoder=decoder)


unconditional_inputs = mx.nd.array([eos_id], dtype=np.int32, ctx=ctx)
samples, scores, valid_length = sampler(unconditional_inputs, None)
print(samples)
