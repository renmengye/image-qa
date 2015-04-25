'''
Soft-attention mechanism for image search
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy
import os
import time

from scipy import optimize, stats
from collections import OrderedDict
from sklearn.cross_validation import KFold
from numpy.random import RandomState

import warnings

from homogeneous_data import HomogeneousData

import flickr8k
#import flickr30k
#import coco

# datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
datasets = {'flickr8k': (flickr8k.load_data, flickr8k.prepare_data)}
            #'flickr30k': (flickr30k.load_data, flickr30k.prepare_data),
            #'coco': (coco.load_data, coco.prepare_data)}

def get_dataset(name):
    return datasets[name][0], datasets[name][1]

'''
Theano shared variables require GPUs, so to
make this code more portable, these two functions
push and pull variables between a shared
variable dictionary and a regular numpy 
dictionary
'''
# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         state_before *
                         trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                         state_before * 0.5)
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

# all parameters
def init_params(options):
    """
    Initalize all model parameters here
    """
    params = OrderedDict()

    # Embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional LSTM_Old
    params = get_layer('lstm')[0](options, params, prefix='encoder', nin=options['dim_word'], dim=options['dim'])
    params = get_layer('lstm')[0](options, params, prefix='encoder_r', nin=options['dim_word'], dim=options['dim'])

    # sentence (or glimpse) projection
    params = get_layer('ff')[0](options, params, prefix='ff_proj', nin=2*options['dim'], nout=options['dim'])

    # init_state, init_cell
    for lidx in xrange(1, options['n_layers_init']):
        params = get_layer('ff')[0](options, params, prefix='ff_init_%d'%lidx, nin=options['ctx_dim'], nout=options['ctx_dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=options['ctx_dim'], nout=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=options['ctx_dim'], nout=options['dim'])

    # decoder: LSTM_Old
    params = get_layer('lstm_cond')[0](options, params, prefix='decoder', nin=options['ctx_dim'], dim=options['dim'], dimctx=options['ctx_dim'])

    return params

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive'%kk)
        params[kk] = pp[kk]
    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          }

def get_layer(name):
    """
    Part of the reason the init is very slow is because,
    the layer's constructor is called even when it isn't needed
    """
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are 
    orthogonal. 
    """
    W = numpy.random.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)

def rectifier(x):
    return tensor.maximum(0., x)

def linear(x):
    return x

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# LSTM_Old layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    # Stack the weight matricies for faster dot prods
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U
    #params[_p(prefix,'b')] = numpy.concatenate([numpy.ones((3 * dim,)).astype('float32'),
    #                                            numpy.zeros((1 * dim,)).astype('float32')], axis=1)
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params

# This function implements the lstm fprop
def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    dim = tparams[_p(prefix,'U')].shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        init_state = tensor.alloc(0., n_samples, dim)
        init_memory = tensor.alloc(0., n_samples, dim)
    else:
        n_samples = 1
        init_state = tensor.alloc(0., dim)
        init_memory = tensor.alloc(0., dim)

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        elif _x.ndim == 2:
            return _x[:, n*dim:(n+1)*dim]
        return _x[n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        h = o * tensor.tanh(c)

        return h, c, i, f, o, preact

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info = [init_state, init_memory, None, None, None, None],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False)
    return rval

# Conditional LSTM_Old layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None):
    """
    Initialize all parameters for conditional LSTM_Old
    """
    if dim == None:
        dim = options['dim']

    # input to LSTM_Old
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W

    # LSTM_Old to LSTM_Old
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    # bias to LSTM_Old
    #params[_p(prefix,'b')] = numpy.concatenate([numpy.ones((3 * dim,)).astype('float32'),
    #                                            numpy.zeros((1 * dim,)).astype('float32')], axis=1)
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, ortho=False)
    params[_p(prefix,'Wc_att')] = Wc_att

    # attention: LSTM_Old -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[_p(prefix,'Wd_att')] = Wd_att

    # attention: bidirectional LSTM_Old -> hidden
    We_att = norm_weight(dim,dimctx)
    params[_p(prefix,'We_att')] = We_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # deeper attention
    if options['n_layers_att'] > 1:
        for lidx in xrange(1, options['n_layers_att']):
            params[_p(prefix,'W_att_%d'%lidx)] = ortho_weight(dimctx)
            params[_p(prefix,'b_att_%d'%lidx)] = numpy.zeros((dimctx,)).astype('float32')

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[_p(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    if options['selector']:
        # attention: selector
        W_sel = norm_weight(dim, 1)
        params[_p(prefix, 'W_sel')] = W_sel
        b_sel = numpy.float32(0.)
        params[_p(prefix, 'b_sel')] = b_sel

    return params

def lstm_cond_layer(tparams, state_below, options, prefix='lstm',
                    mask=None, context=None, one_step=False,
                    init_memory=None, init_state=None,
                    trng=None, use_noise=None,
                    **kwargs):
    """
    Pass through conditional LSTM_Old layer
    Context refers to the image context
    state_below: output from bidirectional LSTM_Old
    """

    assert context, 'Context must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'U')].shape[0]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory 
    if init_memory == None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected context 
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix, 'b_att')]
    if options['n_layers_att'] > 1:
        for lidx in xrange(1, options['n_layers_att']):
            pctx_ = tensor.dot(pctx_, tparams[_p(prefix,'W_att_%d'%lidx)])+tparams[_p(prefix, 'b_att_%d'%lidx)]
            #pctx_list.append(pctx_)
            if lidx < options['n_layers_att'] - 1:
                pctx_ = tanh(pctx_)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_, a_, ct_, pctx_, dp_=None, dp_att_=None):
        """
        x_ is the state_below
        """
        # attention
        pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')]) + tensor.dot(x_, tparams[_p(prefix,'We_att')])
        pctx_ = pctx_ + pstate_[:,None,:]
        pctx_list = []
        pctx_list.append(pctx_)
        pctx_ = tanh(pctx_)
        alpha = tensor.dot(pctx_, tparams[_p(prefix,'U_att')])+tparams[_p(prefix, 'c_tt')]
        alpha_pre = alpha
        alpha_shp = alpha.shape
        alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
        ctx_ = (context * alpha[:,:,None]).sum(1) # current context
        if options['selector']:
            sel_ = tensor.nnet.sigmoid(tensor.dot(h_, tparams[_p(prefix, 'W_sel')])+tparams[_p(prefix,'b_sel')])
            sel_ = sel_.reshape([sel_.shape[0]])
            ctx_ = sel_[:,None] * ctx_

        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += tensor.dot(ctx_, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
  
        i = _slice(preact, 0, dim)
        f = _slice(preact, 1, dim)
        o = _slice(preact, 2, dim)
        if options['use_dropout_lstm']:
            i = i * _slice(dp_, 0, dim)
            f = f * _slice(dp_, 1, dim)
            o = o * _slice(dp_, 2, dim)
        i = tensor.nnet.sigmoid(i)
        f = tensor.nnet.sigmoid(f)
        o = tensor.nnet.sigmoid(o)
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tensor.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        rval = [h, c, alpha, ctx_]
        if options['selector']:
            rval += [sel_]
        rval += [pstate_, pctx_, i, f, o, preact, alpha_pre]+pctx_list
        return rval

    if options['use_dropout_lstm']:
        if options['selector']:
            _step0 = lambda m_, x_, dp_, h_, c_, a_, ct_, sel_, pctx_: \
                            _step(m_, x_, h_, c_, a_, ct_, pctx_, dp_)
        else:
            _step0 = lambda m_, x_, dp_, h_, c_, a_, ct_, pctx_: \
                            _step(m_, x_, h_, c_, a_, ct_, pctx_, dp_)
        dp_shape = state_below.shape
        if one_step:
            dp_mask = tensor.switch(use_noise,
                                    trng.binomial((dp_shape[0], 3*dim),
                                                  p=0.5, n=1, dtype=state_below.dtype),
                                    tensor.alloc(0.5, dp_shape[0], 3 * dim))
        else:
            dp_mask = tensor.switch(use_noise,
                                    trng.binomial((dp_shape[0], dp_shape[1], 3*dim),
                                                  p=0.5, n=1, dtype=state_below.dtype),
                                    tensor.alloc(0.5, dp_shape[0], dp_shape[1], 3*dim))
    else:
        if options['selector']:
            _step0 = lambda m_, x_, h_, c_, a_, ct_, sel_, pctx_: _step(m_, x_, h_, c_, a_, ct_, pctx_)
        else:
            _step0 = lambda m_, x_, h_, c_, a_, ct_, pctx_: _step(m_, x_, h_, c_, a_, ct_, pctx_)

    seqs = [mask, state_below]
    if options['use_dropout_lstm']:
        seqs += [dp_mask]
    outputs_info = [init_state,
                    init_memory,
                    tensor.alloc(0., n_samples, pctx_.shape[1]),
                    tensor.alloc(0., n_samples, context.shape[2])]
    if options['selector']:
        outputs_info += [tensor.alloc(0., n_samples)]
    outputs_info += [None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None] + [None]#*options['n_layers_att']
    rval, updates = theano.scan(_step0,
                                sequences=seqs,
                                outputs_info=outputs_info,
                                non_sequences=[pctx_],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False)

    return rval

# build a training model
def build_model(tparams, options):
    """
    Construct computation graph for the whole model
    """
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    # reverse description and mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    # context: #samples x #annotations x dim
    # ctxc: contrastive image
    ctx = tensor.tensor3('ctx', dtype='float32')
    ctxc = tensor.tensor3('ctxc', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # index into the word embedding matrix
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encoder:
    proj = get_layer('lstm')[1](tparams, emb, options, prefix='encoder', mask=x_mask)
    projr = get_layer('lstm')[1](tparams, embr, options, prefix='encoder_r', mask=xr_mask)
    ctx_lstm = concatenate((proj[0], projr[0]), axis=2)
    ctx_lstm = get_layer('ff')[1](tparams, ctx_lstm, options, prefix='ff_proj', activ='linear')

    # initial state/cell
    if options['use_ctx_mean']:
        ctx_mean = ctx.mean(1)
    else:
        ctx_mean = tensor.zeros_like(ctx.mean(1))
    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = get_layer('ff')[1](tparams, ctx_mean, options,
                                      prefix='ff_init_%d'%lidx, activ='rectifier')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, use_noise, trng)

    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')

    # initial state/cell for constrastive image
    if options['use_ctx_mean']:
        ctxc_mean = ctxc.mean(1)
    else:
        ctxc_mean = tensor.zeros_like(ctxc.mean(1))
    for lidx in xrange(1, options['n_layers_init']):
        ctxc_mean = get_layer('ff')[1](tparams, ctxc_mean, options,
                                       prefix='ff_init_%d'%lidx, activ='rectifier')
        if options['use_dropout']:
             ctxc_mean = dropout_layer(ctxc_mean, use_noise, trng)

    init_state_c = get_layer('ff')[1](tparams, ctxc_mean, options, prefix='ff_state', activ='tanh')
    init_memory_c = get_layer('ff')[1](tparams, ctxc_mean, options, prefix='ff_memory', activ='tanh')
    
    # decoder
    proj = get_layer('lstm_cond')[1](tparams, ctx_lstm, options,
                                     prefix='decoder',
                                     mask=x_mask, context=ctx,
                                     one_step=False,
                                     init_state=init_state,
                                     init_memory=init_memory,
                                     trng=trng,
                                     use_noise=use_noise)

    # decoder: contrastive image
    projc = get_layer('lstm_cond')[1](tparams, ctx_lstm, options,
                                     prefix='decoder',
                                     mask=x_mask, context=ctxc,
                                     one_step=False,
                                     init_state=init_state_c,
                                     init_memory=init_memory_c,
                                     trng=trng,
                                     use_noise=use_noise)

    # Collect some results
    proj_h = proj[0]
    alphas = proj[2]
    ctxs = proj[3]
    if options['selector']:
        sels = proj[4]

    # Collect results from contrastive image
    proj_h_c = projc[0]
    alphas_c = projc[2]
    ctxs_c = projc[3]
    if options['selector']:
        sels_c = projc[4]

    # Compute the cost function (pairwise ranking loss)
    lt = ctx_lstm
    lx = proj_h
    ly = proj_h_c
    if options['use_last']:
        lt = lt[0][None,:,:]
        lx = lx[-1][None,:,:]
        ly = ly[-1][None,:,:]
    if options['use_norm']:
        lnt = tensor.sqrt((lt ** 2).sum(axis=2))
        lnx = tensor.sqrt((lx ** 2).sum(axis=2))
        lny = tensor.sqrt((ly ** 2).sum(axis=2))
        lt = lt / lnt[:,:,None]
        lx = lx / lnx[:,:,None]
        ly = ly / lny[:,:,None]
    cost = (tensor.maximum(0., options['margin'] - (lt * lx).sum(axis=2) + (lt * ly).sum(axis=2))).sum(0).sum(0)  #mean(0).sum(0)

    opt_outs = dict()
    if options['selector']:
        opt_outs['selector'] = sels
        opt_outs['selector_c'] = sels_c

    return trng, use_noise, [x, x_mask, ctx, ctxc], alphas, alphas_c, cost, opt_outs


def build_ranker(tparams, options):
    """
    Construct computation graph for the ranker
    This is similar to the model, except scores are returned (without contrast)
    """
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    # reverse description and mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    # context: #samples x #annotations x dim
    ctx = tensor.tensor3('ctx', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # index into the word embedding matrix
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encoder:
    proj = get_layer('lstm')[1](tparams, emb, options, prefix='encoder', mask=x_mask)
    projr = get_layer('lstm')[1](tparams, embr, options, prefix='encoder_r', mask=xr_mask)
    ctx_lstm = concatenate((proj[0], projr[0]), axis=2)
    ctx_lstm = get_layer('ff')[1](tparams, ctx_lstm, options, prefix='ff_proj', activ='linear')

    # initial state/cell:
    if options['use_ctx_mean']:
        ctx_mean = ctx.mean(1)
    else:
        ctx_mean = tensor.zeros_like(ctx.mean(1))
    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = get_layer('ff')[1](tparams, ctx_mean, options,
                                      prefix='ff_init_%d'%lidx, activ='rectifier')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, use_noise, trng)

    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')

    # decoder
    proj = get_layer('lstm_cond')[1](tparams, ctx_lstm, options,
                                     prefix='decoder',
                                     mask=x_mask, context=ctx,
                                     one_step=False,
                                     init_state=init_state,
                                     init_memory=init_memory,
                                     trng=trng,
                                     use_noise=use_noise)

    # Collect some results
    proj_h = proj[0]
    alphas = proj[2]
    ctxs = proj[3]  
    if options['selector']:
        sels = proj[4]

    # Compute the similarity score between examples
    lt = ctx_lstm
    lx = proj_h
    if options['use_last']:
        lt = lt[0][None,:,:]
        lx = lx[-1][None,:,:]
    if options['use_norm']:
        lnt = tensor.sqrt((lt ** 2).sum(axis=2))
        lnx = tensor.sqrt((lx ** 2).sum(axis=2))
        lt = lt / lnt[:,:,None]
        lx = lx / lnx[:,:,None]
    scores = (lt * lx).sum(axis=2).sum(0)

    opt_outs = dict()
    if options['selector']:
        opt_outs['selector'] = sels

    return trng, use_noise, [x, x_mask, ctx], alphas, scores, opt_outs


#TODO: held out rankings
def pred_ranks(f_ranker, options, worddict, prepare_data, data, iterator, idx, verbose=False):
    """
    Return a list of scores for a single sentence for each image
    """
    n_images = len(data[1])
    scores = numpy.zeros((n_images, 1)).astype('float32')
    
    n_done = 0

    # This computation is wasteful, TODO: make it better!
    for _, valid_index in iterator:
        x, mask, ctx = prepare_data(data[0][idx:idx+1]*len(valid_index) + [data[0][5*t] for t in valid_index],
                                     data[1],
                                     worddict,
                                     maxlen=options['maxlen'],
                                     n_words=options['n_words'])
        pred_scores = f_ranker(x[:,:len(valid_index)], mask[:,:len(valid_index)], ctx[len(valid_index):])
        scores[valid_index] = pred_scores[:,None]
 
        n_done += len(valid_index)
        if verbose:
            print '%d/%d images computed'%(n_done,n_images)

    return scores.flatten()


# Evaluation metrics
def recallK(f_ranker, options, worddict, prepare_data, data, iterator, queries, verbose=False):
    """
    Return recall@k, median rank for the given queries
    Queries: list of sentence indices to use as queries
    """
    n_images = len(data[1])
    n_queries = len(queries)
    scores = numpy.zeros((n_queries, n_images)).astype('float32')
    for i, idx in enumerate(queries):
        print (i,idx)
        scores[i] = pred_ranks(f_ranker, options, worddict, prepare_data, data, iterator, idx, verbose=verbose)

    # Obtain rankings
    gt = [data[0][q][1] for q in queries]
    ranks = numpy.zeros(n_queries)
    for i in range(n_queries):
        idxs = numpy.argsort(scores[i].flatten())[::-1]
        ranks[i] = numpy.where(idxs == gt[i])[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    r25 = 100.0 * len(numpy.where(ranks < 25)[0]) / len(ranks)
    r50 = 100.0 * len(numpy.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(numpy.where(ranks < 100)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1  #add 1 since ranks start at 0

    return (r1, r5, r10, r25, r50, r100, medr)
    

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * numpy.float32(0.))
        v = theano.shared(p.get_value() * numpy.float32(0.))
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update


def validate_options(options):

    if options['dim_word'] > options['dim']:
        warnings.warn('dim_word should only be as large as dim.')

    return options

def train(dim_word=256, # word vector dimensionality
          ctx_dim=512, # context vector dimensionality
          dim=256, # the number of LSTM_Old units
          margin=0.2, # margin for pairwise ranking loss. Should be (0,1] if use_norm is on
          use_norm=True, # whether to L2norm vectors prior to loss
          use_ctx_mean=False, # whether to initialze decoder to annotation means
          use_last=False, #Only use last hidden state for ranking
          n_layers_att=1,
          n_layers_init=1, # This isn't useful if use_ctx_mean=False
          patience=10,
          max_epochs=5000,
          dispFreq=1,
          decay_c=0.,
          alpha_c=1.,
          lrate=0.01,
          selector=True,
          n_words=23461,
          maxlen=100, # maximum length of the description
          optimizer='adam',
          batch_size = 64,
          valid_batch_size = 128,
          saveto='/ais/gobi3/u/rkiros/flickr8k/rank_models/lstm_toy.npz',
          validFreq=200,
          total_queries=5000, # total number of queries
          n_queries=50, # number of queries to validate on, resampled each time
          saveFreq=200, # save the parameters after every saveFreq updates
          sampleFreq=200, # generate some samples after every sampleFreq updates
          dataset='flickr8k',
          dictionary=None, # word dictionary
          use_dropout=False,
          use_dropout_lstm=False,
          reload_=False):

    # Model options
    print alpha_c
    model_options = locals().copy()
    model_options = validate_options(model_options)

    # reload options
    if reload_ and os.path.exists(saveto):
        print "Reloading options"
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)

    print 'Loading data'
    load_data, prepare_data = get_dataset(dataset)
    train, valid, test, worddict = load_data()

    # Invert the dictionary, add special tokens
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print "Reloading model"
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
          inps, alphas, \
          alphas_contrast, cost, \
          opts_out = \
          build_model(tparams, model_options)

    print 'Building ranker'
    trng_r, use_noise_r, \
            inps_r, alphas_r, \
            scores, opts_out_r = \
            build_ranker(tparams, model_options)

    # before any regularizer
    print 'Building functions'
    f_log_probs = theano.function(inps, -cost, profile=False)
    f_ranker = theano.function(inps_r, scores, profile=False)

    print 'Regularization'
    cost = cost.mean()
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    if alpha_c > 0.:
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((1.-alphas.sum(0))**2).sum(0).mean()
        alpha_reg_contrast = alpha_c * ((1.-alphas_contrast.sum(0))**2).sum(0).mean()
        cost += alpha_reg
        cost += alpha_reg_contrast

    # gradient computation
    print 'Computing gradients'
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=maxlen)

    if valid:
        kf_valid = KFold(len(valid[1]), n_folds=len(valid[1])/valid_batch_size, shuffle=False)
    if test:
        kf_test = KFold(len(test[1]), n_folds=len(test[1])/valid_batch_size, shuffle=False)

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = numpy.load(saveto)['history_errs'].tolist()
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size

    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx

        for caps in train_iter:
            n_samples += len(caps)
            uidx += 1
            use_noise.set_value(1.)

            pd_start = time.time()
            x, mask, ctx = prepare_data(caps,
                                        train[1],
                                        worddict,
                                        maxlen=maxlen,
                                        n_words=n_words)
            pd_duration = time.time() - pd_start

            # Get some contrastive images
            prng = RandomState(eidx + n_samples)
            inds = numpy.arange(len(train[1]))
            prng.shuffle(inds)
            contrast_ctx = numpy.zeros((len(caps), train[1][0].shape[1])).astype('float32')
            for cidx in range(len(caps)):
                contrast_ctx[cidx,:] = numpy.array(train[1][inds[cidx]].todense())
            contrast_ctx = contrast_ctx.reshape([contrast_ctx.shape[0], 14*14, 512])

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen
                continue

            ud_start = time.time()
            cost = f_grad_shared(x, mask, ctx, contrast_ctx)
            f_update(lrate)
            ud_duration = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'PD ', pd_duration, 'UD ', ud_duration

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p != None:
                    params = copy.copy(best_p)
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0

                if valid:

                    queries = numpy.arange(total_queries)
                    prng.shuffle(queries)
                    (r1, r5, r10, r25, r50, r100, medr) = recallK(f_ranker, model_options, worddict, prepare_data, valid, kf_valid, queries[:n_queries], verbose=False)
                    print "Recall@(1,5,10,25,50,100): %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, r25, r50, r100, medr)

                    #TODO: Not sure if this is the best choice, maybe explore alternatives
                    valid_err = medr
                    history_errs.append([valid_err, 1e20])

                # Use the median rank to decide when to stop
                if uidx == 0 or valid_err <= numpy.array(history_errs)[:,0].min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if eidx > patience and valid_err >= numpy.array(history_errs)[:,0].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

        print 'Seen %d samples'%n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    train_err = 0
    valid_err = 0
    test_err = 0

    queries = numpy.arange(total_queries)
    prng.shuffle(queries)
    (r1, r5, r10, r25, r50, r100, medr) = recallK(f_ranker, model_options, worddict, prepare_data, valid, kf_valid, queries[:n_queries], verbose=False)
    print "Recall@(1,5,10,25,50,100): %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, r25, r50, r100, medr)
    valid_err = medr

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, train_err=train_err,
                valid_err=valid_err, test_err=test_err, history_errs=history_errs,
                **params)

if __name__ == '__main__':
    pass





