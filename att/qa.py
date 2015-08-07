# Image-based question answering with soft attention
import os
os.environ['THEANO_FLAGS'] = ('device=gpu%d' % 2)
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import sys
import copy
import time

from scipy import optimize, stats
from collections import OrderedDict

import warnings

from homogeneous_data import HomogeneousData

import flickr8k, daquar, cocoqa

# datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
# This  is a modified flickr8k with random 'answer' labels - just for debugging purposes
datasets = {
    'flickr8k': (flickr8k.load_data, flickr8k.prepare_data),
    'daquar': (daquar.load_data, daquar.prepare_data),
    'cocoqa': (cocoqa.load_data, cocoqa.prepare_data)
}

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
   
    # Word embedding
    # NOTE: Consider initializing this to pretrained word2vec
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # Initial LSTM_Old state and memory - optionally more layers
    ctx_dim = options['ctx_dim']
    for lidx in xrange(1, options['n_layers_init']):
        params = get_layer('ff')[0](options, params, prefix='ff_init_%d'%lidx, nin=ctx_dim, nout=ctx_dim)
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=ctx_dim, nout=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=ctx_dim, nout=options['dim'])

    # LSTM_Old decoder. Note that this assumes only 1 LSTM_Old layer
    params = get_layer('lstm_cond')[0](options, params, prefix='decoder',
                                       nin=options['dim_word'], dim=options['dim'],
                                       dimctx=ctx_dim)

    # LSTM_Old hidden state -> hidden layer
    #params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['dim_word'])
    
    # This is if you connect the context into the answer prediction
    #if options['ctx2out']:
    #    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx', nin=ctx_dim, nout=options['dim_word'])

    # Additional layers - here we assume the hidden dimensionality is the same as the word embedding dimension
    #if options['n_layers_out'] > 1:
    #    for lidx in xrange(1, options['n_layers_out']):
    #        params = get_layer('ff')[0](options, params, prefix='ff_logit_h%d'%lidx, nin=options['dim_word'], nout=options['dim_word'])

    # Output parameters for predicting the answer
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim'], nout=options['n_answers'])

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
# ff: feedforward layer, lstm-cond: conditional LSTM_Old layer
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          }

def get_layer(name):
    """
    Part of the reason the init is very slow is because,
    the layer's constructor is called even when it isn't needed
    """
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# UTILITY FUNCTIONS - for weight initializations and activations
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

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

# This by default uses tanh. Consider modifying for ReLU
def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# Conditional LSTM_Old layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None):
    """
    Initialize all conditional LSTM_Old parameters.
    It might be helpful to look at the computation graph construction to see where these are all used.
    """
    if nin == None:
        nin = options['dim']
    if dim == None:
        dim = options['dim']
    if dimctx == None:
        dimctx = options['dim']

    # input to LSTM_Old
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W

    # LSTM_Old to LSTM_Old - use orthogonal weight init. for recurrent connections
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    # bias to LSTM_Old
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    # context to LSTM_Old
    Wc = norm_weight(dimctx,dim*4)
    params[_p(prefix,'Wc')] = Wc

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, ortho=False)
    params[_p(prefix,'Wc_att')] = Wc_att

    # attention: LSTM_Old -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[_p(prefix,'Wd_att')] = Wd_att

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

    # Selector variables: these are used to weight the effect of the context
    # E.g. for caption generation, stop words would have lower selector weight
    if options['selector']:
        W_sel = norm_weight(dim, 1)
        params[_p(prefix, 'W_sel')] = W_sel
        b_sel = numpy.float32(0.)
        params[_p(prefix, 'b_sel')] = b_sel

    return params

# Set up the computational graph for the conditional LSTM_Old
# This is the most complex part of the code: everything needs to be set up specifically for 'scan'
def lstm_cond_layer(tparams, state_below, options, prefix='lstm',
                    mask=None, context=None, one_step=False,
                    init_memory=None, init_state=None,
                    trng=None, use_noise=None,
                    **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_memory, 'previous memory must be provided'
        assert init_state, 'previous state must be provided'

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

    # projected x
    # state_below is timesteps*num samples by d in training
    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_, a_, ct_, pctx_, dp_=None, dp_att_=None):
        # attention
        pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')])
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
        preact += x_
        preact += tensor.dot(ctx_, tparams[_p(prefix, 'Wc')])

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

    if one_step:
        if options['use_dropout_lstm']:
            if options['selector']:
                rval = _step0(mask, state_below, dp_mask, init_state, init_memory, None, None, None, pctx_)
            else:
                rval = _step0(mask, state_below, dp_mask, init_state, init_memory, None, None, pctx_)
        else:
            if options['selector']:
                rval = _step0(mask, state_below, init_state, init_memory, None, None, None, pctx_)
            else:
                rval = _step0(mask, state_below, init_state, init_memory, None, None, pctx_)
    else:
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


# build a training model - sets up the whole computational graph
def build_model(tparams, options):
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    # context: #samples x #annotations x dim
    ctx = tensor.tensor3('ctx', dtype='float32')

    # labels (answers)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # index into the word embedding matrix, shift it forward in time
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    ctx0 = ctx

    # initial state/cell
    ctx_mean = ctx0.mean(1)
    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = get_layer('ff')[1](tparams, ctx_mean, options,
                                      prefix='ff_init_%d'%lidx, activ='rectifier')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, use_noise, trng)

    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')

    # decoder
    proj = get_layer('lstm_cond')[1](tparams, emb, options,
                                     prefix='decoder',
                                     mask=mask, context=ctx0,
                                     one_step=False,
                                     init_state=init_state,
                                     init_memory=init_memory,
                                     trng=trng,
                                     use_noise=use_noise)

    # Extract each component of proj
    proj_h = proj[0]
    alphas = proj[2]
    ctxs = proj[3]
    if options['selector']:
        sels = proj[4]
    if options['use_dropout']:
        proj_h = dropout_layer(proj_h, use_noise, trng)
    
    # We only get the answer at the end - so only take last hidden state
    # TODO: Make sure this does what it should be!
    proj_h = proj_h[-1]
       
    # Make predictions
    scores = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit', activ='linear')
    probs = tensor.nnet.softmax(scores)

    # Cost function
    cost = -tensor.log(probs[tensor.arange(y.size), y] + 1e-8).mean()

    opt_outs = dict()
    if options['selector']:
        opt_outs['selector'] = sels

    return trng, use_noise, [x, mask, ctx, y], alphas, cost, opt_outs, probs 

# OPTIMIZERS
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inps, cost):
    """
    Adam: A Method for Stochastic Optimization (Diederik Kingma, Jimmy Ba)
    """
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inps, cost, updates=gsup)

    # Magic numbers
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
        
        # float32?
        m_t = m_t.astype('float32')
        v_t = v_t.astype('float32')
        p_t = p_t.astype('float32')
        
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update


def validate_options(options):
    """
    This will print some warnings when the program starts
    """
    #if options['ctx2out']:
    #    warnings.warn('Feeding context to output directly seems to hurt.')

    if options['dim_word'] > options['dim']:
        warnings.warn('dim_word should only be as large as dim.')

    return options

def train(dim_word=100, # word vector dimensionality
          ctx_dim=512, # context vector dimensionality
          dim=1000, # the number of LSTM_Old units
          n_layers_att=1,
          n_layers_init=1,
          n_answers=5,
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0.,
          alpha_c=0.,
          lrate=0.01,
          selector=False,
          n_words=23461,
          maxlen=100, # maximum length of the description/question
          optimizer='adam',
          batch_size = 16,
          valid_batch_size = 16,
          saveto='params.npz',
          validFreq=-1,
          saveFreq=-1, # save the parameters after every saveFreq updates
          dataset='flickr8k',
          dictionary=None, # word dictionary
          use_dropout=False,
          use_dropout_lstm=False,
          reload_=False,
          test_=False):

    # Model options
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

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print "Reloading model"
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, \
    use_noise, \
    inps, \
    alphas, \
    cost, \
    opts_out, \
    probs = \
    build_model(tparams, model_options)
    # before any regularizer
    f_pred_probs = \
        theano.function((inps[0], inps[1], inps[2]), probs, profile=False)
    f_alpha = \
        theano.function(inps, alphas, name='f_alpha', on_unused_input='ignore')
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
        cost += alpha_reg

    # gradient computation
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = \
        eval(optimizer)(lr, tparams, grads, inps, cost)
    
    if test_:
        print 'Test'
        use_noise.set_value(0.)
        test_correct = 0
        test_total = 0
        test_iter = HomogeneousData(test, maxlen=maxlen)
        tsave_total = []
        for tbatch in test_iter:
            tx, tx_mask, tctx, ty = prepare_data(\
                tbatch, test[1], worddict)
            tlogprob = f_pred_probs(tx, tx_mask, tctx)
            talpha = f_alpha(tx, tx_mask, tctx, ty)
            tout = numpy.argmax(tlogprob, axis=-1)
            tsave_total.append((tbatch, tlogprob, talpha))
            test_correct += numpy.sum((tout == ty).astype('int64'))
            test_total += ty.size
        tr = test_correct / float(test_total)
        print 'Test Acc %.5f' % tr
        numpy.save('test.out.npy', numpy.array(tsave_total, dtype=object))
        sys.exit()

    print 'Optimization'
    train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=maxlen)
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
    cost_total = 0.
    ex_total = 0
    correct_total = 0
    ep_start = time.time()
    for eidx in xrange(max_epochs):
        n_samples = 0

        for batch in train_iter:
            n_samples += len(batch)
            uidx += 1
            use_noise.set_value(1.)

            # Input question, mask, image context, answer.
            x, mask, ctx, y = prepare_data(batch,
                                           train[1],
                                           worddict)

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen
                continue

            probs = f_pred_probs(x, mask, ctx)
            choice = numpy.argmax(probs, axis=-1)
            cost = f_grad_shared(x, mask, ctx, y)
            cost_total += cost * y.size
            ex_total += y.size
            correct_total += numpy.sum((choice == y).astype('int64'))
            f_update()

        print 'Epoch %4d' % eidx, \
              'Cost %.5f' % (cost_total / float(ex_total)), \
              'Train Acc %.5f' % (correct_total / float(ex_total)),
        use_noise.set_value(0.)
        valid_iter = HomogeneousData(\
            valid, batch_size=100, maxlen=maxlen)
        vcorrect = 0
        vtotal = 0
        vcost_total = 0.0
        for vbatch in valid_iter:
            vx, vx_mask, vctx, vy = prepare_data(\
                vbatch, valid[1], worddict)
            vlogprob = f_pred_probs(vx, vx_mask, vctx)
            vcost_total += f_grad_shared(vx, vx_mask, vctx, vy) * vy.size
            vout = numpy.argmax(vlogprob, axis=-1)
            vcorrect += numpy.sum((vout == vy).astype('int64'))
            vtotal += vy.size
        vr = vcorrect / float(vtotal)
        vcost = vcost_total / float(vtotal)
        # history_errs.append(1-vr)
        history_errs.append(vcost)
        print 'VCost %.5f' % vcost,
        print 'Valid Acc %.5f' % vr,
        print 'Time', int(time.time() - ep_start)
        if uidx == 0 or vcost <= numpy.array(history_errs).min():
        #if uidx == 0 or 1-vr <= numpy.array(history_errs).min():
            best_p = unzip(tparams)
            bad_counter = 0
            numpy.savez(saveto, history_errs=history_errs, **best_p)
            pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
        else:
            bad_counter += 1

        if bad_counter > patience:
            print 'Early stop!'
            break
    
if __name__ == '__main__':
    dropout = False
    #train(dataset='daquar', \
    #    n_answers=63, \
    #    use_dropout=dropout, \
    #    use_dropout_lstm=dropout);
    train(dataset='cocoqa',
        n_answers=431,
        use_dropout=dropout,
        use_dropout_lstm=dropout,
        reload_=False,
        test_=False);
