from fully_connected_layer import *
from activation_fn import *
from weight import *
from weight_initializer import *

layer = FullyConnectedLayer(name='fc',
                            activationFn=SigmoidActivationFn(),
                            weight=Weight(
                                initializer=UniformWeightInitializer(
                                    limit=[-0.5, 0.5],
                                    seed=2,
                                    shape=[5, 6],
                                    affine=True),
                                gdController=None))
X = np.random.rand(5, 4)
if USE_GPU:
    X = gnp.as_garray(X)
print 'X:', X, X.shape
W = layer.weight.get()
print 'W:', W, W.shape
Y = layer.forward(X)
print 'Y:', Y, Y.shape
dEdY = Y ** 2
dEdX = layer.backward(dEdY)
print 'dEdX:', dEdX, dEdX.shape