from stage import *
import numpy as np


class OrdinalRegression(Stage):
    def __init__(
                    self, 
                    outputDim, 
                    inputNames=None, 
                    name=None, 
                    outputdEdX=True):
        Stage.__init__(
                        self, 
                        name=name, 
                        inputNames=inputNames, 
                        outputDim=outputDim, 
                        outputdEdX=outputdEdX)
        # Uniform initialization
        # mu_0 = -1
        # mu_(n-1) = 1
        # mu_i = -1 + 2 * (i / n)
        mu = np.linspace(-1, 1, self.outputDim)
        # pi_i = 1/n
        pi = np.zeros(self.outputDim) + 1 / float(self.outputDim)
        self.W = np.zeros((2, self.outputDim))
        self.W[0] = mu
        self.W[1] = pi

    def forward(self, X):
        mu = self.W[0].reshape(1, self.W.shape[1])
        pi = self.W[1].reshape(1, self.W.shape[1])
        self.Xshape = X.shape
        X = X.reshape(X.shape[0], 1)
        self.X = X
        Z = np.exp(mu * X + (pi - np.power(mu, 2) / 2))
        Y = Z / np.sum(Z, axis=-1).reshape(X.shape[0], 1)
        self.Z = Z
        self.Y = Y
        return Y

    def backward(self, dEdY):
        # Here we ignore the dEdY because this is always the last layer...
        target = dEdY != 0.0
        targetInt = target.astype('int')
        targetIdx = np.nonzero(target)[1]
        mu = self.W[0]
        pi = self.W[1]
        dEdX = (-mu[targetIdx] + np.dot(self.Y, mu)) / float(self.X.shape[0])
        dEdX = dEdX.reshape(self.Xshape)
        dEdMu = np.mean(
            (self.X - mu) *  
            (self.Y - targetInt), axis=0)
        dEdPi = np.mean(self.Y - targetInt, axis=0)
        self.dEdW = np.zeros(self.W.shape)
        self.dEdW[0] = dEdMu
        self.dEdW[1] = dEdPi
        return dEdX