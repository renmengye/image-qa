from stage import *

class TimeSelect(Stage):
    def __init__(self, time):
        Stage.__init__(self)
        self.t = time
        self.X = 0
        self.Y = 0
        pass

    def forwardPass(self, X):
        # X(t, n, i)
        Y = X[:, self.t, :]
        self.X = X
        self.Y = Y
        return Y

    def backPropagate(self, dEdY):
        self.dEdW = 0
        dEdX = np.zeros(self.X.shape)
        dEdX[:, self.t,  :] = dEdY
        return dEdX

