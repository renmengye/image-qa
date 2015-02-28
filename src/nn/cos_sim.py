from stage import *
import numpy as np

class CosSimilarity(Stage):
    """
    Compute the cosine similartiy of vectors with a bank of vectors
    """
    def __init__(self, bankDim, name=None):
        Stage.__init__(self, name=name)
        self.bankDim = bankDim

    def forward(self, X):
        bankDim = self.bankDim
        A = X[:bankDim]
        Z = X[bankDim:]
        Xnorm2 = np.sum(np.power(X, 2), axis=-1)
        Anorm2 = Xnorm2[:bankDim]
        Znorm2 = Xnorm2[bankDim:]
        Xnorm = np.sqrt(Xnorm2)
        Anorm = Xnorm[:bankDim]
        Znorm = Xnorm[bankDim:]
        Zuni = Z / Znorm.reshape(Z.shape[0], 1)
        Auni = A / Anorm.reshape(bankDim, 1)
        self.Y = np.inner(Zuni, Auni)
        self.A = A
        self.Z = Z
        self.Anorm = Anorm
        self.Znorm = Znorm
        self.Auni = Auni
        self.Zuni = Zuni
        return self.Y

    def backward(self, dEdY):
        # For now, output gradient towards the vector bank.
        self.dEdW = 0
        Z = self.Z
        A = self.A
        bankDim = self.bankDim
        Anorm = self.Anorm
        Znorm = self.Znorm
        Auni = self.Auni
        Zuni = self.Zuni

        V = np.dot(dEdY, Auni)
        dEdZ = np.sum(V * Z, axis=-1).reshape(Z.shape[0], 1) * \
        (-Z / (Znorm ** 3).reshape(Z.shape[0], 1)) + \
        V / Znorm.reshape(Z.shape[0], 1)

        U = np.dot(dEdY.transpose(), Zuni)
        dEdA = np.sum(U * A, axis=-1).reshape(A.shape[0], 1) * \
        (-A / (Anorm ** 3).reshape(A.shape[0], 1)) + \
        U / Anorm.reshape(A.shape[0], 1)

        dEdX = np.concatenate((dEdA, dEdZ), axis=0)
        return dEdX