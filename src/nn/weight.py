class Weight():
    """
    Designed as a weights matrix wrapper for the ease of sharing parameters.
    Currently only considered for gradient descent optimization.
    """
    def __init__(self, initializer, gradientDescentParams):
        self.gradientDescentParams = gradientDescentParams
        self.initializer = initializer
        pass

    def init(self):
        pass

    def addGradient(self, gradient):
        pass

    def update(self):
        pass