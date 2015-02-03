import os
os.environ['GNUMPY_USE_GPU'] = 'yes'
import gnumpy as gnp
import numpy as np
import time



start = time.time()
for i in range(10):
    A = np.random.rand((1000, 1000))
    B = np.random.rand((1000, 1000))
    C = np.dot(A, B)

print '%.4f ms' % (time.time() - start)
start = time.time()

for i in range(10):
    Ag = gnp.as_garray(A)
    Bg = gnp.as_garray(B)
    A = np.random.rand((1000, 1000))
    B = np.random.rand((1000, 1000))
    C = np.dot(A, B)

print '%.4f ms' % (time.time() - start)