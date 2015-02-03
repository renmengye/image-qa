import os
os.environ['GNUMPY_USE_GPU'] = 'yes'
import gnumpy as gnp
import numpy as np
import cudamat as cm
from cudamat import gpu_lock2 as gpu_lock
import time

def LockGPU():
  board = gpu_lock.obtain_lock_id()
  if board == -1:
    print 'No GPU board available.'
    sys.exit(1)
  else:
    cm.cuda_set_device(board)
    cm.cublas_init()
  return board

def FreeGPU(board):
    cm.cublas_shutdown()
    gpu_lock.free_lock(board)  # Optional.


def main():
    start = time.time()
    for i in range(10):
        A = np.random.rand((1000, 1000))
        B = np.random.rand((1000, 1000))
        C = np.dot(A, B)

    print '%.4f ms' % (time.time() - start)
    start = time.time()

    for i in range(10):
        A = np.random.rand((1000, 1000))
        B = np.random.rand((1000, 1000))
        Ag = gnp.as_garray(A)
        Bg = gnp.as_garray(B)
        C = gnp.dot(Ag, Bg)

    print '%.4f ms' % (time.time() - start)

if __name__ == '__main__':
    board = LockGPU()
    print 'Using board', board
    cm.CUDAMatrix.init_random(0)
    main()
    FreeGPU(board)
