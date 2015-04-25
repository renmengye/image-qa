import numpy as np

out_folder = '../data/daquar-37'
D = 300
H = 300
start = 0.05
end = 0.05
random = np.random.RandomState(4)
Wxi = random.uniform(start, end, (D, H))
Wxf = random.uniform(start, end, (D, H))
Wxc = random.uniform(start, end, (D, H))
Wxo = random.uniform(start, end, (D, H))
Wyi = random.uniform(start, end, (H, H))
Wyf = random.uniform(start, end, (H, H))
Wyc = random.uniform(start, end, (H, H))
Wyo = random.uniform(start, end, (H, H))
Wci = random.uniform(start, end, (H, H))
Wcf = random.uniform(start, end, (H, H))
Wco = random.uniform(start, end, (H, H))
Wyi, s, v = np.linalg.svd(Wyi)
Wyf, s, v = np.linalg.svd(Wyf)
Wyc, s, v = np.linalg.svd(Wyc)
Wyo, s, v = np.linalg.svd(Wyo)
Wci, s, v = np.linalg.svd(Wci)
Wcf, s, v = np.linalg.svd(Wcf)
Wco, s, v = np.linalg.svd(Wco)

Wbi = np.ones((1, H))
Wbf = np.ones((1, H))
Wbc = np.zeros((1, H))
Wbo = np.ones((1, H))

Wi = np.concatenate((Wxi, Wyi, Wci, Wbi), axis=0)
Wf = np.concatenate((Wxf, Wyf, Wcf, Wbf), axis=0)
Wc = np.concatenate((Wxc, Wyc, Wbc), axis=0)
Wo = np.concatenate((Wxo, Wyo, Wco, Wbo), axis=0)
W = np.concatenate((Wi, Wf, Wc, Wo), axis=0)

np.save(out_folder + '/lstm-init.npy', W)
