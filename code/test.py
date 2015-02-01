from trainer import *

taskId = sys.argv[1]
dataFile = sys.argv[2]

trainer = Trainer.initFromConfig(
    'imgword', '../results/%s/%s.yaml' % (taskId, taskId), '../results')
trainer.loadWeights('../results/%s/%s.w.npy' % (taskId, taskId))
data = np.load(dataFile)
trainer.test(data[2],data[3])