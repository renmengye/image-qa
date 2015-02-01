from trainer import *
trainer = Trainer.initFromConfig(
    'imgword', '../results/imgword-20150201-160840/imgword-20150201-160840.yaml', '../results')
trainer.loadWeights('../results/imgword-20150201-160840/imgword-20150201-160840.w.npy')
data = np.load('../data/imgword/train-37.npy')
trainer.test(data[2],data[3])