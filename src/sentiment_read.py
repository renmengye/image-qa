import pickle
import numpy as np
from tsne import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

version = '20150117-142115'

with open('sentiment_results/sentiment-' + version + '.pip') as pipf:
    pipeline = pickle.load(pipf)

# lin_dict = trainer.stages[1].W
# lin_dict = lin_dict[:, 1:]
# lin_dict2 = tsne(lin_dict.transpose())
# np.save('sentiment_results/sentiment-20150117-142115-embedding.npy', lin_dict2)

data = np.load('../data/sentiment/train.npy')
word_array = data[1]
word_freq = data[2]

# Sort frequency
key = sorted(range(len(word_freq)), key=lambda k: word_freq[k], reverse=True)

lin_dict2 = np.load('sentiment_results/sentiment-' + version + '-embedding.npy')

plt.scatter(lin_dict2[:, 0], lin_dict2[:, 1], c=u'w', marker='.', linewidths=0.1)

# Plot top 100 words
for i in range(0, 100):
    if key[i] < lin_dict2.shape[0]:
        plt.axes().text(lin_dict2[key[i], 0], lin_dict2[key[i], 1], word_array[key[i]], fontsize=3)

plt.draw()
plt.show()
plt.savefig('sentiment_results/sentiment-' + version + '-embedding2.png', dpi=600)
print 'hello'
