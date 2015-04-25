import matplotlib.pyplot as plt
import numpy

# Finetuned Google News word2vec 300
tsneFile = '../results/imgwd_w2v300_word_embed_tsne.npy'
pngFile = '../results/imgwd_w2v300.png'

# Finetuned Custom COCO word2vec 500
# tsneFile = '../results/imgwd_b2i2w_cw2v500_word_embed_tsne.npy'
# pngFile = '../results/imgwd_b2i2w_cw2v500.png'

# Finetuned Random 500
# tsneFile = '../results/imgwd_b2i2w_rw500_word_embed_tsne.npy'
# pngFile = '../results/imgwd_b2i2w_rw500.png'

# Original Custom COCO word2vec 500
#tsneFile = '../results/w2v300_word_embed_tsne.npy'
#pngFile = '../results/w2v300.png'

# Original Google News word2vec 300
#tsneFile = '../results/cw2v500_word_embed_tsne.npy'
#pngFile = '../results/cw2v500.png'

wordFile = '../data/cocoqa-toy/question_vocabs.txt'
freqFile = '../data/cocoqa-toy/question_vocabs_freq.txt'

with open(wordFile) as f:
    words = f.readlines()
freq = numpy.loadtxt(freqFile)
tsneMat = numpy.load(tsneFile)

fig = plt.figure()
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

for i, item in enumerate(zip(words, freq)):
    w = item[0]
    f = item[1]
    if f > 30:
        ax.text(tsneMat[i, 0], tsneMat[i, 1], w, fontsize=8)

ax.axis([-100, 100, -100, 100])

plt.savefig(pngFile)
plt.show()