existingWordEmbedFile = '../../../data/mscoco/word_embed.npy'
existingVocabFile = '../data/cocoqa-toy/question_vocabs.txt'
newVocabFile = '../data/daquar-37/question_vocabs.txt'
newWordEmbedFile = '../data/daquar-37/word_embed_coco.npy'
i = 1
vocabDict = {}
with open(existingVocabFile) as f:
    for line in f:
        vocabDict[line[:-1]] = i
        i += 1
vocabList = []
with open(newVocabFile) as f:
    for line in f:
        word = line[:-1]
        if vocabDict.has_key(word):
            vocabList.append(vocabDict[word])
            print 'yes'
        else:
            vocabList.append(0)
            print 'no'
import numpy
vocabIdx = numpy.array(vocabList)
wordEmbed = numpy.load(existingWordEmbedFile)
newWordEmbed = wordEmbed[vocabIdx]
random = numpy.random.RandomState(2)
for i in range(newWordEmbed.shape[0]):
    if numpy.mean(newWordEmbed[i]) == 0.0:
        newWordEmbed[i] = random.uniform(-0.5, 0.5, (newWordEmbed.shape[1]))
print newWordEmbed
numpy.save(newWordEmbedFile, newWordEmbed)
