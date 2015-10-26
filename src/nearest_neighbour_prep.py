import numpy as np
import scipy.sparse
import h5py
import os
import sys

def prep(data, 
         imageFeat, 
         wordEmbedding, 
         imageFeatMean,
         imageFeatStd,
         normalizeImage=True,
         bowMean=None, 
         bowStd=None,
         normalizeBow=True):
    dataDim = imageFeat.shape[1] + wordEmbedding.shape[1]
    output = np.zeros((data.shape[0], dataDim), dtype='float32')
    for i in range(data.shape[0]):
        imageID = data[i, 0, 0]
        img = imageFeat[imageID - 1]
        bow = []
        for j in range(1, data.shape[1]):
            if data[i, j, 0] > 0:
                bow.append(wordEmbedding[data[i, j, 0] - 1])
            else:
                break
        bow = np.sum(bow, axis=0)
        if normalizeImage:
            output[i, :imageFeat.shape[1]] = \
                (img.todense() - imageFeatMean) / imageFeatStd
        else:
            output[i, :imageFeat.shape[1]] = img.todense()
        output[i, imageFeat.shape[1]:] = bow
    if normalizeBow:
        if bowMean is None:
            bowMean = np.mean(output[:, imageFeat.shape[1]:], axis=0)
        if bowStd is None:
            bowStd = np.std(output[:, imageFeat.shape[1]:], axis=0)
            for i in range(wordEmbedding.shape[1]):
                if bowStd[i] == 0.0:
                    bowStd[i] = 1.0
        output[:, imageFeat.shape[1]:] = \
            (output[:, imageFeat.shape[1]:] - bowMean) / bowStd
    return output, bowMean, bowStd

if __name__ == '__main__':
    dataFolder = '../data/cocoqa'
    outputFolder = '../../../data/cocoqa-nearest'
    normalizeImage = False
    normalizeBow = False
    for i, flag in enumerate(sys.argv):
        if flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-normimg':
            normalizeImage = True
        elif flag == '-normbow':
            normalizeBow = True
        elif flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
    trainFile = 'train.npy'
    validFile = 'valid.npy'
    testFile = 'test.npy'
    if dataFolder == '../data/cocoqa':
        imageFeatFile = '/ais/gobi3/u/mren/data/cocoqa-full/hidden_oxford.h5'
        modelWeightsFile = '/ais/gobi3/u/mren/models/img_bow.h5'
    elif dataFolder == '../data/daquar-37':
        imageFeatFile = \
            '/ais/gobi3/u/mren/data/nyu-depth/hidden_oxford_sparse.h5'
        modelWeightsFile = '/ais/gobi3/u/mren/models/img_bow_dq.h5'
    print 'Data folder', dataFolder
    print 'Image feat file', imageFeatFile
    print 'Output folder', outputFolder
    print 'Normalize image', normalizeImage
    print 'Normalize bow', normalizeBow
    
    trainDataFile = os.path.join(dataFolder, trainFile)
    validDataFile = os.path.join(dataFolder, validFile)
    testDataFile = os.path.join(dataFolder, testFile)

    outputFilename = 'all_i%s_b%s.h5' % ('norm' if normalizeImage else 'raw',
                                         'norm' if normalizeBow else 'raw')
    trainData = np.load(trainDataFile)
    validData = np.load(validDataFile)
    testData = np.load(testDataFile)
    imageFeatH5 = h5py.File(imageFeatFile)
    imageFeatData = imageFeatH5['hidden7_data'][:]
    imageFeatInd = imageFeatH5['hidden7_indices'][:]
    imageFeatPtr = imageFeatH5['hidden7_indptr'][:]
    imageFeatShape = imageFeatH5['hidden7_shape'][:]
    imageFeatMean = imageFeatH5['hidden7_mean'][:]
    imageFeatStd = imageFeatH5['hidden7_std'][:]
    imageFeat = scipy.sparse.csr_matrix(
        (imageFeatData, imageFeatInd, imageFeatPtr), shape=imageFeatShape)
    modelWeightsH5 = h5py.File(modelWeightsFile)
    wordEmbedding = modelWeightsH5['txtDict'][:]
    trainDataNN, bowMean, bowStd = prep(
        data=trainData[0],
        imageFeat=imageFeat, 
        wordEmbedding=wordEmbedding,
        normalizeImage=normalizeImage,
        imageFeatMean=imageFeatMean,
        imageFeatStd=imageFeatStd,
        normalizeBow=normalizeBow)
    validDataNN, _, __ = prep(
        data=validData[0],
        imageFeat=imageFeat, 
        wordEmbedding=wordEmbedding,
        imageFeatMean=imageFeatMean,
        imageFeatStd=imageFeatStd,
        normalizeImage=normalizeImage,
        bowMean=bowMean,
        bowStd=bowStd,
        normalizeBow=normalizeBow)
    testDataNN, _, __ = prep(
        data=testData[0],
        imageFeat=imageFeat, 
        wordEmbedding=wordEmbedding,
        imageFeatMean=imageFeatMean,
        imageFeatStd=imageFeatStd,
        normalizeImage=normalizeImage,
        bowMean=bowMean,
        bowStd=bowStd,
        normalizeBow=normalizeBow)
    print trainDataNN[0]
    allData = h5py.File(os.path.join(outputFolder, outputFilename), 'w')
    allData['trainData'] = trainDataNN
    allData['trainLabel'] = trainData[1]
    allData['validData'] = validDataNN
    allData['validLabel'] = validData[1]
    allData['testData'] = testDataNN
    allData['testLabel'] = testData[1]
