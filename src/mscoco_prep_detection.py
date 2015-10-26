import json
import numpy as np
import os
import cPickle as pkl
import h5py
import scipy.sparse
import sys

def buildImageDict(annotations):
    """
    Returns a dictionary.
    Key is image ID.
    Value is image.
    """
    result = {}
    for image in annotations['images']:
        result[image['id']] = image
    return result

def buildCategoryDict(annotations, offset=1):
    """
    Reindexes category.
    Returns a dictionary.
    Key is old category ID.
    Value is new category ID.
    """
    result = {}
    count = offset
    for category in annotations['categories']:
        result[category['id']] = count
        count += 1
    return result

def getLocalMapAxis(gridMin, gridMax, bboxMin, bboxMax):
    """
    Returns the portion of overlap on 1D.
    """
    if gridMin <= bboxMin <= bboxMax <= gridMax:
        # Inclusion case
        return 1.0
    elif gridMin <= bboxMin <= gridMax <= bboxMax:
        # Split case
        return (gridMax - bboxMin) / (gridMax - gridMin)
    elif gridMin <= gridMax <= bboxMin <= bboxMax:
        # No overlap case
        return 0.0
    elif bboxMin <= bboxMax <= gridMin <= gridMax:
        # No overlap case
        return 0.0
    elif bboxMin <= gridMin <= bboxMax <= gridMax:
        # Split case
        return (bboxMax - gridMin) / (gridMax - gridMin)
    elif bboxMin <= gridMin <= gridMax <= bboxMax:
        # Inclusion case
        return 1.0

def getLocalMap(bbox, gridSize, normalize=True):
    """
    Get a weighted average of the convolution feature given bounding box.
    For VGG, featureGridSize is 14.
    bbox is (xCenter, yCenter, width, height) normalized.
    Outputs (14, 14) weight map.
    """
    N = gridSize
    result = np.zeros((N, N), dtype='float32')
    for i in range(N): # Looping over rows.
        bboxxMin = bbox[0] - bbox[2] / 2.0
        bboxxMax = bbox[0] + bbox[2] / 2.0
        bboxyMin = bbox[1] - bbox[3] / 2.0
        bboxyMax = bbox[1] + bbox[3] / 2.0
        for j in range(N): # Looping over columns.
            gridxMin = j / float(N)
            gridxMax = (j + 1) / float(N)
            gridyMin = i / float(N)
            gridyMax = (i + 1) / float(N)
            widthRatio = \
                getLocalMapAxis(gridxMin, gridxMax, bboxxMin, bboxxMax)
            heightRatio = \
                getLocalMapAxis(gridyMin, gridyMax, bboxyMin, bboxyMax)
            result[j, i] = widthRatio * heightRatio
    # Normalize local map
    if normalize:
        result /= np.sum(result)
    return result

def testGetLocalMap():
    """
    Result should be:
    [[ 0.0625  0.25    0.0625]
     [ 0.25    1.      0.25  ]
     [ 0.0625  0.25    0.0625]]
    """
    bbox = (0.5, 0.5, 0.5, 0.5)
    N = 3
    print getLocalMap(bbox, N, normalize=False) 

def extractBbox(annotations, imageDict, catDict):
    """
    Returns a dictionary.
    Key is image ID.
    Value is a dictionary, with key to be category ID, 
    value to be a list of [x, y, w, h] that are normalized,
    where (x, y) is the center coordinate of the bounding box, 
    w, h are the width and height of the bounding box.
    """
    result = {}
    for i, annotation in enumerate(annotations['annotations']):
        x = annotation['bbox'][0]
        y = annotation['bbox'][1]
        w = annotation['bbox'][2]
        h = annotation['bbox'][3]
        imageId = annotation['image_id']
        catId = catDict[annotation['category_id']]
        image = imageDict[imageId]
        x /= image['width']
        y /= image['height']
        w /= image['width']
        h /= image['height']
        xCenter = x + w / 2.0
        yCenter = y + h / 2.0
        if result.has_key(imageId):
            if result[imageId].has_key(catId):
                result[imageId][catId].append((xCenter, yCenter, w, h))
            else:
                result[imageId][catId] = [(xCenter, yCenter, w, h)]
        else:
            result[imageId] = {catId: [(xCenter, yCenter, w, h)]}
    return result

def calcSparseMeanStd(localFeatureMap, gridSize):
    """
    Calculate feature dimension mean and std for sparse feature map.
    localFeatureMap has shape (N, D x Fy x Fx), scipy sparse csr matrix format.
    F is the grid size. Fy is rows, Fx is columns.
    D is the feature dimension.
    N is number of examples.
    Outputs D dimension vectors for both mean and std.
    """
    localFeatureDim = localFeatureMap.shape[1] / (gridSize ** 2)
    featureMean = np.zeros(localFeatureDim)
    print 'Calculating mean'
    progress = 0
    for n in range(localFeatureMap.shape[0]):
        while n / float(localFeatureMap.shape[0]) > progress / 80.0:
            sys.stdout.write('.')
            sys.stdout.flush()
            progress += 1
        featureMean += np.sum(np.array(localFeatureMap[n].todense())\
            .reshape(localFeatureDim, gridSize ** 2), axis=1) / (gridSize ** 2)
    featureMean /= float(localFeatureMap.shape[0])
    print
    print 'Calculating std'
    featureStd = np.zeros(localFeatureDim)
    progress = 0
    for n in range(localFeatureMap.shape[0]):
        while n / float(localFeatureMap.shape[0]) > progress / 80.0:
            sys.stdout.write('.')
            sys.stdout.flush()
            progress += 1
        featureStd += np.sum(
            np.power(
            np.array(localFeatureMap[n].todense())\
            .reshape(localFeatureDim, gridSize ** 2) - \
            featureMean.reshape(localFeatureDim, 1), 2), 
            axis=1)
    featureStd /= float(localFeatureMap.shape[0]) * gridSize ** 2
    featureStd = np.sqrt(featureStd)
    print
    return featureMean, featureStd

def assemble(originalInput, imageIdDict, bboxDict, numBboxPerImage, gridSize,
    buildLocalFeature=False, localFeatureMap=None, normalizeLocalFeature=False,
    localFeatureMean=None, localFeatureStd=None):
    """
    Assemble the new detection sequence input.
    Takes original input that is like 
        [imageId, wordId_1, wordId_2, ..., wordId_n] for each example.
    imgageIdDict maps the 1-based ID to original MS-COCO ID.
    bboxDict maps original MS-COCO ID to the bounding boxes.
    numBboxPerImage controls the fixed length detection sequence.
    Outputs [imageId, detectionCategoryId_1, detectionBboxXcenter_1, 
        detectionBboxYcenter_1, detectionBboxWidth_1, detectionBboxHeight_1,
        detectionLocalFeature_1, detectionCategoryId_2, ..., 
        detectionLocalFeature_m, wordId_1, wordId_2, ..., wordId_n] 
        for each example.
    """
    if buildLocalFeature:
        localFeatureDim = localFeatureMap.shape[1] / (gridSize ** 2)
    else:
        localFeatureDim = 0

    detectionItemLength = 5 + localFeatureDim
    newInput = np.zeros(
        (originalInput.shape[0], 
        originalInput.shape[1] + numBboxPerImage * detectionItemLength, 
        1), dtype='float32')
    print originalInput.shape
    print newInput.shape
    numSkip = 0
    progress = 0
    for n in range(originalInput.shape[0]):
        while n / float(originalInput.shape[0]) > progress / 80.0:
            sys.stdout.write('.')
            sys.stdout.flush()
            progress += 1
        imageId = originalInput[n, 0, 0] - 1
        originalImageId = int(imageIdDict[imageId])
        if originalImageId not in bboxDict:
            numSkip += 1
            continue
        bboxset = bboxDict[originalImageId]
        bboxcount = 0
        skip = False
        imageLocalFeatureMap = \
            localFeatureMap[imageId].todense()\
            .reshape(localFeatureDim, gridSize ** 2)
        for catId in bboxset.keys():
            if skip: 
                break
            for bbox in bboxset[catId]:
                newInput[n, bboxcount * detectionItemLength + 1, 0] = catId
                newInput[n, bboxcount * detectionItemLength + 2, 0] = bbox[0]
                newInput[n, bboxcount * detectionItemLength + 3, 0] = bbox[1]
                newInput[n, bboxcount * detectionItemLength + 4, 0] = bbox[2]
                newInput[n, bboxcount * detectionItemLength + 5, 0] = bbox[3]
                if buildLocalFeature:
                    weightMap = getLocalMap(bbox, gridSize=gridSize)
                    bboxLocalFeatureMap = \
                        np.dot(
                        imageLocalFeatureMap,
                        weightMap.reshape(gridSize ** 2)).reshape(
                        localFeatureDim)
                    if normalizeLocalFeature:
                        bboxLocalFeatureMap -= localFeatureMean
                        bboxLocalFeatureMap /= localFeatureStd
                    newInput[n, bboxcount * detectionItemLength + 6 : \
                        bboxcount * detectionItemLength + \
                        6 + localFeatureDim, 0] = \
                        bboxLocalFeatureMap
                bboxcount += 1
                if bboxcount == numBboxPerImage:
                    skip = True
                    break
    newInput[:, 0, :] = originalInput[:, 0, :]
    newInput[:, numBboxPerImage * detectionItemLength + 1:, :] = \
        originalInput[:, 1:, :]
    print
    print numSkip, ' of ', originalInput.shape[0], 'images are missing bbox'
    return newInput

def testCalcSparseMeanStd():
    M = np.random.rand(100, 160)
    Msparse = scipy.sparse.csr_matrix(M)
    mean1, std1 = calcSparseMeanStd(Msparse, 2)
    Mreshape = M.reshape(400, 40)
    mean2 = np.mean(Mreshape, axis=0)
    std2 = np.std(Mreshape, axis=0)
    print mean1 / mean2
    print std1 /std2

if __name__ == '__main__':
    # testGetLocalMap()
    # testCalcSparseMeanStd()
    # import sys
    # sys.exit(0)
    mscocoAnnotationTrainFilename = \
        '/ais/gobi3/datasets/mscoco/annotations/instances_train2014.json'
    mscocoAnnotationValidFilename = \
        '/ais/gobi3/datasets/mscoco/annotations/instances_val2014.json'
    cocoqaFolder = \
        '../data/cocoqa/'
    outputFolder = \
        '/ais/gobi3/u/mren/data/cocoqa-detection-convfeatnorm/'
    cocoqaImgDictFile = \
        '../data/cocoqa/imgid_dict.pkl'
    mscocoImgFeatFile = \
        '/ais/gobi3/u/mren/data/cocoqa-full/hidden_oxford.h5'
    convLayer = 'hidden5_4_conv'
    buildLocalFeature = True
    gridSize = 14
    normalizeLocalFeature = True

    print 'Building to', outputFolder
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    
    print 'Reading MS-COCO'
    f = open(mscocoAnnotationTrainFilename)
    fstr = f.read()
    annotationsTrain = json.loads(fstr)
    f.close()
    f = open(mscocoAnnotationValidFilename)
    fstr = f.read()
    annotationsValid = json.loads(fstr)
    
    print 'Building image dictionary'
    imageDictTrain = buildImageDict(annotationsTrain)
    imageDictValid = buildImageDict(annotationsValid)
    
    print 'Building category dictionary'
    catDict = buildCategoryDict(annotationsTrain)
    
    print 'Extracting bounding boxes'
    numBboxPerImage = 10
    bboxDictTrain = extractBbox(
        annotationsTrain, imageDictTrain, catDict)
    bboxDictValid = extractBbox(
        annotationsValid, imageDictValid, catDict)
    imageIdDict = pkl.load(open(cocoqaImgDictFile))
    
    print 'Loading conv features'
    mscocoImgFeat = h5py.File(mscocoImgFeatFile, 'r')
    convShape = mscocoImgFeat[convLayer + '_shape'][:]
    convData = mscocoImgFeat[convLayer + '_data'][:]
    convInd = mscocoImgFeat[convLayer + '_indices'][:]
    convPtr = mscocoImgFeat[convLayer + '_indptr'][:]
    localFeatureMap = scipy.sparse.csr_matrix(
        (convData, convInd, convPtr), shape=convShape)

    print 'Building training data'
    trainData = np.load(os.path.join(cocoqaFolder, 'train.npy'))
    trainInput = trainData[0]
    if normalizeLocalFeature:
        print 'Calculating mean/std on training conv features'
        localFeatureMean, localFeatureStd = \
            calcSparseMeanStd(localFeatureMap[0:trainInput.shape[0]], gridSize)
    else:
        localFeatureMean = None
        localFeatureStd = None

    newTrainInput = assemble(trainInput, imageIdDict, bboxDictTrain, 
        gridSize=gridSize, numBboxPerImage=numBboxPerImage, 
        buildLocalFeature=buildLocalFeature, localFeatureMap=localFeatureMap,
        normalizeLocalFeature=normalizeLocalFeature,
        localFeatureMean=localFeatureMean, localFeatureStd=localFeatureStd)

    print 'Building validation data'
    validData = np.load(os.path.join(cocoqaFolder, 'valid.npy'))
    validInput = validData[0]
    newValidInput = assemble(validInput, imageIdDict, bboxDictTrain,
        gridSize=gridSize, numBboxPerImage=numBboxPerImage, 
        buildLocalFeature=buildLocalFeature, localFeatureMap=localFeatureMap,
        normalizeLocalFeature=normalizeLocalFeature,
        localFeatureMean=localFeatureMean, localFeatureStd=localFeatureStd)

    print 'Building testing data'
    testData = np.load(os.path.join(cocoqaFolder, 'test.npy'))
    testInput = testData[0]
    newTestInput = assemble(testInput, imageIdDict, bboxDictValid,
        gridSize=gridSize, numBboxPerImage=numBboxPerImage, 
        buildLocalFeature=buildLocalFeature, localFeatureMap=localFeatureMap,
        normalizeLocalFeature=normalizeLocalFeature,
        localFeatureMean=localFeatureMean, localFeatureStd=localFeatureStd)

    newTrainData = np.array(
        (newTrainInput, trainData[1], 0), dtype='object')
    newValidData = np.array(
        (newValidInput, validData[1], 0), dtype='object')
    newTestData = np.array(
        (newTestInput, testData[1], 0), dtype='object')
    np.save(os.path.join(outputFolder, 'train.npy'), newTrainData)
    np.save(os.path.join(outputFolder, 'valid.npy'), newValidData)
    np.save(os.path.join(outputFolder, 'test.npy'), newTestData)

