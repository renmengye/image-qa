import json
import os
import cPickle as pkl
import sys
import re

# [u'info', u'licenses', u'images', u'type', u'annotations', u'categories']
import cv2
import numpy as np

imgidTrainFilename = '../../../data/mscoco/train/image_list.txt'
imgidValidFilename = '../../../data/mscoco/valid/image_list.txt'

def buildImgIdDict(imgidTrainFilename, imgidValidFilename):
    with open(imgidTrainFilename) as f:
        lines = f.readlines()
    trainStart = 0
    trainEnd = len(lines) * 9 / 10
    validStart = trainEnd
    validEnd = len(lines)
    with open(imgidValidFilename) as f:
        lines.extend(f.readlines())
    testStart = validEnd
    testEnd = len(lines)

    cocoImgIdRegex = 'COCO_((train)|(val))2014_0*(?P<imgid>[1-9][0-9]*)'
    # Mark for train/valid/test.
    imgidDict = {} 
    # Reindex the image, 1-based.
    imgidDict2 = {}
    # Reverse dict for image, 0-based.
    imgidDict3 = []
    # Path for jpg file
    imgPathDict = {}

    # Separate image ids into train-valid-test
    # 0 for train, 1 for valid, 2 for test.
    for i in range(trainStart, trainEnd):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 0
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)
        imgPathDict[imgid] = lines[i][:-1]

    for i in range(validStart, validEnd):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 1
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)
        imgPathDict[imgid] = lines[i][:-1]

    for i in range(testStart, testEnd):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 2
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)
        imgPathDict[imgid] = lines[i][:-1]

    return imgidDict, imgidDict2, imgidDict3, imgPathDict

def buildImgDict(imgList, imgidDict):
    imgDict = {}
    count = 1
    for img in imgList:
        if not imgDict.has_key(img['id']):
            img['new_id'] = imgidDict[str(img['id'])]
            imgDict[img['id']] = img
            count += 1
    return imgDict

def buildCatDict(catList):
    catDict = {}
    for cat in catList:
        if not catDict.has_key(cat['id']):
            catDict[cat['id']] = cat
    return catDict

def polyFill(img, segmentation):
    polys = []
    for seg in segmentation:
        N = len(seg)
        # print 'Seg:', seg
        # print 'SegY:', seg[1:N:2]
        # print 'SegX:', seg[0:N:2]
        poly = np.concatenate(
            (np.array(seg[0:N:2]).reshape(N/2, 1), 
             np.array(seg[1:N:2]).reshape(N/2, 1)), axis=-1).astype('int')
        polys.append(poly)
    pts = np.array(polys, dtype='int')
    cv2.fillPoly(img=img, pts=pts, color=(1, 1, 1))

def countPts(img):
    if len(img.shape) == 3:
        count = np.sum(img[:, :, 0])
    elif len(img.shape) == 2:
        count = np.sum(img)
    return count

def distributeAtt(numX, numY, filledPolyImg):
    width = filledPolyImg.shape[1]
    height = filledPolyImg.shape[0]
    w = width / numX
    h = height / numY
    totalCount = 0
    count = np.zeros((numY, numX), dtype='float32')
    for x in range(numX):
        for y in range(numY):
            startX = w * x
            startY = h * y
            if x < numX - 1:
                endX = w * (x + 1)
            else:
                endX = width
            if y < numY - 1:
                endY = h * (y + 1)
            else:
                endY = height
            count[y, x] = \
                countPts(filledPolyImg[startY:endY,startX:endX])
            totalCount += count[y, x]
    count /= totalCount
    return count

def gatherAttention(trainJsonFilename, validJsonFilename):
    with open(trainJsonFilename) as f:
        insttxt = f.read()
    instances = json.loads(insttxt)
    annotations = instances['annotations']
    images = instances['images']
    categories = instances['categories']
    with open(validJsonFilename) as f:
        insttxt = f.read()
    instances = json.loads(insttxt)
    annotations.extend(instances['annotations'])
    images.extend(instances['images'])
    categories.extend(instances['categories'])
    
    splitDict, imgidDict, imgidIdict, imgPathDict = \
        buildImgIdDict(imgidTrainFilename, imgidValidFilename)
    catDict = buildCatDict(instances['categories'])
    imgDict = buildImgDict(instances['images'], imgidDict)

    trainInput = []
    trainTarget = []
    validInput = []
    validTarget = []
    testInput = []
    testTarget = []

    L = len(instances['annotations'])
    print 'Total instances:', L
    for i in range(L):
        if i % 1000 == 0: print i
        ann = instances['annotations'][i]
        seg = ann['segmentation']
        catId = ann['category_id']
        img = imgDict[ann['image_id']]
        imgid = str(img['id'])
        width = int(img['width'])
        height = int(img['height'])

        # imgMat = cv2.imread(imgPathDict[imgid])
        zeroMat = np.zeros((height, width, 3), dtype='int')
        # polyFill(imgMat, seg)
        polyFill(zeroMat, seg)
        # cv2.imwrite('../%s_%s.jpg' % \
        #     (i, catDict[catId]['name']), 
        #     imgMat)
        att = distributeAtt(14, 14, zeroMat)
        # print att

        # M = np.zeros((14, 14)) + 255
        # att2 = (att * 5000).astype('int')
        # att2 = np.minimum(att2, M)
        # print att2

        # attUpsample = cv2.resize(att2, (height, width))
        # cv2.imwrite('../%s_%s_att.jpg' % \
        #     (i, catDict[catId]['name']),
        #     attUpsample)


        # Assemble input data
        # imgId, catId
        inputData = [img['new_id'], catId]

        # Assemble target data
        # attention, flattened
        att = att.reshape(196)
        targetData = att

        s = splitDict[imgid]
        if s == 0:
            trainInput.append(inputData)
            trainTarget.append(targetData)
        elif s == 1:
            validInput.append(inputData)
            validTarget.append(targetData)
        elif s == 2:
            testInput.append(inputData)
            testTarget.append(targetData)
    trainData = np.array((trainInput, trainTarget, 0), dtype='object')
    validData = np.array((validInput, validTarget, 0), dtype='object')
    testData = np.array((testInput, testTarget, 0), dtype='object')

    return trainData, validData, testData

# To retrive image ID and url, get caption['images'][i]['id'] and caption['images'][i]['url']
if __name__ == '__main__':
    outputFolder = '../data/coco-att'
    for i, flag in enumerate(sys.argv):
        if flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    trainJsonFilename = \
        '../../../data/mscoco/%s/instances.json' % 'train'
    validtrainJsonFilename = \
        '../../../data/mscoco/%s/instances.json' % 'valid'
    trainData, validData, testData = \
        gatherAttention(trainJsonFilename, validtrainJsonFilename)
    np.save(os.path.join(outputFolder, 'train.npy'), trainData)
    np.save(os.path.join(outputFolder, 'valid.npy'), validData)
    np.save(os.path.join(outputFolder, 'test.npy'), testData)