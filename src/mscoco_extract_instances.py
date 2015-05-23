import json
import os
import cPickle as pkl
import sys
import re

# [u'info', u'licenses', u'images', u'type', u'annotations', u'categories']
import cv2
import numpy as np

if len(sys.argv) < 2:
    dataset = 'train'
elif sys.argv[1] == '-valid':
    dataset = 'valid'
folder = '../../../data/mscoco/%s' % dataset
jsonFilename = '%s/instances.json' % (folder)
imgidTrainFilename = '../../../data/mscoco/train/image_list.txt'
imgidValidFilename = '../../../data/mscoco/valid/image_list.txt'

def buildImgIdDict():
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

def polyFill(img, width, height, segmentation):
    polys = []
    for seg in segmentation:
        N = len(seg)
        print 'Seg:', seg
        print 'SegY:', seg[1:N:2]
        print 'SegX:', seg[0:N:2]
        poly = np.concatenate(
            (np.array(seg[1:N:2]).reshape(N/2, 1), 
             np.array(seg[0:N:2]).reshape(N/2, 1)), axis=-1).astype(int)
        print 'Poly', poly, poly.shape
        polys.append(poly)
    cv2.fillPoly(img=img, pts=np.array(polys), color=(1, 1, 1))

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

# To retrive image ID and url, get caption['images'][i]['id'] and caption['images'][i]['url']
if __name__ == '__main__':
    with open(jsonFilename) as f:
        insttxt = f.read()
    instances = json.loads(insttxt)
    #L = len(instances['annotations'])
    L = 10
    splitDict, imgidDict, imgidIdict, imgPathDict = buildImgIdDict()
    catDict = buildCatDict(instances['categories'])
    imgDict = buildImgDict(instances['images'], imgidDict)
    inputData = []
    targetData = []

    for i in range(L):
        ann = instances['annotations'][i]
        seg = ann['segmentation']
        catId = ann['category_id']
        img = imgDict[ann['image_id']]
        width = img['width']
        height = img['height']

        #img = np.zeros((height, width, 3))
        imgMat = cv2.imread(imgPathDict[str(img['id'])])
        filledPolyImg = polyFill(imgMat, width, height, seg)
        cv2.imwrite('../%s_%s.jpg' % \
            (L, catDict[catId]['name']), 
            imgMat)
        att = distributeAtt(14, 14, filledPolyImg)
        att = att.reshape(196)

        # Assemble train data
        # imgId, catId
        # Need to check if the new id is consistaent with our previous indexing.
        # Notice category ID is 1-90 but only 80 in total.
        inputData.append([img['new_id'], catId])
        
        # Assemble target data
        # attention, flattened
        targetData.append(att)