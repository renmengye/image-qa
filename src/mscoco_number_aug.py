import cPickle as pkl
import os
import pattern.en
import numpy as np
import shutil

import mscoco_extract_instances as ext
import imageqa_test as it
import prep

countFile = '../data/coco-count/count.pkl'
numberDatasetFolder = '../data/cocoqa-number-rv'
augmentDatasetFolder = '../data/cocoqa-number-rv-aug'
categoriesFile = '/ais/gobi3/u/mren/data/mscoco/categories.pkl'
foundSingular = 0
foundPlural = 0
foundBoth = 0
notFound = 0
number = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten']

def lookupWord(catname):
    global foundSingular
    global foundPlural
    global foundBoth
    global notFound
    catnames = pattern.en.pluralize(catname)
    result = []
    if questionDict.has_key(catname):
        print 'Found %s: %d' % (catname, questionDict[catname])
        result.append(questionDict[catname])
        foundSingular += 1
    else:
        print 'Not found %s' % catname
        result.append(0)

    if questionDict.has_key(catnames):
        print 'Found plural %s: %d' % (catnames, questionDict[catnames])
        result.append(questionDict[catnames])
        foundPlural += 1
    else:
        print 'Not found %s' % catnames
        result.append(0)

    if result[0] > 0 and result[1] > 0:
        foundBoth += 1

    if result[0] == 0 and result[1] == 0:
        notFound += 1
    return result

if __name__ == '__main__':
    if not os.path.exists(augmentDatasetFolder):
        os.makedirs(augmentDatasetFolder)

    with open(countFile) as f:
        countDict = pkl.load(f)

    # annotations, images, categories = \
    #     ext.parse(ext.trainJsonFilename, ext.validJsonFilenam)
    splitDict, imgidDict, imgidIdict, imgPathDict = \
        ext.buildImgIdDict(ext.imgidTrainFilename, ext.imgidValidFilename)
    # catDict = ext.buildCatDict(categories)
    # imgDict = ext.buildImgDict(images, imgidDict)

    if os.path.exists(categoriesFile):
        with open(categoriesFile, 'rb') as f:
            categories = pkl.load(f)
            catDict = ext.buildCatDict(categories)
    else:
        annotations, images, categories = \
            ext.parse(ext.trainJsonFilename, ext.validJsonFilename)
        catDict = ext.buildCatDict(categories)
        with open(categoriesFile, 'wb') as f:
            pkl.dump(categories, f)
     
    data = it.loadDataset(numberDatasetFolder)
    questionDict = data['questionDict']
    questionIdict = data['questionIdict']
    catToWordDict = {}
    
    total = 0
    for cat in categories:
        catname = cat['name']
        catId = cat['id']
        if not catToWordDict.has_key(catId):
            print catname
            if ' ' in catname:
                catname = catname.split(' ')[-1]
            catToWordDict[catId] = lookupWord(catname)
            total += 1
    
    print 'Found singular', foundSingular
    print 'Found plural', foundPlural
    print 'Found both', foundBoth
    print 'Not found', notFound
    print 'Total', total
    
    questions = []
    answers = []
    imgIds = []

    for imgId in countDict.iterkeys():
        if splitDict[imgId] != 0:
            continue
        imgNewId = imgidDict[imgId]
        for catId in countDict[imgId].iterkeys():
            catname = catDict[catId]['name']
            count = countDict[imgId][catId]
            wordId = 0
            if count > 1 and count <= 10:
                if catToWordDict[catId][1] > 0:
                    wordId = catToWordDict[catId][1]
                elif catToWordDict[catId][0] > 0:
                    wordId = catToWordDict[catId][0]
                else:
                    continue
            elif count == 1:
                if catToWordDict[catId][0] > 0:
                    wordId = catToWordDict[catId][0]
                elif catToWordDict[catId][1] > 0:
                    wordId = catToWordDict[catId][1]
                else:
                    continue
            else:
                continue
            word = questionIdict[wordId - 1]
            verb = 'is' if count == 1 else 'are'
            question = 'how many %s %s there' % (word, verb)
            print question
            questions.append(question)
            answer = number[count]
            print answer
            answers.append(number[count])
            imgIds.append(imgNewId)

    maxlen = data['trainData'][0].shape[1] - 1
    # survivors = removeQuestions(answers
    # questions = np.array(questions, dtype='object')
    trainInput = prep.combine(\
        prep.lookupQID(questions, data['questionDict'], maxlen), imgIds)
    trainTarget = prep.lookupAnsID(answers, data['ansDict'])
    print data['trainData'][0].shape
    print trainInput.shape
    trainInputAll = np.concatenate(
        (data['trainData'][0], trainInput), axis=0)
    trainTargetAll = np.concatenate(
        (data['trainData'][1], trainTarget), axis=0)
    np.save(os.path.join(augmentDatasetFolder, 'train.npy'),
        np.array((trainInputAll, trainTargetAll, 0), dtype='object'))
    shutil.copy(
        os.path.join(numberDatasetFolder, 'valid.npy'),
        os.path.join(augmentDatasetFolder, 'valid.npy'))
    shutil.copy(
        os.path.join(numberDatasetFolder, 'test.npy'),
        os.path.join(augmentDatasetFolder, 'test.npy'))
    shutil.copy(
        os.path.join(numberDatasetFolder, 'test-qtype.npy'),
        os.path.join(augmentDatasetFolder, 'test-qtype.npy'))
    shutil.copy(
        os.path.join(numberDatasetFolder, 'vocab-dict.npy'),
        os.path.join(augmentDatasetFolder, 'vocab-dict.npy'))

