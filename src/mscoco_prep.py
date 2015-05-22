import re
import os
import gc
import cPickle as pkl
import numpy as np
import h5py
import sys
import scipy.sparse
import calculate_wups
import hist
import prep
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

imgidTrainFilename = '../../../data/mscoco/train/image_list.txt'
imgidValidFilename = '../../../data/mscoco/valid/image_list.txt'
qaTrainFilename = '../../../data/mscoco/train/qa.pkl'
qaValidFilename = '../../../data/mscoco/valid/qa.pkl'
imgHidFeatTrainFilename = '/ais/gobi3/u/mren/data/mscoco/hidden_oxford_train.h5'
imgHidFeatValidFilename = '/ais/gobi3/u/mren/data/mscoco/hidden_oxford_valid.h5'

def removeQuestions(
                    answers, 
                    lowerBound, 
                    upperBound,
                    upperUpperBound,
                    mustIncludeAns={}):
    """
    Removes questions with answer appearing less than N times.
    Probability function to decide whether or not to enroll an 
    answer (remove too frequent answers).
    Returns a list of enrolled IDs.
    Parameters:
    answers: list, answers of all questions before enrolment.
    lowerBound: int, answers appearing less than the lower bound will
                be rejected.
    upperBound: int, answers above the upper bound will be rejected
                according to some heuristics.
    upperUpperBound: half life of the exponential decay. Frequency at which
                answers will be rejected at probability 0.5.
    mustIncludeAns: dictionary, a set of answers that must be considered
                without lower bound constraints.

    """
    answerdict, answeridict, answerfreq = prep.buildDict(answers, 0)
    random = np.random.RandomState(2)
    # Ongoing frequency count
    answerfreq2 = []
    survivor = []
    for item in answerfreq:
        answerfreq2.append(0)
    for i, ans in enumerate(answers):
        if answerfreq[answerdict[ans]] < lowerBound and \
            not mustIncludeAns.has_key(ans):
            continue
        else:
            if answerfreq2[answerdict[ans]] <= upperBound:
                survivor.append(i)
                answerfreq2[answerdict[ans]] += 1
            else:
                # Exponential distribution
                prob = np.exp(-(answerfreq2[answerdict[ans]] - \
                    upperBound) / float(upperUpperBound))
                r = random.uniform(0, 1, [1])
                if r < prob:
                    survivor.append(i)
                    answerfreq2[answerdict[ans]] += 1
    return survivor

def synonymDetect(iansdict, output=None):
    """
    Look for synonyms using WUPS measure
    """
    wups_dict = {}
    for a in range(len(iansdict)):
        for b in range(a + 1, len(iansdict)):
            ans1 = iansdict[a]
            ans2 = iansdict[b]
            score = calculate_wups.wup_measure(ans1, ans2, similarity_threshold=0)
            wups_dict[ans1 + '_' + ans2] = score
            print a, ans1, b, ans2, score
    keys = wups_dict.keys()
    sorted_keys = sorted(keys, key=lambda k: wups_dict[k], reverse=True)
    for i in range(500):
        print '%s %.4f' % (sorted_keys[i], wups_dict[sorted_keys[i]])

    if output is not None:
        with open(output, 'w') as f:
            for i in range(len(sorted_keys)):
                f.write('%s %.4f\n' % (sorted_keys[i], wups_dict[sorted_keys[i]]))

def compressFeatAllIter(feat, num=0, bat=10000):
    if num == 0:
        num = feat.shape[0]
    numBat = int(np.ceil(num / float(bat)))
    featSparse = []
    for i in range(numBat):
        start = bat * i
        end = min(bat * (i + 1), num)
        featCopy = feat[start : end]
        featSparseTmp = scipy.sparse.csr_matrix(featCopy)
        featSparse.append(featSparseTmp)
    return scipy.sparse.vstack(featSparse, format='csr')

def buildImageFeature(
                    trainFilename, 
                    validFilename,
                    outFilename,
                    numTrain=0, 
                    numValid=0, 
                    numTest=0, 
                    sparse=True):
    """
    Build image features

    Parameters:
    trainFilename: H5 file produced by the CNN on the COCO training set.
    validFilename: H5 file produced by the CNN on the COOC validation set.
    outFilename: COCO-QA image features H5 file name.
    numTrain: Number of training images. If 0, then build entire training set.
    numValid: Number of validation images. If 0, then build entire training set.
    numTest: Number of testing images. If 0, then build entire test set.
    sparse: Whether output a sparse matrix.
    """
    print 'Building image features'
    imgHidFeatTrain = h5py.File(imgHidFeatTrainFilename)
    imgHidFeatValid = h5py.File(imgHidFeatValidFilename)
    imgOutFile = h5py.File(imgHidFeatOutFilename, 'w')
    layers = ['hidden7', 'hidden6', 'hidden5_maxpool', 'hidden5_4_conv']
    for name in layers:
        gc.collect()
        if sparse:
            hidFeatTrainSparse = compressFeatAllIter(
                        imgHidFeatTrain[name], num=numTrain+numValid)
            hidFeatValidSparse = compressFeatAllIter(
                        imgHidFeatValid[name], num=numTest)
            hidFeatSparse = scipy.sparse.vstack(
                    (hidFeatTrainSparse, hidFeatValidSparse), format='csr')
            imgOutFile[name + '_shape'] = hidFeatSparse._shape
            imgOutFile[name + '_data'] = hidFeatSparse.data
            imgOutFile[name + '_indices'] = hidFeatSparse.indices
            imgOutFile[name + '_indptr'] = hidFeatSparse.indptr
            print name, hidFeatSparse._shape
        else:
            if numTrain == 0 or numValid == 0:
                hidFeatTrain = imgHidFeatTrain[name][:]
            else:
                hidFeatTrain = imgHidFeatTrain[name][0 : numTrain + numValid]
            if numTest == 0:
                hidFeatValid = imgHidFeatValid[name][:]
            else:
                hidFeatValid = imgHidFeatValid[name][0 : numTest]
            hidFeat = np.concatenate((hidFeatTrain, hidFeatValid), axis=0)
            imgOutFile[name] = hidFeat
            print name, hidFeat.shape

    if numTrain == 0:
        hidden7Train = imgHidFeatTrain['hidden7'][:]
    else:
        hidden7Train = imgHidFeatTrain['hidden7'][0 : numTrain]
    mean = np.mean(hidden7Train, axis=0)
    std = np.std(hidden7Train, axis=0)
    for i in range(0, std.shape[0]):
        if std[i] == 0:
            std[i] = 1.0
    if not sparse:
        hidden7Ms = (imgOutFile['hidden7'][:] - mean) / std
        imgOutFile['hidden7_ms'] = hidden7Ms.astype('float32')
    imgOutFile['hidden7_mean'] = mean.astype('float32')
    imgOutFile['hidden7_std'] = std.astype('float32')

if __name__ == '__main__':
    """
    Assemble COCO-QA dataset.
    Make sure you have already run parsing and question generation.
    This program only assembles generated questions.

    Usage:
    mscoco_prep.py  [-toy] Build the toy set, default is full dataset
                    [-image] Build image features, default is not building
                    [-object] Object type question only
                    [-number] Number type question only
                    [-color] Color type question only
                    [-location] Location type question only
                    [-len {length}] Maximum length/timespan
                    [-noreject] No rejection
                    [-o[utput] {outputFolder}] Output folder, 
                                default '../data/cocoqa-toy' or 
                                        '../data/cocoqa-full'
                    [-imgo[utput] {imageFeatOutputFolder}]
                                Image feature output folder,
                                default '/ais/gobi3/u/mren/data/cocoqa-full'
    """
    buildToy = False
    buildImage = False
    buildObject = True
    buildNumber = True
    buildColor = True
    buildLocation = True
    maxlen = -1
    outputFolder = None
    imgHidFeatOutputFolder = None
    reject = True

    for i in range(len(sys.argv)):
        flag = sys.argv[i]
        if flag == '-toy':
            buildToy = True
        elif flag == '-image':
            buildImage = True
        elif flag == '-object':
            print 'Only building object questions'
            buildObject = True
            buildNumber = False
            buildColor = False
            buildLocation = False
        elif flag == '-number':
            print 'Only building number questions'
            buildObject = False
            buildNumber = True
            buildColor = False
            buildLocation = False
        elif flag == '-color':
            print 'Only building color questions'
            buildObject = False
            buildNumber = False
            buildColor = True
            buildLocation = False
        elif flag == '-location':
            print 'Only building location questions'
            buildObject = False
            buildNumber = False
            buildColor = False
            buildLocation = True
        elif flag == '-noreject':
            reject = False
        elif flag == '-len':
            maxlen = int(sys.argv[i + 1])
        elif flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
        elif flag == '-imgo' or flag == '-imgoutput':
            imgHidFeatOutputFolder = sys.argv[i + 1]
    buildType = [buildObject, buildNumber, buildColor, buildLocation]

    if buildToy:
        # Build toy dataset
        print 'Building toy dataset'
        if outputFolder is None:
            outputFolder = '../data/cocoqa-toy'
        if imgHidFeatOutputFolder is None:
            imgHidFeatOutputFolder = '/ais/gobi3/u/mren/data/cocoqa-toy'
        if not os.path.exists(imgHidFeatOutputFolder):
            os.makedirs(imgHidFeatOutputFolder)
        imgHidFeatOutFilename = \
            os.path.join(imgHidFeatOutputFolder, 'hidden_oxford.h5')
        numTrain = 6000
        numValid = 1200
        numTest = 6000
        LB = 8
        UB = 150
        UUB = 300
        sparse = False
        with open(imgidTrainFilename) as f:
            lines = f.readlines()
        trainStart = 0
        trainEnd = numTrain
        validStart = trainEnd
        validEnd = validStart + numValid
        totalTrainLen = len(lines)
        with open(imgidValidFilename) as f:
            lines.extend(f.readlines())
        testStart = totalTrainLen 
        testEnd = testStart + numTest
    else:
        # Build full dataset
        print 'Building full dataset'
        if outputFolder is None:
            outputFolder = '../data/cocoqa-full/'
        if imgHidFeatOutputFolder is None:
            imgHidFeatOutputFolder = '/ais/gobi3/u/mren/data/cocoqa-full'
        if not os.path.exists(imgHidFeatOutputFolder):
            os.makedirs(imgHidFeatOutputFolder)
        imgHidFeatOutFilename = \
            os.path.join(imgHidFeatOutputFolder, 'hidden_oxford.h5')
        LB = 25
        UB = 350
        UUB = 700
        numTrain = 0
        numValid = 0
        numTest = 0
        sparse = True
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

    if buildImage:
        # Build image features.
        buildImageFeature(
            imgHidFeatTrainFilename,
            imgHidFeatValidFilename,
            imgHidFeatOutFilename,
            numTrain=numTrain,
            numValid=numValid,
            numTest=numTest,
            sparse=sparse)
    else:
        print 'Not building image features'

    print 'Will build to', outputFolder
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # Mark for train/valid/test.
    imgidDict = {} 
    # Reindex the image, 1-based.
    imgidDict2 = {}
    # Reverse dict for image, 0-based.
    imgidDict3 = []

    # Separate image ids into train-valid-test
    # 0 for train, 1 for valid, 2 for test.
    cocoImgIdRegex = 'COCO_((train)|(val))2014_0*(?P<imgid>[1-9][0-9]*)'
    for i in range(trainStart, trainEnd):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 0
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)

    for i in range(validStart, validEnd):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 1
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)

    for i in range(testStart, testEnd):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 2
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)

    with open(qaTrainFilename) as qaf:
        qaAll = pkl.load(qaf)
    with open(qaValidFilename) as qaf:
        qaAll.extend(pkl.load(qaf))

    print 'Total number of images', len(imgidDict)

    allQuestions = []
    allAnswers = []
    allImgIds = []
    allQuestionTypes = []

    for item in qaAll:
        allQuestions.append(item[0])
        allAnswers.append(item[1])
        allImgIds.append(item[2])
        allQuestionTypes.append(item[3])

    print 'Distribution Before Rejection'
    beforeWorddict, beforeIdict, beforeFreq = \
        prep.buildDict(allAnswers, 0, pr=True)

    print 'GUESS Before Rejection'
    prep.guessBaseline(allQuestions, allAnswers, allQuestionTypes)        

    # Shuffle the questions.
    r = np.random.RandomState(1)
    shuffle = r.permutation(len(allQuestions))
    allQuestions = np.array(allQuestions, dtype='object')[shuffle]
    allAnswers = np.array(allAnswers, dtype='object')[shuffle]
    allImgIds = np.array(allImgIds, dtype='object')[shuffle]
    allQuestionTypes = np.array(
        allQuestionTypes,dtype='int')[shuffle]

    if reject:
        # Truncate rare-common answers.
        print 'Rejecting rare-common answers'
        survivor = np.array(removeQuestions(
            answers=allAnswers, 
            lowerBound=LB, 
            upperBound=UB,
            upperUpperBound=UUB), dtype='int')
        allQuestions = allQuestions[survivor]
        allAnswers = allAnswers[survivor]
        allImgIds = allImgIds[survivor]
        allQuestionTypes = allQuestionTypes[survivor]
    else:
        print 'Not rejecting rare-common answers'

    # Shuffle the questions again.
    r = np.random.RandomState(2000)
    shuffle = r.permutation(len(allQuestions))
    allQuestions = allQuestions[shuffle]
    allAnswers = allAnswers[shuffle]
    allImgIds = allImgIds[shuffle]
    allQuestionTypes = allQuestionTypes[shuffle]

    print 'Distribution After Rejection'
    afterWorddict, afterIdict, afterFreq = \
        prep.buildDict(allAnswers, 0, pr=True)

    print 'GUESS After Rejection on All Data'
    prep.guessBaseline(allQuestions, allAnswers, allQuestionTypes)

    # Filter question types
    if not (buildObject and buildNumber and buildColor and buildLocation):
        # Now only build one type!
        if buildObject:
            typ = 0
        elif buildNumber:
            typ = 1
        elif buildColor:
            typ = 2
        elif buildLocation:
            typ = 3
        allQuestions = allQuestions[allQuestionTypes == typ]
        allAnswers = allAnswers[allQuestionTypes == typ]
        allImgIds = allImgIds[allQuestionTypes == typ]
        allQuestionTypes = allQuestionTypes[allQuestionTypes == typ]
    trainQuestions = []
    trainAnswers = []
    trainImgIds = []
    trainImgIdsRelease = []
    trainQuestionTypes = []
    validQuestions = []
    validAnswers = []
    validImgIds = []
    validImgIdsRelease = []
    validQuestionTypes = []
    testQuestions = []
    testAnswers = []
    testImgIds = []
    testQuestionTypes = []
    testImgIdsRelease = []

    # Separate dataset into train-valid-test.
    for item in zip(allQuestions, allAnswers, allImgIds, allQuestionTypes):
        imgid = item[2]
        if imgidDict.has_key(imgid):
            if imgidDict[imgid] == 0:
                trainQuestions.append(item[0][:-2])
                trainAnswers.append(item[1])
                trainImgIds.append(imgidDict2[imgid])
                trainImgIdsRelease.append(imgid)
                trainQuestionTypes.append(item[3])
            elif imgidDict[imgid] == 1:
                validQuestions.append(item[0][:-2])
                validAnswers.append(item[1])
                validImgIds.append(imgidDict2[imgid])
                validQuestionTypes.append(item[3])
                validImgIdsRelease.append(imgid)
            elif imgidDict[imgid] == 2:
                testQuestions.append(item[0][:-2])
                testAnswers.append(item[1])
                testImgIds.append(imgidDict2[imgid])
                testQuestionTypes.append(item[3])
                testImgIdsRelease.append(imgid)

    # Build statistics
    trainCount = np.zeros(4, dtype='int')
    validCount = np.zeros(4, dtype='int')
    testCount = np.zeros(4, dtype='int')
    
    for n in range(0, len(trainQuestions)):
        question = trainQuestions[n]
        trainCount[trainQuestionTypes[n]] += 1
    for n in range(0, len(validQuestions)):
        question = validQuestions[n]
        validCount[validQuestionTypes[n]] += 1
    for n in range(0, len(testQuestions)):
        question = testQuestions[n]
        testCount[testQuestionTypes[n]] += 1

    print 'Train Questions After Trunk: ', len(trainQuestions)
    print 'Train Question Dist: ', trainCount
    print 'Train Question Dist: ', \
            trainCount / float(len(trainQuestions))
    print 'Valid Questions After Trunk: ', len(validQuestions)
    print 'Valid Question Dist: ', validCount
    print 'Valid Question Dist: ', \
            validCount / float(len(validQuestions))

    trainValidQuestionsLen = len(trainQuestions) + len(validQuestions)
    print 'Train+Valid questions: ', trainValidQuestionsLen
    print 'Train+Valid Dist: ', trainCount + validCount
    print 'Trian+Valid Dist: ', \
            (trainCount + validCount) / float(trainValidQuestionsLen)

    print 'Test Questions After Trunk: ', len(testQuestions)
    print 'Test Question Dist: ', testCount
    print 'Test Question Dist: ', testCount / float(len(testQuestions))
    
    # Build dictionary based on training questions/answers.
    worddict, idict, wordfreq = prep.buildDict(trainQuestions, 1, pr=False)
    print 'Train answer distribution'
    ansdict, iansdict, _ = prep.buildDict(trainAnswers, 0, pr=True)
    print 'Valid answer distribution'
    prep.buildDict(validAnswers, 0, pr=True)
    print 'Test answer distribution'
    testAfterWorddict, testAfterIdict, testAfterFreq = \
        prep.buildDict(testAnswers, 0, pr=True)

    print 'GUESS After Rejection on Test'
    baseline = prep.guessBaseline(
                    testQuestions, 
                    testAnswers, 
                    testQuestionTypes,
                    outputFolder=outputFolder,
                    calcWups=True)

    # Find max length
    if maxlen == -1:
        maxlen = prep.findMaxlen(allQuestions)

    # Build output
    trainInput = prep.combine(\
        prep.lookupQID(trainQuestions, worddict, maxlen), trainImgIds)
    trainTarget = prep.lookupAnsID(trainAnswers, ansdict)
    validInput = prep.combine(\
        prep.lookupQID(validQuestions, worddict, maxlen), validImgIds)
    validTarget = prep.lookupAnsID(validAnswers, ansdict)
    testInput = prep.combine(\
        prep.lookupQID(testQuestions, worddict, maxlen), testImgIds)
    testTarget = prep.lookupAnsID(testAnswers, ansdict)

    # Dataset files
    np.save(\
        os.path.join(outputFolder, 'train.npy'),\
        np.array((trainInput, trainTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'valid.npy'),\
        np.array((validInput, validTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'test.npy'),\
        np.array((testInput, testTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'test-qtype.npy'),\
        testQuestionTypes)
    np.save(\
        os.path.join(outputFolder, 'vocab-dict.npy'),\
        np.array((worddict, idict, 
            ansdict, iansdict, 0), dtype=object))

    # Vocabulary files
    with open(os.path.join(outputFolder, \
        'question_vocabs.txt'), 'w+') as f:
        for word in idict:
            f.write(word + '\n')
    with open(os.path.join(outputFolder, \
        'answer_vocabs.txt'), 'w+') as f:
        for word in iansdict:
            f.write(word + '\n')

    # Frequency file
    with open(os.path.join(outputFolder, \
        'question_vocabs_freq.txt'), 'w+') as f:
        for i in wordfreq:
            f.write('%d\n' % i)

    # Image ID file
    with open(os.path.join(outputFolder, 'imgid_dict.pkl'), 'wb') as f:
        pkl.dump(imgidDict3, f)

    # For dataset release, output plain text file
    releaseFolder = os.path.join(outputFolder, 'release')
    if not os.path.exists(releaseFolder):
        os.makedirs(releaseFolder)
    trainFolder = os.path.join(releaseFolder, 'train')
    testFolder = os.path.join(releaseFolder, 'test')
    if not os.path.exists(trainFolder):
        os.makedirs(trainFolder)
    if not os.path.exists(testFolder):
        os.makedirs(testFolder)
    with open(os.path.join(releaseFolder, 'train', 'img_ids.txt'), 'w') as f:
        for imgid in trainImgIdsRelease:
            f.write('%s\n' % imgid)
        for imgid in validImgIdsRelease:
            f.write('%s\n' % imgid)
    with open(os.path.join(releaseFolder, 'train', 'questions.txt'), 'w') as f:
        for question in trainQuestions:
            f.write(question + '\n')
        for question in validQuestions:
            f.write(question + '\n')
    with open(os.path.join(releaseFolder, 'train', 'answers.txt'), 'w') as f:
        for answer in trainAnswers:
            f.write(answer + '\n')
        for answer in validAnswers:
            f.write(answer + '\n')
    with open(os.path.join(releaseFolder, 'train', 'types.txt'), 'w') as f:
        for typ in trainQuestionTypes:
            f.write(str(typ) + '\n')
        for typ in validQuestionTypes:
            f.write(str(typ) + '\n')
    with open(os.path.join(releaseFolder, 'test', 'img_ids.txt'), 'w') as f:
        for imgid in testImgIdsRelease:
            f.write('%s\n' % imgid)
    with open(os.path.join(releaseFolder, 'test', 'questions.txt'), 'w') as f:
        for question in testQuestions:
            f.write(question + '\n')
    with open(os.path.join(releaseFolder, 'test', 'answers.txt'), 'w') as f:
        for answer in testAnswers:
            f.write(answer + '\n')
    with open(os.path.join(releaseFolder, 'test', 'types.txt'), 'w') as f:
        for typ in testQuestionTypes:
            f.write(str(typ) + '\n')

    # Plot answer distribution
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    topKAnswers = min(100, len(afterFreq))
    plt.rc('font',**{'family':'serif','serif':['Time New Roman']})
    plt.rc('text', usetex=True)

    sorted_keys = sorted(range(len(afterFreq)), key=lambda k: afterFreq[k], reverse=True)
    bins = np.linspace(0, topKAnswers, num=topKAnswers + 1)
    beforeFreq2 = []
    for k in sorted_keys[:topKAnswers]:
        word = afterIdict[k]
        beforeIndex = beforeWorddict[word]
        beforeFreq2.append(beforeFreq[beforeIndex])

    (left, right, bottom, top) = hist.calcPath(beforeFreq2, bins)
    hist.hist(left, right, bottom, top, ax, 'blue')
    hist.setLimit(left, right, bottom, top, ax)

    (left, right, bottom, top) = hist.calcPath(sorted(afterFreq)[::-1][:topKAnswers], bins)
    hist.hist(left, right, bottom, top, ax, 'red')

    plt.legend(['Before', 'After'])
    plt.xlabel('Top 100 Answers')
    plt.ylabel('Number of Appearances in the Entire COCO-QA')
    plt.title(r'\textbf{Effect of Common Answer Rejection}')
    
    plt.savefig(os.path.join(outputFolder, 'answer_dist.pdf'))
    plt.savefig(os.path.join(outputFolder, 'answer_dist.eps'))