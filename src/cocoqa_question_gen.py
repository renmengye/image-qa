import os
import subprocess
import time
import re
import copy
import cPickle as pkl
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import sys

if len(sys.argv) < 2:
    parseFilename = '../../../data/mscoco/train/caption.parse.txt'
    imgidsFilename = '../../../data/mscoco/train/imgids.txt'
    outputFilename = '../../../data/mscoco/train/qa.pkl'
else:
    folder = sys.argv[1]
    parseFilename = '%s/caption.parse.txt' %(sys.argv[1],sys.argv[1])
    imgidsFilename = '%s/imgids.txt' %(sys.argv[1],sys.argv[1])
    outputFilename = '%s/qa.pkl' %(sys.argv[1],sys.argv[1])

lemmatizer = WordNetLemmatizer()

class TreeNode:
    def __init__(self, className, text, children, level):
        self.className = className
        self.text = text
        self.children = children
        self.level = level
        pass
    def __str__(self):
        strlist = []
        for i in range(self.level):
            strlist.append('    ')
        strlist.extend(['(', self.className])
        if len(self.children) > 0:
            strlist.append('\n')
            for child in self.children:
                strlist.append(child.__str__())
            if len(self.text) > 0:
                for i in range(self.level + 1):
                    strlist.append('    ')
            else:
                for i in range(self.level):
                    strlist.append('    ')
        else:
            strlist.append(' ')
        strlist.append(self.text)
        strlist.append(')\n')
        return ''.join(strlist)
    def toSentence(self):
        strlist = []
        for child in self.children:
            childSent = child.toSentence()
            if len(childSent) > 0:
                strlist.append(childSent)
        if len(self.text) > 0:
            strlist.append(self.text)
        return ' '.join(strlist)
    def relevel(self,level):
        self.level = level
        for child in self.children:
            child.relevel(level + 1)
    def copy(self):
        children = []
        for child in self.children:
            children.append(child.copy())
        return TreeNode(self.className, self.text, children, self.level)

class QuestionGenerator:
    def __init__(self):
        self.lexnameDict = {}
        pass

    @staticmethod
    def escapeNumber(line):
        line = re.sub('^11$', 'eleven', line)
        line = re.sub('^12$', 'twelve', line)
        line = re.sub('^13$', 'thirteen', line)
        line = re.sub('^14$', 'fourteen', line)
        line = re.sub('^15$', 'fifteen', line)
        line = re.sub('^16$', 'sixteen', line)
        line = re.sub('^17$', 'seventeen', line)
        line = re.sub('^18$', 'eighteen', line)
        line = re.sub('^19$', 'nineteen', line)
        line = re.sub('^20$', 'twenty', line)
        line = re.sub('^10$', 'ten', line)
        line = re.sub('^0$', 'zero', line)
        line = re.sub('^1$', 'one', line)
        line = re.sub('^2$', 'two', line)
        line = re.sub('^3$', 'three', line)
        line = re.sub('^4$', 'four', line)
        line = re.sub('^5$', 'five', line)
        line = re.sub('^6$', 'six', line)
        line = re.sub('^7$', 'seven', line)
        line = re.sub('^8$', 'eight', line)
        line = re.sub('^9$', 'nine', line)
        return line

    def whMovement(self, root):
        #print 'Original:', root
        stack = [[]] # A hack for closure support
        found = [False]
        def traverseFindTopClass(node, className):
            if not found[0]:
                stack[0].append(node)
                if node.className == className:
                    found[0] = True
                else:
                    for child in node.children:
                        traverseFindTopClass(child, className)
                    if not found[0]:
                        del stack[0][-1]

        # Find the subject (first NP) and change determiner to 'the'
        traverseFindTopClass(root, 'NP')
        topNoun = None
        if found[0]:
            np = stack[0][-1]
            while np.className != 'DT' and len(np.children) > 0:
                np = np.children[0]
            if np.className == 'DT' and np.text.lower() == 'a':
                np.text = 'the'
            np = stack[0][-1]
            def lookForNoun(np):
                if len(np.children) > 0:
                    for child in np.children:
                        answer = lookForNoun(child)
                        if (answer != None):
                            return answer
                    return None
                else:
                    if np.className == 'NN' or np.className == 'NNS':
                        return np
                    else:
                        return None
            topNoun = lookForNoun(np)

        # Find the top verb
        found[0] = False
        stack[0] = []
        traverseFindTopClass(root, 'VP')
        topVP = None
        if found[0]:
            topVP = stack[0][-1]

        # First look for the position of WHNP
        found[0] = False
        stack[0] = []
        traverseFindTopClass(root, 'WHNP')
        if not found[0]: 
            return False

        # Check if the WHNP is inside an SBAR, not handling this case for now.
        insideSBar = False
        # Check if inside NP, violates A-over-A principal
        insideNP = False
        insideVP = False

        whStack = stack[0][:]
        whPosition = len(whStack) - 1

        for item in whStack:
            #print item.className,
            if item.className == 'SBAR':
                insideSBar = True
            # if item.className == 'SBAR':
            #     whPosition = whStack.index(item)
            elif item.className == 'NP' and item.level > 1:
                ##lexname = self.lookupLexname(item.children[0].text)
                ##if item.children[0].className != 'VBG' and lexname != 'noun.act':
                insideNP = True
            elif insideNP and item.className == 'VP':
                insideVP = True
                pass

        # Look for VP
        found[0] = False
        stack[0] = []
        traverseFindTopClass(root, 'VP')

        node = root
        parent = root
        while len(node.children) > 0:
            parent = node
            node = node.children[0]
        if parent.className == 'WHNP':
        #if root.toSentence().startswith('how many'):
            #print 'WH already at the front'
            if found[0]:
            # Add in missing verbs if possible
                vpnode = stack[0][-1]
                vpchild = vpnode.children[0]
                frontWord = None
                if vpchild.className == 'VBG': # only doing present, no is/are
                    verb = 'are' if root.answer.className == 'NNS' else 'is'
                    verbnode = TreeNode('VB', verb, [], vpchild.level)
                    vpnode.children.insert(0, verbnode)
            return True
        else:
            pass
            #print 'The first node is ' + parent.className
            #print root
        if insideSBar:
            #print 'Inside SBar'
            return False
        if insideVP:
            #print 'Inside NP'
            return False


        if not found[0]:
            #print 'Not found VP'
            return True
        else:
            vpnode = stack[0][-1]
            vpchild = vpnode.children[0]
            frontWord = None
            if vpchild.className == 'VBZ': # is, has, singular present
                if vpchild.text == 'is':
                    frontWord = vpchild
                    vpnode.children.remove(vpchild)
                elif vpchild.text == 'has': # Could be has something or has done
                    done = False
                    for child in vpnode.children:
                        if child.className == 'VP':
                            done = True
                            break
                    if done:
                        frontWord = vpchild
                        vpnode.children.remove(vpchild)
                    else:
                        frontWord = TreeNode('VBZ', 'does', [], 0)
                        vpchild.text = 'have'
                        vpchild.className = 'VB'
                else:
                    # need to lemmatize the verb and separate does
                    frontWord = TreeNode('VBZ', 'does', [], 0)
                    vpchild.className = 'VB'
                    vpchild.text = lemmatizer.lemmatize(vpchild.text, 'v')
                pass
            elif vpchild.className == 'VBP': # do, have, present
                if vpchild.text == 'are':
                    frontWord = vpchild
                    vpnode.children.remove(vpchild)
                else:    
                    frontWord = TreeNode('VBP', 'do', [], 0)
                    vpchild.className = 'VB'
                pass
            elif vpchild.className == 'VBD': # did, past tense
                if vpchild.text == 'was' or vpchild.text == 'were':
                    frontWord = vpchild
                    vpnode.children.remove(vpchild)
                elif vpchild.text == 'had': # Could be had something or had done
                    done = False
                    for child in vpnode.children:
                        if child.className == 'VP':
                            done = True
                            break
                    if done:
                        frontWord = vpchild
                        vpnode.children.remove(vpchild)
                    else:
                        frontWord = TreeNode('VBD', 'did', [], 0)
                        vpchild.text = 'have'
                        vpchild.className = 'VB'
                else:
                    # need to lemmatize the verb and separate did
                    frontWord = TreeNode('VBD', 'did', [], 0)
                    vpchild.className = 'VB'
                    vpchild.text = lemmatizer.lemmatize(vpchild.text, 'v')
                pass
            elif vpchild.className == 'MD': # will, may, shall
                frontWord = vpchild
                vpnode.children.remove(vpchild)
                pass
            elif vpchild.className == 'VBG': # only doing present, no is/are
                verb = 'are' if topNoun != None and topNoun.className == 'NNS' else 'is'
                frontWord = TreeNode('VBZ', verb, [], 0)
            if frontWord is not None:
                # Remove WHNP from its parent
                whStack[whPosition-1].children.remove(whStack[whPosition])
                bigS = TreeNode('S', '', [whStack[whPosition], stack[0][1]], 0)
                stack[0][0].children = [bigS]
                bigS.children[1].children.insert(0, frontWord)
                #print root
            else:
                pass
                #print 'Not found front word'

        # reassign levels to the new tree
        root.relevel(0)
        #print 'WH-movement'
        return True
        #print 'WH-movement:', root

    def splitCCStructure(self, root):
        # Find (ROOT (S ...) (CC ...) (S ...)) structure and split them into separate trees.
        # Issue: need to resolve coreference in the later sentences.
        roots = []
        node = root.children[0] # directly search for the top-most S.
        if node.className == 'S':
            if len(node.children) >= 3:
                childrenClasses = []
                for child in node.children:
                    childrenClasses.append(child.className)
                renew = True
                index = 0
                for c in childrenClasses:
                    if c == 'S' and renew:
                        root_ = TreeNode('ROOT', '', [node.children[index]], 0)
                        root_.relevel(0)
                        roots.append(root_)
                    elif c == 'CC':
                        renew = True
                    index += 1
        if len(roots) == 0:
            roots.append(root)
        return roots

    def lookupLexname(self, word):
        if self.lexnameDict.has_key(word):
            #print self.lexnameDict[word]
            return self.lexnameDict[word]
        else:
            synsets = wordnet.synsets(word) # Just pick the first definition
            if len(synsets) > 0:
                self.lexnameDict[word] = synsets[0].lexname()
                #print self.lexnameDict[word]
                return self.lexnameDict[word]
            else:
                return None

    def askWhoWhat(self, root):
        found = [False] # A hack for closure support in python 2.7
        answer = ['']
        rootsReplaceWhat = [[]] # Unlike in 'how many', here we enumerate all possible 'what's
        def traverse(node):
            #if node.className != 'PP':
            cont = True
            # For now, not asking any questions inside PP!
            if node.className == 'PP':
                cont = False
            if (node.level > 1 and node.className == 'S') or node.className == 'SBAR':
                # Ignore subsentences.
                cont = False
            ccNoun = False
            for child in node.children:
                if child.className == 'CC':
                    ccNoun = True
                    break
            if node.className == 'NP' and ccNoun:
                cont = False            
            if len(node.children) > 1 and \
                node.children[1].className == 'PP':
                node.children.remove(node.children[1])
            # if node.className == 'NP' and \
            # len(node.children) > 1 and \
            # node.children[1].className == 'PP':
            #     node.children.remove(node.children[1])
                # cont = False
            if cont:
                for child in node.children:
                    traverse(child)

            if node.className == 'NP' and not ccNoun:
                replace = None
                whword = None
                for child in node.children:
                    if child.className == 'NN' or child.className == 'NNS':
                        lexname = self.lookupLexname(child.text)
                        # print child.text
                        # print lexname
                        if lexname is not None:
                            if lexname == 'noun.person':
                                whword = 'who'
                            elif lexname == 'noun.animal' or \
                            lexname == 'noun.artifact' or \
                            lexname == 'noun.body' or \
                            lexname == 'noun.food' or \
                            lexname == 'noun.object' or \
                            lexname == 'noun.plant' or \
                            lexname == 'noun.possession' or \
                            lexname == 'noun.shape':
                                whword = 'what'
                            if whword is not None:
                                answer[0] = child.text
                                found[0] = True
                                replace = child
                if replace != None:
                    what = TreeNode('WP', whword, [], node.level + 1)
                    children_bak = copy.copy(node.children)
                    toremove = []

                    for child in node.children:
                        lexname = self.lookupLexname(child.text)
                        if child != replace and lexname != 'noun.act':
                            # print 'removed', child.text
                            toremove.append(child)
                    for item in toremove:
                        node.children.remove(item)
                    if len(node.children) == 1:
                        node.children = [what]
                        node.className = 'WHNP'
                    else:
                        node.children[node.children.index(replace)] = TreeNode('WHNP', '', [what], node.level + 2)
                    rootcopy = root.copy()
                    rootcopy.answer = replace
                    rootsReplaceWhat[0].append(rootcopy)
                    node.className = 'NP'
                    node.children = children_bak

        rootsSplitCC = self.splitCCStructure(root)
        for r in rootsSplitCC:
            traverse(r)
            for r2 in rootsReplaceWhat[0]:
                if r2.children[0].children[-1].className != '.':
                    r2.children[0].children.append(TreeNode('.', '?', [], 2))
                else:
                    r2.children[0].children[-1].text = '?'
                if found[0]:
                    #print r2
                    self.whMovement(r2)
                    #print r2
                    yield (r2.toSentence().lower(), self.escapeNumber(r2.answer.text.lower()))
                else:
                    pass
            found[0] = False
            answer[0] = None
            rootsReplaceWhat[0] = []

    def askHowMany(self, root):
        found = [False] # A hack for closure support in python 2.7
        answer = [None]
        def traverse(node):
            if not found[0]:
                for child in node.children:
                    traverse(child)
                if node.className == 'NP':
                    #obj = None
                    count = None
                    for child in node.children:
                        if child.className == 'CD':
                            found[0] = True
                            answer[0] = child
                            count = child
                    if found[0] and count is not None:
                        how = TreeNode('WRB', 'how', [], node.level + 2)
                        many = TreeNode('JJ', 'many', [], node.level + 2)
                        howmany = TreeNode('WHNP', '', [how, many], node.level + 1)
                        children = [howmany]
                        children.extend(node.children[node.children.index(count)+1:])
                        #node.children.remove(count)
                        node.children = children
                        node.className = 'WHNP'

        roots = self.splitCCStructure(root)

        for r in roots:
            traverse(r)
            if r.children[0].children[-1].className != '.':
                r.children[0].children.append(TreeNode('.', '?', [], 2))
            else:
                r.children[0].children[-1].text = '?'
            if found[0]:
                r.answer = answer[0]
                self.whMovement(r)
                yield (r.toSentence().lower(), self.escapeNumber(answer[0].text.lower()))
            else:
                pass
                #return None
            found[0] = False
            answer[0] = None

    def askColor(self, root):
        found = [False]
        answer = [None]
        obj = [None]
        qa = [[]]
        def traverse(node):
            for child in node.children:
                traverse(child)
            if node.className == 'NP':
                for child in node.children:
                    if child.className == 'JJ' and \
                    (child.text == 'red' or\
                    child.text == 'yellow' or\
                    child.text == 'orange' or\
                    child.text == 'brown' or\
                    child.text == 'green' or\
                    child.text == 'blue' or\
                    child.text == 'purple' or\
                    child.text == 'black' or\
                    child.text == 'white' or\
                    child.text == 'gray' or\
                    child.text == 'grey' or\
                    child.text == 'violet'):
                        found[0] = True
                        answer[0] = child
                    if child.className == 'CC' and child.text == 'and':
                        # Blue and white? No.
                        found[0] = False
                        answer[0] = None
                    if child.className == 'NN' or child.className == 'NNS':
                        obj[0] = child
                if found[0] and obj[0] is not None:
                    qa[0].append((('what is the color of the %s ?' % obj[0].text).lower(), answer[0].text.lower()))
                    found[0] = False
                    obj[0] = None
                    answer[0] = None
        traverse(root)
        return qa[0]

def stanfordParse(sentence):
    with open('tmp.txt', 'w+') as f:
        f.write(sentence)
    with open('tmpout.txt', 'w+') as fout:
        subprocess.call(['../../../tools/stanford-parser-full-2015-01-30/lexparser.sh', 'tmp.txt'], stdout=fout)
    with open('tmpout.txt') as f:
        result = f.read()
    os.remove('tmp.txt')
    os.remove('tmpout.txt')
    return result

# Finite state machine implementation of syntax tree parser.
class TreeParser:
    def __init__(self):
        self.state = 0
        self.currentClassStart = 0
        self.currentTextStart = 0
        self.classNameStack = []
        self.childrenStack = [[]]
        self.root = None
        self.rootsList = []
        self.level = 0
        self.stateTable = [self.state0,self.state1,self.state2,self.state3,self.state4,self.state5,self.state6]
        self.raw = None
        self.state = 0

    def parse(self, raw):
        if not self.isAlpha(raw[0]):
            self.raw = raw
            for i in range(len(raw)):
                self.state = self.stateTable[self.state](i)

    @staticmethod
    def isAlpha(c):
        return 65 <= ord(c) <= 90 or 97 <= ord(c) <= 122

    @staticmethod
    def isNumber(c):
        return 48 <= ord(c) <= 57

    @staticmethod
    def exception(raw, i):
        print raw
        raise Exception(
            'Unexpected character "%c" (%d) at position %d' \
            % (raw[i], ord(raw[i]), i))

    @staticmethod
    def isClassName(s):
        if TreeParser.isAlpha(s) or s == '.' or s == ',' or s == '$' or s == '\'' or s == '`' or s == ':' or s == '-' or s == '#':
            return True
        else:
            return False

    @staticmethod
    def isText(s):
        if TreeParser.isAlpha(s) or TreeParser.isNumber(s) or s == '.' or s == ',' or s == '-' or s == '\'' or s == '`' or s == '/' or s == '>' or s == ':' or s == ';' or s == '\\' or s == '!' or s == '?' or s == '&' or s == '-' or s == '=' or s == '#' or s == '$' or s == '@' or s == '_' or s == '*' or s == '+' or s == '%' or s == chr(194) or s == chr(160):
            return True
        else:
            return False 

    def state0(self, i):
        if self.raw[i] == '(':
            return 1
        else:
            return 0

    def state1(self, i):
        if self.isClassName(self.raw[i]):
            #global currentClassStart, level, childrenStack
            self.currentClassStart = i
            self.level += 1
            self.childrenStack.append([])
            return 2
        else:
            self.exception(self.raw, i)

    def state2(self, i):
        if self.isClassName(self.raw[i]):
            return 2
        else:
            #global classNameStack
            self.classNameStack.append(self.raw[self.currentClassStart:i])
            if self.raw[i] == ' ' and self.raw[i + 1] == '(':
                return 0
            elif self.raw[i] == ' ' and self.isText(self.raw[i + 1]):
                return 4
            elif self.raw[i] == '\n':
                return 3
            else:
                self.exception(self.raw, i)

    def state3(self, i):
        if self.raw[i] == ' ' and self.raw[i + 1] == '(':
            return 0
        elif self.raw[i] == ' ' and self.raw[i + 1] == ' ':
            return 3
        elif self.raw[i] == ' ' and self.isText(self.raw[i + 1]):
            return 4
        else:
            return 3

    def state4(self, i):
        if self.isText(self.raw[i]):
            #global currentTextStart
            self.currentTextStart = i
            return 5
        else:
            self.exception(self.raw, i)

    def state5(self, i):
        if self.isText(self.raw[i]):
            return 5
        elif i == len(self.raw) - 1:
            return 5
        elif self.raw[i] == ')':
            self.wrapup(self.raw[self.currentTextStart:i])
            if self.level == 0:
                return 0
            elif self.raw[i + 1] == ')':
                return 6
            else:
                return 3
        else:
            self.exception(self.raw, i)

    def state6(self, i):
        if self.level == 0:
            return 0
        elif self.raw[i] == ')':
            self.wrapup('')
            return 6
        else:
            return 3

    def wrapup(self, text):
        #global childrenStack, root, self.level, rootsList
        self.level -= 1
        root = TreeNode(self.classNameStack[-1], text, self.childrenStack[-1][:], self.level)
        del self.childrenStack[-1]
        del self.classNameStack[-1]
        self.childrenStack[-1].append(root)
        if self.level == 0:
            self.rootsList.append(root)
            # print 'Parsed tree:'
            # print root

def questionGen():
    startTime = time.time()
    questionCount = 0
    questionWhatCount = 0
    questionHowmanyCount = 0
    questionColorCount = 0
    numSentences = 0
    parser = TreeParser()
    gen = QuestionGenerator()

    questionAll = []
    with open(imgidsFilename) as f:
        imgids = f.readlines()

    with open(parseFilename) as f:
        for line in f:
            if len(parser.rootsList) > 0:
                if len(imgids) > numSentences:
                    imgid = imgids[numSentences][:-1]
                else:
                    print numSentences
                    print 'Finished'
                originalSent = parser.rootsList[0].toSentence()
                hasItem = False
                for qaitem in gen.askWhoWhat(parser.rootsList[0].copy()):
                    if qaitem[0] == 'what ?' or qaitem == 'who ?':
                        continue
                    else:
                        question = qaitem[0]
                    questionCount += 1
                    questionWhatCount += 1
                    hasItem = True
                    print ('Question %d:' % questionCount), qaitem[0], 'Answer:', qaitem[1]
                    # 0 is what-who question type
                    questionAll.append((qaitem[0], qaitem[1], imgid, 0))
                for qaitem in gen.askHowMany(parser.rootsList[0].copy()):
                    questionCount += 1
                    questionHowmanyCount += 1
                    hasItem = True
                    print ('Question %d:' % questionCount), qaitem[0], 'Answer:', qaitem[1]
                    # 1 is how-many question type
                    questionAll.append((qaitem[0], qaitem[1], imgid, 1))
                for qaitem in gen.askColor(parser.rootsList[0].copy()):
                    questionCount += 1
                    questionColorCount += 1
                    hasItem = True
                    print ('Question %d:' % questionCount), qaitem[0], 'Answer:', qaitem[1]
                    # 2 is color question type
                    questionAll.append((qaitem[0], qaitem[1], imgid, 2))
                if hasItem:
                    print 'Original:', originalSent
                    print '-' * 20
                    pass
                del(parser.rootsList[0])
                numSentences += 1
            parser.parse(line)
            if numSentences > 500:
                break
    # Approx. 3447.5 sentences per second
    #print questionAll
    print 'Number of sentences parsed:', numSentences
    print 'Number of images', len(questionAll)
    print 'Number of seconds:', time.time() - startTime
    print 'Number of questions:', questionCount
    print 'Number of what questions:', questionWhatCount
    print 'Number of how many questions:', questionHowmanyCount
    print 'Number of color questions:', questionColorCount

    # Output a list of tuples (q, a, imgid)
    # with open(outputFilename, 'wb') as f:
    #     pkl.dump(questionAll, f)
    with open(outputFilename+'.csv', 'w') as f:
        f.write('#,question,answer,type\n')
        for i,item in enumerate(questionAll):
            question = item[0]
            if 'how many' in question:
                typ = 2
            elif question.startswith('what is the color'):
                typ = 3
            elif 'who' in question:
                typ = 4
            else:
                typ = 1
            f.write(str(i+1)+','+item[0].replace(',','')+','+item[1]+','+str(typ)+'\n')

def testHook():
    #s = stanfordParse('There are two ovens in a kitchen restaurant , and one of them is being used .')
    #s = stanfordParse('A bathroom with two sinks a bathtub and a shower with lots of lighting from the windows .')
    #s = stanfordParse('A man waits at the crosswalk with his bicycle .')
    s = stanfordParse('A boy is playing with two dogs .')

    #print s
    s = s.split('\n')
    parser = TreeParser()
    gen = QuestionGenerator()
    for i in range(len(s)):
        #print s[i]
        parser.parse(s[i] + '\n')
    tree = parser.rootsList[0]
    print tree
    qaiter = gen.askWhoWhat(tree.copy())
    for qaitem in qaiter:
        print ('Question:'), qaitem[0], 'Answer:', qaitem[1]
    qaiter = gen.askHowMany(tree.copy())
    for qaitem in qaiter:
        print ('Question:'), qaitem[0], 'Answer:', qaitem[1]
    qaiter = gen.askColor(tree.copy())
    for qaitem in qaiter:
        print ('Question:'), qaitem[0], 'Answer:', qaitem[1]

if __name__ == '__main__':
    #testHook()
    questionGen()