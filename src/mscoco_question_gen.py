import time
import re
from nltk.stem.wordnet import WordNetLemmatizer

parseFilename = '../../../data/mscoco/mscoco_caption.parse.txt'
lemmatizer = WordNetLemmatizer()

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

def whMovement(root):
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
    if found[0]:
        np = stack[0][-1]
        while np.className != 'DT' and len(np.children) > 0:
            np = np.children[0]
        if np.className == 'DT':
            np.text = 'the'

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

    for item in stack[0]:
        #print item.className,
        if item.className == 'SBAR':
            insideSBar = True
        if item.className == 'NP':
            insideNP = True

    if root.toSentence().startswith('how many'):
        #print 'WH already at the front'
        return False
    if insideSBar:
        #print 'Inside SBar'
        return False
    if insideNP:
        #print 'Inside NP'
        return False

    whStack = stack[0][:]

    # Look for VP
    found[0] = False
    stack[0] = []
    traverseFindTopClass(root, 'VP')
    #print
    # for item in stack[0]:
    #     print item.className,

    if not found[0]:
        print 'Not found VP'
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
        if frontWord is not None:
            # Remove WHNP from its parent
            whStack[-2].children.remove(whStack[-1])
            bigS = TreeNode('S', '', [whStack[-1], stack[0][1]], 0)
            stack[0][0].children = [bigS]
            bigS.children[1].children.insert(0, frontWord)
        else:
            print 'Not found front word'


    # reassign levels to the new tree
    root.relevel(0)
    #print 'WH-movement'
    return True
    #print 'WH-movement:', root

def splitCCStructure(root):
    # Find (ROOT (S ...) (CC ...) (S ...)) structure and split them into separate trees.
    roots = []
    node = root.children[0] # directly search for the top-most S.
    if node.className == 'S':
        if len(node.children) >= 3:
            if node.children[0].className == 'S' and node.children[1].className == 'CC' and node.children[2].className == 'S':
                roots.append(
                    TreeNode('ROOT', '', [node.children[0]], 0)
                    )
                roots.append(
                    TreeNode('ROOT', '', [node.children[2]], 0)
                    )
    if len(roots) == 0:
        roots.append(root)
    return roots

def askHowMany(root):
    found = [False] # A hack for closure support in python 2.7
    answer = ['']
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
                        answer[0] = child.text
                        count = child
                    # elif found[0] and (child.className == 'NN' or child.className == 'NNS'):
                    #     obj = child
                if found[0] and count is not None:
                    how = TreeNode('WRB', 'how', [], node.level + 2)
                    many = TreeNode('JJ', 'many', [], node.level + 2)
                    howmany = TreeNode('WHNP', '', [how, many], node.level + 1)
                    children = [howmany]
                    children.extend(node.children[node.children.index(count)+1:])
                    #node.children.remove(count)
                    node.children = children
                    node.className = 'WHNP'
                # else:
                #     found[0] = False
        if node.className == '.':
            node.text = '?' # Make it into question mark.

    #rootPreQuestion = root.copy()
    #if root.toSentence() == 'Two guys and a girl playing croquet in the yard .':
    roots = splitCCStructure(root)

    #if len(roots) >=2 :
    for r in roots:
        traverse(r)
        #print r
        if r.children[0].children[-1].className != '.':
            r.children[0].children.append(TreeNode('.', '?', [], 2))
        if found[0]:
            # if whMovement(r):
            #     yield (r.toSentence().lower(), escapeNumber(answer[0].lower()))
            whMovement(r)
            yield (r.toSentence().lower(), escapeNumber(answer[0].lower()))
        else:
            pass
            #return None
        found[0] = False
        answer[0] = ''


def ask(root):
    pass

if __name__ == '__main__':
    # Finite state machine implementation of syntax tree parser.
    state = 0
    currentClassStart = 0
    currentTextStart = 0
    classNameStack = []
    childrenStack = [[]]
    root = None
    rootsList = []
    level = 0

    def isAlpha(c):
        return 65 <= ord(c) <= 90 or 97 <= ord(c) <= 122

    def isNumber(c):
        return 48 <= ord(c) <= 57

    def exception(raw, i):
        print raw
        raise Exception(
            'Unexpected character "%c" (%d) at position %d' \
            % (raw[i], ord(raw[i]), i))

    def isClassName(s):
        if isAlpha(s) or s == '.' or s == ',' or s == '$' or s == '\'' or s == '`' or s == ':' or s == '-' or s == '#':
            return True
        else:
            return False 
    def isText(s):
        if isAlpha(s) or isNumber(s) or s == '.' or s == ',' or s == '-' or s == '\'' or s == '`' or s == '/' or s == '>' or s == ':' or s == ';' or s == '\\' or s == '!' or s == '?' or s == '&' or s == '-' or s == '=' or s == '#' or s == '$' or s == '@' or s == '_' or s == '*' or s == '+' or s == chr(194) or s == chr(160):
            return True
        else:
            return False 

    def state0(i):
        if raw[i] == '(':
            return 1
        else:
            return 0

    def state1(i):
        if isClassName(raw[i]):
            global currentClassStart, level, childrenStack
            currentClassStart = i
            level += 1
            childrenStack.append([])
            return 2
        else:
            exception(raw, i)

    def state2(i):
        if isClassName(raw[i]):
            return 2
        else:
            global classNameStack
            classNameStack.append(raw[currentClassStart:i])
            if raw[i] == ' ' and raw[i + 1] == '(':
                return 0
            elif raw[i] == ' ' and isText(raw[i + 1]):
                return 4
            elif raw[i] == '\n':
                return 3
            else:
                exception(raw, i)

    def state3(i):
        if raw[i] == ' ' and raw[i + 1] == '(':
            return 0
        elif raw[i] == ' ' and raw[i + 1] == ' ':
            return 3
        elif raw[i] == ' ' and isText(raw[i + 1]):
            return 4
        else:
            return 3

    def state4(i):
        if isText(raw[i]):
            global currentTextStart
            currentTextStart = i
            return 5
        else:
            exception(raw, i)

    def state5(i):
        if isText(raw[i]):
            return 5
        elif raw[i] == ')':
            wrapup(raw[currentTextStart:i])
            if level == 0:
                return 0
            elif raw[i + 1] == ')':
                return 6
            else:
                return 3
        else:
            exception(raw, i)

    def state6(i):
        if level == 0:
            return 0
        elif raw[i] == ')':
            wrapup('')
            return 6
        else:
            return 3

    def wrapup(text):
        global childrenStack, root, level, rootsList
        level -= 1
        # Note childrenStack[-1][:] gives a shallow copy of the list.
        root = TreeNode(classNameStack[-1], text, childrenStack[-1][:], level)
        del childrenStack[-1]
        del classNameStack[-1]
        childrenStack[-1].append(root)
        if level == 0:
            rootsList.append(root)
            # print 'Parsed tree:'
            # print root

    stateTable = [state0,state1,state2,state3,state4,state5,state6]

    startTime = time.time()
    questionCount = 0
    numSentences = 0
    with open(parseFilename) as f:
        for line in f:
            if len(rootsList) > 0:
                #print rootsList[0]
                numSentences += 1
                originalSent = rootsList[0].toSentence()
                qa = askHowMany(rootsList[0])
                hasItem = False
                for qaitem in qa:
                    questionCount += 1
                    hasItem = True
                    print ('Question %d:' % questionCount), qaitem[0], 'Answer:', qaitem[1]
                if hasItem:
                    print 'Original:', originalSent
                del(rootsList[0])
                if questionCount > 500:
                    break

            if not isAlpha(line[0]):
                raw = line
                #print line
                for i in range(len(line)):
                    #print state, level
                    state = stateTable[state](i)

    # Approx. 3447.5 sentences per second
    print 'Number of sentences parsed:', numSentences
    print 'Number of seconds:', time.time() - startTime
