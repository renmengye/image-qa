import os
import subprocess
import time
import re
import copy
import cPickle as pkl
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import sys

stanfordParserPath = \
    '../../../tools/stanford-parser-full-2015-01-30/lexparser.sh'

whiteListColorAdj = {
    'red': 1,
    'yellow': 1,
    'orange': 1,
    'brown': 1,
    'green': 1, 
    'blue': 1,
    'purple': 1,
    'black': 1,
    'white': 1,
    'gray': 1,
    'grey': 1,
    'violet': 1
}

whiteListLexname = {
    'noun.animal': 1,
    'noun.artifact': 1,
    'noun.food': 1,
    'noun.object': 1,
    'noun.plant': 1,
    'noun.possession': 1,
    'noun.shape': 1
}

blackListColorNoun = {
    'ride': 1,
    'riding': 1,
    'past': 1,
    'stand': 1,
    'standing': 1,
    'eating': 1,
    'holding': 1,
    'frosting': 1,
    'glow': 1,
    'glowing': 1,
    'resting': 1,
    'parked': 1
}

blackListNoun = {
    'female': 1,
    'females': 1,
    'male': 1,
    'males': 1,
    'commuter': 1,
    'commuters': 1,
    'player': 1,
    'players': 1,
    'match': 1,
    'matches': 1,
    'rider': 1,
    'riders': 1,
    'doll': 1,
    'dolls': 1,
    'ride': 1,
    'rides': 1,
    'riding': 1,
    'past': 1,
    'pasts': 1,
    'teddy': 1,
    'fan': 1,
    'fans': 1,
    'street': 1,
    'streets': 1,
    'arm': 1,
    'arms': 1,
    'head': 1,
    'heads': 1,
    'slope': 1,
    'slopes': 1,
    'shoot': 1,
    'shoots': 1,
    'photo': 1,
    'photos': 1,
    'space': 1,
    'spaces': 1,
    'stand': 1,
    'stands': 1,
    'standing': 1,
    'cross': 1,
    'crosses': 1,
    'crossing': 1,
    'eating': 1,
    'walking': 1,
    'driving': 1,
    'upright': 1,
    'structure': 1,
    'turn': 1,
    'system': 1,
    'arrangement': 1,
    'set': 1,
    'top': 1,
    'while': 1,
    'well': 1,
    'area': 1,
    'produce': 1,
    'thing': 1,
    'things': 1,
    'cut': 1,
    'cuts': 1,
    'holding': 1,
    'frosting': 1,
    'glow': 1,
    'glowing': 1,
    'ground': 1,
    'parked': 1
}

blackListCompoundNoun = {
    'tennis': 1,
    'delivery': 1,
    'soccer': 1,
    'baseball': 1,
    'fighter': 1,
    'mother': 1,
    'window': 1
}

blackListVerb = {
    'sink': 1,
    'sinks': 1,
    'counter': 1,
    'counters': 1,
    'cupboard': 1,
    'cupboards': 1,
    'has': 1,
    'have': 1,
    'contain': 1,
    'contains': 1,
    'containing': 1,
    'contained': 1,
    'spaniel': 1,
    'spaniels': 1,
    'mirror': 1,
    'mirrors': 1,
    'shower': 1,
    'showers': 1,
    'stove': 1,
    'stoves': 1,
    'bowl': 1,
    'bowls': 1,
    'tile': 1,
    'tiles': 1,
    'mouthwash': 1,
    'mouthwashes': 1,
    'smoke': 1,
    'smokes': 1
}

blackListPrep = {
    'with': 1,
    'of': 1,
    'in': 1,
    'down': 1,
    'as': 1
}

blackListLocation = {
    't-shirt': 1,
    't-shirts': 1,
    'jeans': 1,
    'shirt': 1,
    'shirts': 1,
    'uniform': 1,
    'uniforms': 1,
    'jacket': 1,
    'jackets': 1,
    'dress': 1,
    'dresses': 1,
    'hat': 1,
    'hats': 1,
    'tie': 1,
    'ties': 1,
    'costume': 1,
    'costumes': 1,
    'attire': 1,
    'attires': 1,
    'match': 1,
    'matches': 1,
    'coat': 1,
    'coats': 1,
    'cap': 1,
    'caps': 1,
    'gear': 1,
    'gears': 1,
    'sweatshirt': 1,
    'sweatshirts': 1,
    'helmet': 1,
    'helmets': 1,
    'clothing': 1,
    'clothings': 1,
    'cloth': 1,
    'clothes': 1,
    'blanket': 1,
    'blankets': 1,
    'enclosure': 1,
    'enclosures': 1,
    'suit': 1,
    'suits': 1,
    'photo': 1,
    'photos': 1,
    'picture': 1,
    'pictures': 1,
    'round': 1,
    'rounds': 1,
    'area': 1,
    'well': 1,
    'skirt': 1,
    'snowsuit': 1,
    'sunglasses': 1,
    'sweater': 1,
    'mask': 1,
    'frisbee': 1,
    'frisbees': 1,
    'shoe': 1,
    'umbrella': 1,
    'towel': 1,
    'scarf': 1,
    'phone': 1,
    'cellphone': 1,
    'motorcycle': 1,
    'device': 1,
    'computer': 1,
    'cake': 1,
    'hydrant': 1,
    'desk': 1,
    'stove': 1,
    'sculpture': 1,
    'lamp': 1,
    'fireplace': 1, 
    'bags': 1 ,
    'laptop': 1,
    'trolley': 1,
    'toy': 1,
    'bus': 1,
    'counter': 1, 
    'buffet': 1, 
    'engine': 1, 
    'graffiti':1,
    'clock': 1,
    'jet': 1,
    'ramp': 1,
    'brick': 1,
    'taxi': 1,
    'knife': 1,
    'flag': 1,
    'screen': 1,
    'parked': 1
}

blackListVerbLocation = {
    'sink': 1,
    'sinks': 1,
    'counter': 1,
    'counters': 1,
    'cupboard': 1,
    'cupboards': 1,
    'has': 1,
    'have': 1,
    'contain': 1,
    'contains': 1,
    'containing': 1,
    'contained': 1,
    'can': 1,
    'cans': 1
}

blackListNumberNoun = {
    'pole': 1,
    'vase': 1,
    'kite': 1, 
    'hay': 1,
    'shower': 1,
    'paddle': 1,
    'buffet': 1,
    'bicycle': 1,
    'bike': 1,
    'elephants': 1
}

synonymConvert = {
    'busses': 'buses',
    'plane': 'airplane',
    'planes': 'airplanes',
    'aircraft': 'airplane',
    'aircrafts': 'airplane',
    'jetliner': 'airliner',
    'jetliners': 'airliners',
    'bike': 'bicycle',
    'bikes': 'bicycles',
    'cycle': 'bicycle',
    'cycles': 'bicycles',
    'motorbike': 'motorcycle',
    'motorbikes': 'motorcycles',
    'grey': 'gray',
    'railroad': 'rail',
    'cell': 'cellphone',
    'doughnut': 'donut',
    'doughnuts': 'donuts'
}

# This problem is unresolved.
compoundNoun = {
    'fighter jet': 1,
    'soccer ball': 1,
    'tennis ball': 1
}

charText = {
    '.': 1,
    ',': 1,
    '-': 1,
    '\'': 1,
    '`': 1,
    '/': 1,
    '>': 1,
    ':': 1,
    ';': 1,
    '\\': 1,
    '!': 1,
    '?': 1,
    '&': 1,
    '-': 1,
    '=': 1,
    '#': 1,
    '$': 1,
    '@': 1,
    '_': 1,
    '*': 1,
    '+': 1,
    '%': 1,
    chr(194): 1,
    chr(160): 1
}

charClassName = {
    '.': 1,
    ',': 1,
    '$': 1,
    '\'': 1,
    '`': 1,
    ':': 1,
    '-': 1,
    '#': 1
}

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
            if item.className == 'SBAR':
                insideSBar = True
            elif item.className == 'NP' and item.level > 1:
                insideNP = True
            elif insideNP and item.className == 'VP':
                insideVP = True

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

        if insideSBar:
            return False
        if insideVP:
            return False


        if not found[0]:
            return False

        # Look for the verb that needs to be moved to the front.
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

        # Verb not found
        if frontWord is None:
            return False

        # Remove WHNP from its parent.
        whStack[whPosition-1].children.remove(whStack[whPosition])
        bigS = TreeNode('S', '', [whStack[whPosition], stack[0][1]], 0)
        stack[0][0].children = [bigS]
        bigS.children[1].children.insert(0, frontWord)

        # Reassign levels to the new tree.
        root.relevel(0)
        return True

    def splitCCStructure(self, root):
        # Find (ROOT (S ...) (CC ...) (S ...)) structure and split them into separate trees.
        # Issue: need to resolve coreference in the later sentences.
        roots = []

        # Directly search for the top-most S.
        node = root.children[0]
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
            return self.lexnameDict[word]
        else:
            synsets = wordnet.synsets(word)
            # Just pick the first definition
            if len(synsets) > 0:
                self.lexnameDict[word] = synsets[0].lexname()
                return self.lexnameDict[word]
            else:
                return None

    def askWhere(self, root):
        found = [False]
        answer = ['']
        def traverse(node, parent):
            # Ask one question for now.
            cont = True

            if node.text.lower() == 'this' or \
                node.text.lower() == 'that' or \
                node.text.lower() == 'there':
                node.text = ''

            if len(node.children) > 1 and \
                node.children[1].className == 'VP':
                    c = node.children[1]
                    while(len(c.children) > 0):
                        c = c.children[0]
                    if blackListVerbLocation.has_key(c.text.lower()):
                        cont = False

            if not found[0] and cont and node.className != 'PP':
                for child in node.children:
                    traverse(child, node)
            if node.className == 'PP' and \
                node.children[0].text == 'in':
                c = node

                while(len(c.children) > 0 and \
                    (c.children[-1].className == 'NP' \
                    or c.children[-1].className == 'NN')):
                    c = c.children[-1]

                if c.className == 'NN'and \
                    self.lookupLexname(c.text) == 'noun.artifact' and \
                    not blackListLocation.has_key(c.text.lower()):
                    found[0] = True
                    answer[0] = c.text 
                    # Treat ``where'' as WHNP for now.
                    where = TreeNode('WRB', 'where', [], 0)
                    parent.children.insert(parent.children.index(node), 
                        TreeNode('WHNP', '', [where], 0))
                    parent.children.remove(node)
                    # Remove other PP and ADVP in the parent
                    for child in parent.children:
                        if child.className == 'PP' or \
                            child.className == 'ADVP':
                            parent.children.remove(child)
        traverse(root, None)
        if found[0]:
            if self.whMovement(root):
                if root.children[0].children[-1].className != '.':
                    root.children[0].children.append(TreeNode('.', '?', [], 2))
                return [(root.toSentence().lower(), answer[0])]
            else:
                return []
        else:
            return []

    def askWhoWhat(self, root):
        found = [False] # A hack for closure support in python 2.7
        answer = ['']
        # Unlike in 'how many', here we enumerate all possible 'what's
        rootsReplaceWhat = [[]] 
        def traverse(node, parent):
            #if node.className != 'PP':
            cont = True
            # For now, not asking any questions inside PP.
            if node.className == 'PP' and blackListPrep.has_key(node.text.lower()):
                cont = False
            if (node.level > 1 and node.className == 'S') or node.className == 'SBAR':
                # Ignore possible answers in any clauses.
                cont = False
            ccNoun = False
            for child in node.children:
                if child.className == 'CC' or child.className == ',':
                    ccNoun = True
                    break
            if node.className == 'NP' and ccNoun:
                cont = False

            if len(node.children) > 1 and \
                node.children[1].className == 'PP':
                    cont = False
            
            if len(node.children) > 1 and \
                node.children[1].className == 'VP':
                    c = node.children[1]
                    while(len(c.children) > 0):
                        c = c.children[0]
                    if blackListVerb.has_key(c.text.lower()):
                        cont = False
            
            if node.className == 'VP' and \
                (node.children[0].text.startswith('attach') or \
                node.children[0].text.startswith('take')):
                cont = False

            # TRUNCATE SBAR!!!!!
            for child in node.children:
                if child.className == 'SBAR' or \
                    (child.level > 1 and child.className == 'S'):
                    node.children.remove(child)

            if cont:
                for child in node.children:
                    if child.className != 'PP' and \
                        child.className != 'ADVP':
                        traverse(child, node)

            if node.className == 'NP' and not ccNoun:
                replace = None
                whword = None
                for child in node.children:
                    # A wide ``angle'' view of the kitchen work area
                    if parent is not None:
                        if node.children.index(child) == len(node.children) - 1:
                            if parent.children.index(node) != \
                                len(parent.children) - 1:
                                if parent.children[\
                                    parent.children.index(node) + 1]\
                                    .className == 'NP':
                                    break
                        # The two people are walking down the ``beach''
                        foundDown = False
                        if parent.children.index(node) != 0:
                            for sib in parent.children[\
                                parent.children.index(node) - 1].children:
                                if sib.text == 'down':
                                    foundDown = True
                        if foundDown:
                            break
                    if child.className == 'NN' or child.className == 'NNS':
                        lexname = self.lookupLexname(child.text)
                        if lexname is not None:
                            if whiteListLexname.has_key(lexname) and \
                                not blackListNoun.has_key(child.text.lower()):
                                    whword = 'what'
                            if whword is not None:
                                answer[0] = child.text
                                found[0] = True
                                replace = child
                if replace != None and not blackListNoun.has_key(answer[0].lower()):
                    what = TreeNode('WP', whword, [], node.level + 1)
                    children_bak = copy.copy(node.children)
                    toremove = []

                    for child in node.children:
                        lexname = self.lookupLexname(child.text)
                        if child != replace and (
                            lexname != 'noun.act' or \
                            child.className != 'NN' or \
                            blackListCompoundNoun.has_key(child.text.lower())):
                            toremove.append(child)
                    for item in toremove:
                        node.children.remove(item)
                    if len(node.children) == 1:
                        node.children = [what]
                        node.className = 'WHNP'
                    else:
                        node.children[node.children.index(replace)] = \
                            TreeNode('WHNP', '', [what], node.level + 2)
                    rootcopy = root.copy()
                    rootcopy.answer = replace
                    rootsReplaceWhat[0].append(rootcopy)
                    node.className = 'NP'
                    node.children = children_bak

        rootsSplitCC = self.splitCCStructure(root)
        for r in rootsSplitCC:
            traverse(r, None)
            for r2 in rootsReplaceWhat[0]:
                if r2.children[0].children[-1].className != '.':
                    r2.children[0].children.append(TreeNode('.', '?', [], 2))
                else:
                    r2.children[0].children[-1].text = '?'
                if found[0]:
                    self.whMovement(r2)
                    yield (r2.toSentence().lower(), \
                        self.escapeNumber(r2.answer.text.lower()))
                else:
                    pass
            found[0] = False
            answer[0] = None
            rootsReplaceWhat[0] = []

    def askHowMany(self, root):
        # A hack for closure support in python 2.7
        found = [False]
        answer = [None]
        def traverse(node):
            if not found[0]:
                ccNoun = False
                cont = True
                for child in node.children:
                    if child.className == 'CC' or child.className == ',':
                        ccNoun = True
                        break

                if node.className == 'NP' and ccNoun:
                    cont = False
                
                if node.className == 'PP':
                    cont = False

                if cont:
                    for child in node.children:
                        traverse(child)
                    if node.className == 'NP' and (
                        node.children[-1].className == 'NNS' or \
                        node.children[-1].className == 'NN') and \
                        not node.children[-1].text.startswith('end'):
                        count = None
                        for child in node.children:
                            if child.className == 'CD' and \
                                not blackListNumberNoun.has_key(child.text.lower()):
                                found[0] = True
                                answer[0] = child
                                count = child
                        if found[0] and count is not None:
                            how = TreeNode('WRB', 'how', [], node.level + 2)
                            many = TreeNode('JJ', 'many', [], node.level + 2)
                            howmany = TreeNode('WHNP', '', [how, many], \
                                        node.level + 1)
                            children = [howmany]
                            children.extend(node.children[\
                                node.children.index(count) + 1:])
                            node.children = children
                            node.className = 'WHNP'
                            return

        roots = self.splitCCStructure(root)

        for r in roots:
            traverse(r)
            if r.children[0].children[-1].className != '.':
                r.children[0].children.append(TreeNode('.', '?', [], 2))
            else:
                r.children[0].children[-1].text = '?'
            if found[0] and not blackListNumberNoun.has_key(answer[0].text.lower()):
                r.answer = answer[0]
                self.whMovement(r)
                yield (r.toSentence().lower(), self.escapeNumber(\
                        answer[0].text.lower()))
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
                        whiteListColorAdj.has_key(child.text.lower()):
                        found[0] = True
                        answer[0] = child
                    if child.className == 'CC' and \
                        child.text == 'and':
                        # Blue and white? No.
                        found[0] = False
                        answer[0] = None
                        break
                    if (child.className == 'NN' or child.className == 'NNS') and \
                        not blackListColorNoun.has_key(child.text.lower()):
                        obj[0] = child
                if found[0] and obj[0] is not None:
                    qa[0].append((('what is the color of the %s ?' % \
                        obj[0].text).lower(), answer[0].text.lower()))
                    found[0] = False
                    obj[0] = None
                    answer[0] = None
        traverse(root)
        return qa[0]

def stanfordParse(sentence):
    with open('tmp.txt', 'w+') as f:
        f.write(sentence)
    with open('tmpout.txt', 'w+') as fout:
        subprocess.call([stanfordParserPath, 'tmp.txt'], stdout=fout)
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
        self.stateTable = [self.state0, self.state1, self.state2, \
            self.state3,self.state4,self.state5,self.state6]
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
        if TreeParser.isAlpha(s) or \
            charClassName.has_key(s):
            return True
        else:
            return False

    @staticmethod
    def isText(s):
        if TreeParser.isAlpha(s) or \
            TreeParser.isNumber(s) or \
            charText.has_key(s):
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
        self.level -= 1
        root = TreeNode(self.classNameStack[-1], text, \
            self.childrenStack[-1][:], self.level)
        del self.childrenStack[-1]
        del self.classNameStack[-1]
        self.childrenStack[-1].append(root)
        if self.level == 0:
            self.rootsList.append(root)

def lookupSynonym(word):
    if synonymConvert.has_key(word):
        return synonymConvert[word]
    else:
        return word

def questionGen(inputFolder):
    parseFilename = os.path.join(inputFolder, 'captions.parse.txt')
    imgidsFilename = os.path.join(inputFolder, 'imgids.txt')
    outputFilename = os.path.join(inputFolder, 'qa.pkl')
    startTime = time.time()
    questionCount = 0
    questionWhatCount = 0
    questionHowmanyCount = 0
    questionColorCount = 0
    questionWhereCount = 0
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
                    if qaitem[0] == 'what ?' or qaitem[0] == 'who ?' or \
                        len(qaitem[0].split(' ')) < 5:
                        # Ignore too short questions
                        continue
                    question = qaitem[0]
                    answer = lookupSynonym(qaitem[1])
                    questionCount += 1
                    questionWhatCount += 1
                    hasItem = True
                    print ('Question %d:' % questionCount), question, \
                        'Answer:', answer
                    # 0 is what-who question type
                    questionAll.append((question, answer, imgid, 0, \
                            originalSent))
                for qaitem in gen.askHowMany(parser.rootsList[0].copy()):
                    question = qaitem[0]
                    answer = lookupSynonym(qaitem[1])
                    questionCount += 1
                    questionHowmanyCount += 1
                    hasItem = True
                    print ('Question %d:' % questionCount), question, \
                        'Answer:', answer
                    # 1 is how-many question type
                    questionAll.append((question, answer, imgid, 1, \
                            originalSent))
                for qaitem in gen.askColor(parser.rootsList[0].copy()):
                    question = qaitem[0]
                    answer = lookupSynonym(qaitem[1])
                    questionCount += 1
                    questionColorCount += 1
                    hasItem = True
                    print ('Question %d:' % questionCount), question, \
                        'Answer:', answer
                    # 2 is color question type
                    questionAll.append((question, answer, imgid, 2, \
                            originalSent))
                for qaitem in gen.askWhere(parser.rootsList[0].copy()):
                    question = qaitem[0]
                    answer = lookupSynonym(qaitem[1])
                    questionCount += 1
                    questionWhereCount += 1
                    hasItem = True
                    print ('Question %d:' % questionCount), question, \
                        'Answer:', answer
                    questionAll.append((question, answer, imgid, 3, \
                            originalSent))
                if hasItem:
                    print 'Original:', originalSent
                    print '-' * 20
                    pass
                del(parser.rootsList[0])
                numSentences += 1
            parser.parse(line)

    print 'Number of sentences parsed:', numSentences
    print 'Number of images', len(imgids) / 5
    print 'Number of seconds:', time.time() - startTime
    print 'Number of questions:', questionCount
    print 'Number of what questions:', questionWhatCount
    print 'Number of how many questions:', questionHowmanyCount
    print 'Number of color questions:', questionColorCount
    print 'Number of where questions:', questionWhereCount

    # Output a list of tuples (q, a, imgid)
    with open(outputFilename, 'wb') as f:
        pkl.dump(questionAll, f)

def printQAs(qaiter, qid=0):
    for qaitem in qaiter:
        print ('Question %d:' % qid), qaitem[0], 'Answer:', qaitem[1]

def testHook(sentence):
    s = stanfordParse(sentence)
    s = s.split('\n')
    parser = TreeParser()
    gen = QuestionGenerator()
    for i in range(len(s)):
        parser.parse(s[i] + '\n')
    tree = parser.rootsList[0]
    print tree
    qaiter = gen.askWhoWhat(tree.copy())
    printQAs(qaiter)
    qaiter = gen.askHowMany(tree.copy())
    printQAs(qaiter)
    qaiter = gen.askColor(tree.copy())
    printQAs(qaiter)
    qaiter = gen.askWhere(tree.copy())
    printQAs(qaiter)

if __name__ == '__main__':
    """
    Usage:
    python cocoqa_question_gen_v4.py
                    -f[older] Input folder default '../../../data/mscoco/train'
                              Make sure the folder contains
                              1. captions.parse.txt Parsed by Stanford Parser
                              2. imgids.txt Image ID of the sentences
                              The program will output qa.pkl to the same folder
                    [-t {test input}]
                    [-l {lex input}]
    If not parameters then the program will generate the entire COCO-QA.
    """
    testMode = False
    lexMode = False
    folder = '../../../data/mscoco/train'

    for i, flag in enumerate(sys.argv):
        if flag == '-t' or flag == '-test':
            testMode = True
            testHook(sys.argv[i + 1])
        elif flag == '-l' or flag == '-lex':
            lexMode = True
            gen = QuestionGenerator()
            print gen.lookupLexname(sys.argv[i + 1])
        elif flag == '-f' or flag == '-folder':
            folder = sys.argv[i + 1]

    if not testMode and not lexMode:
        questionGen(folder)