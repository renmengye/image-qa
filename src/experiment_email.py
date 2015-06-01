#! /usr/bin/python
import os
import sys

def appendList(resultsFolder, name):
    print 'Appended to email list.'
    tosend = os.path.join(resultsFolder, 'tosend.txt')  
    with open(tosend, 'a+') as f:
        f.write(name + '\n')

if __name__ == '__main__':
    resultsFolder = '/u/mren/code/image-qa/results'
    if len(sys.argv) > 1:
        resultsFolder = sys.argv[1]
    print resultsFolder
    tosend = os.path.join(resultsFolder, 'tosend.txt')
    if not os.path.exists(tosend):
        print 'No email task detected.'
        sys.exit()

    with open(tosend,'r') as f:
        id_s = f.readlines()

    for id_ in id_s:
        i = id_[:-1] # Remove newline character
        errfig = os.path.join(resultsFolder, '%s/%s_err.png' %(i, i))
        lossfig = os.path.join(resultsFolder, '%s/%s_loss.png' %(i, i))
        csv = os.path.join(resultsFolder, '%s/%s.csv' %(i, i))
        result = os.path.join(resultsFolder, '%s/result.txt' % i)
        if not os.path.exists(result):
            result = csv
        emailCommand = ('mutt renmengye@gmail.com -s "Experiment Summary %s" -a "%s" -a "%s" -a "%s" < "%s"'
             % (i, lossfig, errfig, csv, result))
        os.system(emailCommand)
        print 'Sent %s' % i

    with open(tosend, 'w') as f:
        f.write('')

    print 'Finished sending all emails.'
