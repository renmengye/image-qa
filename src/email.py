#! /usr/bin/python
import os
import sys

resultsFolder = '/u/mren/code/image-qa/results/'
tosend = resultsFolder + 'tosend.txt'
if not os.path.exists(tosend):
    print 'No email task detected.'
    sys.exit()

with open(tosend,'r') as f:
    id_s = f.readlines()

for id_ in id_s:
    i = id_[:-1] # Remove newline character
    errfig = resultsFolder + '%s/%s_err.png' %(i, i)
    lossfig = resultsFolder + '%s/%s_loss.png' %(i,i)
    csv = resultsFolder + '%s/%s.csv' %(i,i)
    result = resultsFolder + '%s/result.txt'
    if not os.path.exists(result):
        result = csv
    emailCommand = ('mutt renmengye@gmail.com -s "Experiment Summary %s" -a "%s" -a "%s" -a "%s" < "%s"'
         % (i, lossfig, errfig, csv, result))
    os.system(emailCommand)
    print 'Sent %s' % i

with open(tosend, 'w') as f:
    f.write('')

print 'Finished.'
