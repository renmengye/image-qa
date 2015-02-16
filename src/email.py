import os
import sys

tosend = '../results/tosend.txt'
if not os.path.exists(tosend):
    print 'No email task detected.'
    sys.exit()

with open(tosend,'r') as f:
    id_s = f.readlines()

for id_ in id_s:
    i = id_[:-1] # Remove newline character
    errfig = '../results/%s/%s_err.png' %(i, i)
    lossfig = '../results/%s/%s_loss.png' %(i,i)
    csv = '../results/%s/%s.csv' %(i,i)
    emailCommand = ('mutt renmengye@gmail.com -s "Experiment Summary %s" -a "%s" -a "%s" < "%s"'
         % (i, lossfig, errfig, csv))
    os.system(emailCommand)
    print 'Sent %s' % i

with open(tosend, 'w') as f:
    f.write('')

