import os
import sys

tosend = '../results/tosend.txt'
if not os.path.exists(tosend):
    print 'No email task detected.'
    sys.exit()

with open(tosend,'r') as f:
    id_s = f.readlines()

for id_ in id_s:
    errfig = '../results/%s/%s_err.png' %(id_, id_)
    lossfig = '../results/%s/%s_loss.png' %(id_,id_)
    csv = '../results/%s/%s.csv' %(id_,id_)
    emailCommand = ('mutt renmengye@gmail.com -s "Experiment Summary %s" -a "%s" -a "%s" < "%s"'
         % (id_, lossfig, errfig, csv))
    os.system(emailCommand)
    print 'Sent %s' % id_

with open(tosend, 'w') as f:
    f.write('')

