import sys
import random

def readfile(fileName):
    lines = []
    with open(fileName, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    sourceSentences = []
    maxSequenceLength = 0
    sentence=[]

    for line in lines:
        if line in ['\n', '\r\n']:
            sourceSentences.append(sentence)
            if(len(sentence)>maxSequenceLength):
                maxSequenceLength=len(sentence)
            sentence=[]
        else:
            sentence.append(line)
    return sourceSentences,maxSequenceLength

#inputfile=sys.argv[1]
inputfile='nn_input.txt'
trainfile='kan.train'
devfile='kan.val'
testfile='kan.test'

sentences,max_seq_len=readfile(inputfile)

n=len(sentences)
ntest=int(n*0.2)
rnums=random.sample(range(n-1), ntest)
sn=int(ntest/2)
dev_rnum=rnums[:sn]
test_rnum=rnums[sn:]

ftrain=open(trainfile,'w',encoding='utf-8')
fdev=open(devfile,'w',encoding='utf-8')
ftest=open(testfile,'w',encoding='utf-8')
for i,sentl in enumerate(sentences):
    sent=''.join(sentl)
    if i in dev_rnum:
        fdev.write(sent+'\n')
    elif i in test_rnum:
        ftest.write(sent+'\n')
    else:
        ftrain.write(sent+'\n')

ftrain.close()
fdev.close()
ftest.close()


print(n)
print(ntest)
print(rnums)