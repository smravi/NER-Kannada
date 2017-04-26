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

inputfile=sys.argv[1]
#inputfile='nn_input1.txt'
trainfile='kan.train3'
devfile='kan.val3'
testfile='kan.test3'

sentences,max_seq_len=readfile(inputfile)
print("Maximum Length",max_seq_len)

n=len(sentences)
ntest=int(n*0.2)
rnums=random.sample(range(n-1), ntest)
sn=int(ntest/2)
dev_rnum=rnums[:sn]
test_rnum=rnums[sn:]

print('Train file size',ntest)
print('Dev and Test size',sn)


ftrain=open(trainfile,'w',encoding='utf-8')
fdev=open(devfile,'w',encoding='utf-8')
ftest=open(testfile,'w',encoding='utf-8')
for i,sentl in enumerate(sentences):
    sent = ''.join(sentl)
    if i in dev_rnum:
    	fdev.write(sent)
    elif i in test_rnum:
    	ftest.write(sent)
    else:
    	ftrain.write(sent)    

ftrain.close()
fdev.close()
ftest.close()




