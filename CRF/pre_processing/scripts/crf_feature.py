import sys
import random
import math

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

inputfile=['kan.train3w','kan.val3w','kan.test3w']
#inputfile=['out_w']
#trainfile='kan.train'
#devfile='kan.val'
#testfile='kan.test'

for file in inputfile:
    sentences,max_seq_len=readfile(file)
    writefile=file+'_bf'
    fw=open(writefile,'w',encoding='utf-8')
    fw.write('Word'+'\t'+'firstword'+'\t'+'Prefix3'+'\t'+'Prefix4'+'\t'+'Prefix5'+'\t'+'Suffix3'+'\t'+'Suffix4'+'\t'+'Suffix5'+'\n')
    for sentl in sentences:
        firstword=sentl[0].strip()
        for word in sentl:
            charlist=list(word.strip())
            nchars=len(charlist)
            i3=min(3,nchars)
            i4=min(4,nchars)
            i5=min(5,nchars)

            pref3=''.join(charlist[:i3])
            pref4=''.join(charlist[:i4])
            pref5=''.join(charlist[:i5])

            suf3=''.join(charlist[nchars-i3:])
            suf4=''.join(charlist[nchars-i4:])
            suf5=''.join(charlist[nchars-i5:])

            fw.write(word.strip()+'\t'+firstword+'\t')
            fw.write(pref3+'\t'+pref4+'\t'+pref5+'\t')
            fw.write(suf3+'\t'+suf4+'\t'+suf5+'\n')
    fw.close()




