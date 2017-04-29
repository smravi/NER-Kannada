import sys

input1=sys.argv[1]
input2=sys.argv[2]
output=sys.argv[3]

with open(input1,'r',encoding='utf-8') as fp1:
	lines1=fp1.readlines()

with open(input2,'r',encoding='utf-8') as fp2:
	lines2=fp2.readlines()

fw=open(output,'w',encoding='utf-8')

n=len(lines1)

for i in range(n):
	tokens1=lines1[i].split()
	if(len(tokens1)<2):
		pass
	else:
		word=tokens1[0].strip()
		actual_tag=tokens1[1].strip()
	tokens2=lines2[i].split()
	if(len(tokens2)<2):
		pass
	else:
		pred_tag=tokens2[1].strip()
	fw.write(word+'\t'+actual_tag+'\t'+pred_tag+'\n')


