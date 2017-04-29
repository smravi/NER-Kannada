import sys

input1=sys.argv[1]

with open(input1,'r',encoding='utf-8') as fp1:
	lines1=fp1.readlines()


n=len(lines1)

for i in range(n):
	tokens1=lines1[i].split()
	if(len(tokens1)<2):
		pass
	else:
		actual_tag=tokens1[1].strip()
	


