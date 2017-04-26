import re
import sys

readfile=sys.argv[1]
writefile=sys.argv[2]


lines=[]
with open(readfile,'r',encoding='utf-8') as fp:
	lines=fp.readlines()

fw=open(writefile,'w',encoding='utf-8')


for line in lines:
	if line.strip()=='.':
		fw.write(line)
		fw.write('\n') 
	else:
		fw.write(line)
	

	


