import sys
import re

readfile=sys.argv[1]
tempfile='temp.txt'
writefile=sys.argv[2]

#Period Split
lines=[]
with open(readfile,'r',encoding='utf-8') as fp:
	lines=fp.readlines()
fw=open(tempfile,'w',encoding='utf-8')

for line in lines:
	nline = re.sub(r'[\.]+', r'.\n',line)
	fw.write(nline+'\n')

fw.close()

#---------------------------------------------
#Tokenize
lines=[]
with open(tempfile,'r',encoding='utf-8') as fr:
	lines=fr.readlines()

fw=open(writefile,'w',encoding='utf-8')
for line in lines:
	tokens=re.split('([\,\.]|\s)+',line)
	for token in tokens:
		if token.strip()!='':
			fw.write(token+' ')
	#fw.write('\n')
			
	


	

