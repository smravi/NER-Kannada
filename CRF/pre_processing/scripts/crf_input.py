import sys

inputfile=sys.argv[1]
out_file=sys.argv[2]

lines=[]
with open(inputfile,'r',encoding='utf-8') as fp:
	lines=fp.readlines()

fw=open(out_file,'w',encoding='utf-8')

for line in lines:
	tokens=line.split()
	if(len(tokens)==0):
		fw.write(line)
		print(line)
	else:
		word=tokens[0]
		tag=tokens[1].upper()
		line=word.strip()+'\t'+tag.strip()
		fw.write(line)
		fw.write('\n')
