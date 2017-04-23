import re
import sys

readfile=sys.argv[1]
writefile=sys.argv[2]
discardfile='discard.txt'
wikitags='wikitags.txt'

lines=[]
with open(readfile,'r',encoding='utf-8') as fp:
	lines=fp.readlines()

fw=open(writefile,'w',encoding='utf-8')

fd=open(discardfile,'w',encoding='utf-8')

ft=open(wikitags,'w',encoding='utf-8')

for line in lines:
	nline = re.sub(r'[^ \.\,0-9\u0C80-\u0CFF]+', r'',line)
	if len(nline.split())<2:
		fd.write(nline+'\n')
	elif nline.startswith(u'ವಿಕಿಪೀಡಿಯ') or nline.startswith(u'ಈ ಪುಟವನ್ನು ಕೊನೆಯಾಗಿ'):
		ft.write(nline+'\n')
	else:
		fw.write(nline+'\n')



	


