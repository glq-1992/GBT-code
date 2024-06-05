train_file = open('train.txt','r')
train_overlap_file = open('train_overlap.txt','w')
dev_file = open('dev.txt','r')
dev_overlap_file = open('dev_overlap.txt','w')
test_file = open('test.txt','r')
test_overlap_file = open('test_overlap.txt','w')

for line in train_file.readlines():
    index,name,length,gloss,char,word,question= line.strip().split('|')[0:7]
    # charNoSpace = ''.join([i.strip() for i in char.split(' ')])
    charNoSpace = ''.join([i.strip() for i in char.split(' ')]).replace(' ','')
    questionNoSpace = ''.join([i.strip() for i in question.split(' ')]).replace(' ','')
    if charNoSpace == questionNoSpace:
        train_overlap_file.write(line.strip()+'|'+'Overlap!'+'\n')
    else:
        train_overlap_file.write(line)

for line in test_file.readlines():
    index,name,length,gloss,char,word,question= line.strip().split('|')[0:7]
    # charNoSpace = ''.join([i.strip() for i in char.split(' ')])
    charNoSpace = ''.join([i.strip() for i in char.split(' ')]).replace(' ','')
    questionNoSpace = ''.join([i.strip() for i in question.split(' ')]).replace(' ','')
    if charNoSpace == questionNoSpace:
        test_overlap_file.write(line.strip()+'|'+'Overlap!'+'\n')
    else:
        test_overlap_file.write(line)

for line in dev_file.readlines():
    index,name,length,gloss,char,word,question= line.strip().split('|')[0:7]
    # charNoSpace = ''.join([i.strip() for i in char.split(' ')])
    charNoSpace = ''.join([i.strip() for i in char.split(' ')]).replace(' ','')
    questionNoSpace = ''.join([i.strip() for i in question.split(' ')]).replace(' ','')
    if charNoSpace == questionNoSpace:
        dev_overlap_file.write(line.strip()+'|'+'Overlap!'+'\n')
    else:
        dev_overlap_file.write(line)


train_file.close()
train_overlap_file.close()
dev_file.close()
dev_overlap_file.close()
test_file.close()
test_overlap_file.close()