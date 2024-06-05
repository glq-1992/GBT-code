splitFile =  open('split_1.txt','r')
questionFile = open('CSL-Daily-question.txt','r')
train = open('train.txt','w')
test = open('test.txt','w')
dev = open('dev.txt','w')


split_dict = dict()
for i,line in enumerate(splitFile):
    if not i == 0:
        folder, type = line.strip().split('|')
        split_dict[folder] = type

    
for i,line in enumerate(questionFile):
    if not i == 0:
        index,name,length,gloss,char,word,question = line.strip().split('|')[0:7]
        if not name in split_dict.keys():
            print(name)
            train.write(line.strip()+'\n')
        elif split_dict[name] == 'train':
            train.write(line.strip()+'\n')
        elif split_dict[name] == 'test':
            test.write(line.strip()+'\n')
        else:
            dev.write(line.strip()+'\n')


splitFile.close()
questionFile.close()
train.close()
test.close()
dev.close()
