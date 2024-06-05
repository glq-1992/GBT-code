import random
alls = open('dev_test.txt', encoding='utf-8').readlines()
dev = open('test.txt', 'w', encoding='utf-8')
test = open('dev.txt', 'w', encoding='utf-8')
random.shuffle(alls)
for all in alls[:3004]:
    dev.write(all)
for all in alls[3004:]:
    test.write(all)
test.close()
dev.close()