import os

slts = open('test_slt.txt').readlines()
other = open('test_slt_2p.txt', 'w', encoding='utf-8')
slt_set = set()
for slt in slts:
    other.write(slt)
    other.write(slt.split('Q')[0][:-1] + '2' + slt[58:])

other.close()
