import os

file_path = '/Users/xyzh/PycharmProjects/WXY_Project/examples/tmp.txt'
f = open(file_path, 'r')
txt = list(f)

for i in range(0, len(txt), 3):
    C = txt[i].split(' ')[0]
    TEXT = txt[i+1]
    print(C, TEXT)
