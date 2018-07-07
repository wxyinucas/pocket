# https://www.hackerrank.com/challenges/matrix-script/problem
import re

n_row, n_col = list(map(int, input().split()))
raw_str = [' '] * (n_row * n_col)
string = []

for i in range(n_row):
    raw_str[i*n_col: (i+1)*n_col+1] = list(input())

for i in range(n_col):
    for j in range(n_row):
        string.append(raw_str[i+j*n_col])

sentence = ''.join(string)

result = re.sub(r'(?<=\w)(\W+)(?=\w)', r' ', sentence)

print(result)



