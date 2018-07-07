# https://www.hackerrank.com/challenges/down-to-zero-ii/problem
#!/bin/python

import sys

Q = int(input().strip())
nums = []  # All the query numbers
for a0 in range(Q):
    N = int(input().strip())
    nums.append(N)

max_nums = max(nums)  # Max of all queries
moves = [-1] * (1 + max_nums)  # A matrix of moves for all 									 # numbers till max number

moves[0] = 0
moves[1] = 1
moves[2] = 2
moves[3] = 3

for i in range(max_nums + 1):
    if (moves[i] == -1 or moves[i] > moves[i - 1] + 1):
        moves[i] = moves[i - 1] + 1
    for j in range(1, i + 1):
        if j * i > max_nums:
            break
        if (moves[j * i] == -1) or (moves[j * i] > moves[i] + 1):
            moves[j * i] = moves[i] + 1

for num in nums:
    print(moves[num])