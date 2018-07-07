# Equal Stacks
# https://www.hackerrank.com/challenges/equal-stacks/problem
#
# 比较三个堆什么时候相等


from collections import deque


def read_queue():
    return deque(map(int, input().strip().split()))


nstacks = len(input().split())
stacks = [read_queue() for i in range(nstacks)]
heights = list(map(sum, stacks))

while len(set(heights)) > 1:
    ihighest = heights.index(max(heights))
    heights[ihighest] -= stacks[ihighest].popleft()

print(heights[0])


