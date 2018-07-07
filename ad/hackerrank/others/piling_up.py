# https://www.hackerrank.com/challenges/piling-up/problem
from collections import deque
import numpy as np


def piling_check():
    input()
    cubes = deque(map(int, input().split()))
    pointer = np.inf

    while len(cubes):
        if cubes[0] < cubes[-1]:
            if cubes[-1] <= pointer:
                pointer = cubes.pop()
            else:
                return 'No'
        else:
            if cubes[0] <= pointer:
                pointer = cubes.popleft()
            else:
                return 'No'

    return 'Yes'


if __name__ == '__main__':
    for _ in range(int(input())):
        print(piling_check())
