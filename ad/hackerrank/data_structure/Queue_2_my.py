from builtins import print
from collections import deque

n, inf = int(input()), float('inf')
grid = [list(input()) for _ in range(n)]
x_start, y_start, x_target, y_target = list(map(int, input().split()))
dist = [[inf] * n for _ in range(n)]
dist[x_start][y_start], grid[x_target][y_target] = 0, '*'

queue = deque([[x_start, y_start]])
d = 0
while queue:
    x0, y0 = queue.popleft()
    d = dist[x0][y0]
    if grid[x0][y0] == '*':
        break

    for x_itr in [range(x0, n, 1), range(x0-1, -1, -1)]:
        for x in x_itr:
            if grid[x][y0] == 'X':
                break
            if dist[x][y0] == inf:
                dist[x][y0] = d + 1
                queue.append([x, y0])

    for y_itr in [range(y0, n, 1), range(y0-1, -1, -1)]:
        for y in y_itr:
            if grid[x0][y] == 'X':
                break
            if dist[x0][y] == inf:
                dist[x0][y] = d + 1
                queue.append([x0, y])

print(d)


