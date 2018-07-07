# https://www.hackerrank.com/challenges/maximum-element/problem


orders = [input() for _ in range(int(input()))]
stack = [0]
_max = 0

for order in orders:
    order = list(map(int, order.split()))
    if len(order) > 1:
        _max = order[1] if order[1] > stack[-1] else stack[-1]
        stack.append(_max)
    elif order[0] == 2:
        stack.pop()
    elif order[0] == 3:
        print(stack[-1])