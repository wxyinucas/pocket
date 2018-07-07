# https://www.hackerrank.com/challenges/queue-using-two-stacks/problem

orders = [input() for _ in range(int(input()))]
stack = []
queue = []

for order in orders:
    order = list(map(int, order.split()))
    if len(order) > 1:
        stack.append(order[1])
    elif order[0] == 2:
        for _ in range(len(stack)):
            queue.append(stack.pop())
        queue.pop()
        for _ in range(len(queue)):
            stack.append(queue.pop())
    elif order[0] == 3:
        print(stack[0])