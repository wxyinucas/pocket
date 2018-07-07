orders = [input() for _ in range(int(input()))]
stack_adder = []
stack_popper = []


def queue_pop():
    global stack_adder, stack_popper
    if len(stack_popper):
        return stack_popper.pop()
    else:
        for _ in range(len(stack_adder)):
            stack_popper.append(stack_adder.pop())
        return stack_popper.pop()


for order in orders:
    if order[0] == '1':
        stack_adder.append(int(order[1:]))
    elif order[0] == '2':
        queue_pop()
    elif order[0] == '3':
        tail = queue_pop()
        print(tail)
        stack_popper.append(tail)
