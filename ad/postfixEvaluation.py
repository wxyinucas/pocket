from ds import Stack


def postfix_evaluate(string):
    str_split = string.strip().split()
    oprand_stack = Stack()

    for i in str_split:
        if i not in ['+', '-', '*', '/']:
            oprand_stack.push(int(i))
        else:
            oprand_stack.push(culculate(i, oprand_stack.pop(), oprand_stack.pop()))

    return oprand_stack.peek()


def culculate(op, x1, x2):
    if op == '*':
        return x2 * x1
    elif op == '+':
        return x2 + x1
    elif op == '-':
        return x2 - x1
    elif op == '/':
        return x2 / x1
    else:
        raise TypeError('Wrong operator!')


if __name__ == '__main__':
    print(postfix_evaluate('15 3 /'))
    print(postfix_evaluate('17 10 + 3 * 9 / '))