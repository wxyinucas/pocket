from ds import Stack
import re

operator = ['*', '/', '+', '-']


def postfix_evaluat(string):
    string = re.sub(r'\(', r'( ', string)
    string = re.sub(r'\)', r' )', string)
    string = string.strip().split()

    operand = Stack()

    for s_itr in string:
        if s_itr not in operator:
            operand.push(s_itr)
        else:
            operand.push(op(s_itr, operand.pop(), operand.pop()))

    return operand.peek()


def op(operator, x, y):
    x, y = list(map(float, [x, y]))
    if operator == '+':
        return x + y
    elif operator == '*':
        return x * y
    elif operator == '-':
        return y - x
    elif operator == '/':
        return y / x
    else:
        raise ImportError('What is your operator??')


if __name__ == '__main__':
    print(postfix_evaluat('10 3 5 * 16 4 - / +'))
