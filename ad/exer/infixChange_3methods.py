from ds import Stack
import re


def infix_to_postfix(expression):
    maps = {i: j for i, j in
            zip(['*', '/', '+', '-', '(', ')'], [3, 3, 2, 2, 1, 1])}

    expression = re.sub(r'\(', '( ', expression)
    expression = re.sub(r'\)', ' )', expression)
    raw = expression.split()
    operator = Stack()
    new_str = []

    for itr in raw:
        if itr == '(':
            operator.push(itr)
        elif itr == ')':
            while operator.peek() != '(':
                new_str.append(operator.pop())
            operator.pop()
        elif itr in list(maps.keys())[:4]:
            while (not operator.is_empty()) and maps[itr] <= maps[operator.peek()]:
                new_str.append(operator.pop())
            operator.push(itr)
        else:
            new_str.append(itr)

    while not operator.is_empty():
        new_str.append(operator.pop())

    return ' '.join(new_str)


def infix_to_prefix(expression):
    maps = {i: j for i, j in
            zip(['*', '/', '+', '-', '(', ')'], [3, 3, 2, 2, 1, 1])}

    operator = Stack()
    expression = re.sub(r'\(', '( ', expression)
    expression = re.sub(r'\)', ' )', expression)
    raw = list(reversed(expression.split()))

    output = []
    for itr in raw:
        if itr == ')':
            operator.push(itr)
        elif itr == '(':
            while operator.peek() != ')':
                output.append(operator.pop())
            operator.pop()
        elif itr in maps.keys():
            while not operator.is_empty() and maps[itr] <= maps[operator.peek()]:
                output.append(operator.pop())
            operator.push(itr)
        else:
            output.append(itr)

    while not operator.is_empty():
        output.append(operator.pop())

    output = reversed(output)

    return ' '.join(output)


def infix_change(expression, change_type='postfix'):
    assert change_type in ['postfix', 'prefix'], '{} is not in postfix or prefix'.format(change_type)

    maps = {i: j for i, j in
            zip(['*', '/', '+', '-', '(', ')'], [3, 3, 2, 2, 1, 1])}

    operator = Stack()

    if change_type == 'prefix':
        expression = re.sub(r'\(', '} ', expression)
        expression = re.sub(r'\)', ' (', expression)
        expression = re.sub(r'\}', ') ', expression)
        raw = list(reversed(expression.split()))
    else:
        expression = re.sub(r'\(', '( ', expression)
        expression = re.sub(r'\)', ' )', expression)
        raw = list(expression.split())

    output = []
    for itr in raw:
        if itr == '(':
            operator.push(itr)
        elif itr == ')':
            while operator.peek() != '(':
                output.append(operator.pop())
            operator.pop()
        elif itr in maps.keys():
            while not operator.is_empty() and maps[itr] <= maps[operator.peek()]:
                output.append(operator.pop())
            operator.push(itr)
        else:
            output.append(itr)

    while not operator.is_empty():
        output.append(operator.pop())

    if change_type == 'prefix':
        output = list(reversed(output))

    return ' '.join(output)


if __name__ == '__main__':
    print(infix_to_postfix('A + B * C'))
    print(infix_to_postfix('(A + B) * C'))
    print(infix_to_prefix('A + B * C'))
    print(infix_to_prefix('(A + B) * C'))
    # print(infix_change('A + B * C', 'postfix'))
    # print(infix_change('(A + B) * C', 'postfix'))
    # print(infix_change('A + B * C', 'prefix'))
    # print(infix_change('(A + B) * C', 'prefix'))
