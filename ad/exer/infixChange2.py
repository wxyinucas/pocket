from ds import Stack
import re


def infix_change(expression, new_type):
    assert new_type in ['post', 'pre']

    maps = {i: j for i, j in
            zip(['*', '/', '+', '-', '(', ')'], [3, 3, 2, 2, 1, 1])}
    raw = list(expression)
    raw = list(filter(lambda x: x != ' ', raw))

    if new_type == 'pre':
        raw = list(map(exchange, raw))
        raw = list(reversed(raw))
    opr_stack = Stack()
    output = []

    for itr in raw:
        if itr == '(':
            opr_stack.push(itr)
        elif itr == ')':
            while not opr_stack.is_empty() and opr_stack.peek() != '(':
                output.append(opr_stack.pop())
            opr_stack.pop()
        elif itr in maps.keys():
            while not opr_stack.is_empty() and maps[itr] <= maps[opr_stack.peek()]:
                output.append(opr_stack.pop())
            opr_stack.push(itr)
        else:
            output.append(itr)

    while not opr_stack.is_empty():
        output.append(opr_stack.pop())

    if new_type == 'pre':
        output = reversed(output)

    return ' '.join(output)


def exchange(x):
    if x == '(':
        return ')'
    elif x == ')':
        return '('
    else:
        return x


if __name__ == '__main__':
    # print(infix_change('A + B * C', new_type='post'))
    # print(infix_change('A + B * C', new_type='pre'))
    # print(infix_change('(A + B) * C', new_type='post'))
    # print(infix_change('(A + B) * C', new_type='pre'))
    for itr in ['(A+B)*(C+D)*(E+F)', 'A+((B+C)*(D+E))', 'A*B*C*D+E+F', 'A + B * C']:
        print(infix_change(itr, new_type='pre'))
