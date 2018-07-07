from ds import Stack
import re


def infix_to_postfix(string):
    string = re.sub(r'\(', r'( ', string)
    string = re.sub(r'\)', r' )', string)
    string = string.strip().split()

    prec = {i: j for i, j in
            zip(['**', '*', '/', '+', '-', '(', ')'], [4, 3, 3, 2, 2, 1, 1, ])}
    opstack = Stack()
    output = []

    for s_itr in string:
        if s_itr == '(':
            opstack.push(s_itr)
        elif s_itr == ')':
            while opstack.peek() != '(':
                output.append(opstack.pop())
            opstack.pop()
        elif s_itr in list(prec.keys())[:-2]:
            while not opstack.isEmpty() and prec[opstack.peek()] >= prec[s_itr]:
                output.append(opstack.pop())
            opstack.push(s_itr)
        else:
            output.append(s_itr)

    while not opstack.isEmpty():
        output.append(opstack.pop())

    return ' '.join(output)


if __name__ == '__main__':
    print(infix_to_postfix("( A + B ) * ( C + D )"))
    print(infix_to_postfix("( A + B ) * C - ( D - E ) * ( F + G )"))
    print(infix_to_postfix('10 + 3 * 5 / (16 - 4)'))