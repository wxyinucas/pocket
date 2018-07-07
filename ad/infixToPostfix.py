from ds import Stack


def infix_to_postfix(string):
    # parenthesis = ['(', ')']
    operator = ['+', '-', '*', '/', '**']
    prec = {i: j for i, j in
            zip(['**', '*', '/', '+', '-', '(', ')'], [4, 3, 3, 2, 2, 1, 1, ])}

    postack = Stack()
    output = []
    str_split = string.split()

    for s in str_split:
        if s == '(':
            postack.push(s)

        elif s == ')':
            while postack.peek() != '(':
                output.append(postack.pop())
            postack.pop()

        elif s in operator:
            while not postack.is_empty() and (prec[postack.peek()] >= prec[s]):
                output.append(postack.pop())
            postack.push(s)

        else:
            output.append(s)

    while not postack.is_empty():
        output.append(postack.pop())

    return ' '.join(output)


if __name__ == '__main__':
    # print(infix_to_postfix("A * B + C * D"))
    # print(infix_to_postfix("( A + B ) * C - ( D - E ) * ( F + G )"))
    # print(infix_to_postfix("( A + B ) * ( C + D )"))
    # print(infix_to_postfix(' 10 + 3 * 5 / ( 16 - 4 )'))
    print(infix_to_postfix(' 5 * 3 ** ( 4 - 2 )'))
