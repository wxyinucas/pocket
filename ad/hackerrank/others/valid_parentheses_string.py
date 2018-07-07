# valid parentheses string


def check(string):
    star_queue = []
    left_queue = []

    for loc, opr in enumerate(string):
        if opr == '(':
            left_queue.append([opr, loc])
        elif opr == '*':
            star_queue.append([opr, loc])
        else:
            if len(left_queue):
                left_queue.pop()
            elif len(star_queue):
                star_queue.pop()
            else:
                return False

    while len(left_queue):
        if not len(star_queue):
            return False
        if left_queue[-1][1] > star_queue[-1][1]:
            return False
        else:
            left_queue.pop()
            star_queue.pop()

    return True


if __name__ == '__main__':
    string = input('Type in the string you want to test:')
    print(check(string))
