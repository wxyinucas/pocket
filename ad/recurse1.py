import re


def to_str(n, base):
    digits = '01234567890ABCDEFG'

    if n < base:
        return digits[n]
    else:
        return to_str(n // base, base) + digits[n % base]


def reverse(string):
    if len(string) == 1:
        return string
    else:
        return reverse(string[1:]) + string[0]


def check_abcba(string):

    if len(string) <= 1:
        return True
    elif string[0].lower() == string[-1].lower():
        return check_abcba(string[1:-1])
    else:
        return False


def pre(string):
    return re.sub(r'\W', '', string)


if __name__ == '__main__':
    # print(to_str(10, 2))
    # print(reverse('wangxioayu'))
    # print(check_abcba('asddsa'))

    tar = ['kayak',
           'aibohphobia',
           'Live not on evil',
           'Reviled    did    I    live, said    I, as evil    I    did    deliver',
           'Go    hang    a    salami;    I’m    a    lasagna    hog.',
           'Able    was    I    ere    I    saw    Elba',
           'Kanakanak',
           'Wassamassaw']
    for i in tar:
        print(check_abcba(pre(i)))
