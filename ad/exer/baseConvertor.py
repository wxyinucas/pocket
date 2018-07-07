from ds import Stack


def conver_base(number, base=2):
    digit = '0123456789ABCDEFGHIJKLMNOPQRST'
    rem = Stack()

    while number > 0:
        rem.push(number % base)
        number = number // base

    new_str = ''
    while not rem.is_empty():
        new_str = new_str + digit[rem.pop()]

    return new_str


if __name__ == '__main__':
    print(conver_base(14, 13))
