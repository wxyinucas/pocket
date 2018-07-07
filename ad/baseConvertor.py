from ds import Stack


def base_convert(decNumber, base=2):
    digit = '0123456789ABCDEF'

    if base not in [2, 8, 16]:
        raise RuntimeError('{} is not proper base, which should be 2, 8 or 16.'.format(base))

    rem = Stack()

    while decNumber > 0:
        rem.push(decNumber % base)
        decNumber = decNumber // base

    new_string = ''
    while not rem.isEmpty():
        new_string = new_string + digit[rem.pop()]

    return new_string


if __name__ == '__main__':
    print(base_convert(256, 16))
