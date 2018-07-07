from ds import Deque


def pal_checker(string):
    str_deque = Deque()

    for s in string:
        str_deque.add_front(s)

    while not len(str_deque) in [0, 1]:
        if str_deque.remove_front() != str_deque.remove_rear():
            return False

    return True


if __name__ == '__main__':
    print(pal_checker("lsdkjfskf"))
    print(pal_checker("radar"))