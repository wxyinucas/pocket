def sum_list(alist):
    if len(alist) == 1:
        return alist[-1]
    else:
        return alist[-1] + sum_list(alist[:-1])


if __name__ == '__main__':
    print(sum_list([2, 3, 1, 4, 1]))
