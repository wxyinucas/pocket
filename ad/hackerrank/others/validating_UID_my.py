# https://www.hackerrank.com/challenges/validating-uid/problem?h_r=next-challenge&h_v=zen

import re


def check_UID(uid):
    if len(re.findall(r'[A-Z]', uid)) < 2 or len(re.findall(r'[0-9]', uid)) < 3 or len(re.findall(r'[^A-Za-z0-9]', uid)) \
            or len(uid) != 10 or len(set(uid)) < 10:
        return 'Invalid'
    else:
        return 'Valid'


if __name__ == '__main__':
    for _ in range(int(input())):
        print(check_UID(input()))
