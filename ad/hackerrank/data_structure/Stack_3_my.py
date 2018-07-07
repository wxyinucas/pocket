from collections import deque


# def twoStacks(x, a, b):
#     length = []
#     picked = deque([])
#     a, b = list(map(deque, [a, b]))
#     cumsum = 0
#
#     while cumsum < x and a != deque([]):
#         # print(g_itr)
#         cumsum += a[0]
#         picked.append(a.popleft())
#         length.append(len(picked))
#
#     if cumsum > x:
#         cumsum -= picked.pop()
#
#     length.append(len(picked))
#     counter_num = len(picked)
#
#     while counter_num:
#         if cumsum >= x:
#             counter_num -= 1
#             cumsum -= picked[-1]
#             picked.popleft()
#             length.append(len(picked))
#         else:
#             if b == deque([]):
#                 break
#             while cumsum < x:
#                 cumsum += b[0]
#                 picked.appendleft(b.popleft())
#
#     return max(length)

def twoStacks(x, a, b):
    apicked = deque([])
    a, b = list(map(deque, [a, b]))
    cumsum = 0

    for i in range(len(a)):
        if a[0] + cumsum > x:
            break

        cumsum += a[0]
        apicked.append(a.popleft())

    max_len = len(apicked)
    cur_len = max_len

    m = len(b)
    while m:
        if b[0] + cumsum <= x:
            cumsum += b.popleft()
            cur_len += 1
            m -= 1
            max_len = cur_len if cur_len > max_len else max_len
            continue
        if not len(apicked):
            break
        cumsum -= apicked.pop()
        cur_len -= 1

    return max_len


if __name__ == '__main__':

    g = int(input())

    for g_itr in range(g):
        nmx = input().split()

        n = int(nmx[0])

        m = int(nmx[1])

        x = int(nmx[2])

        a = list(map(int, input().rstrip().split()))

        b = list(map(int, input().rstrip().split()))

        result = twoStacks(x, a, b)
        print(result)
