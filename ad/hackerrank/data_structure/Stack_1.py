# Balanced Brackets
# https://www.hackerrank.com/challenges/balanced-brackets/problem
#
# 这个练习是检测括号匹配是否正确，算法很巧妙：
#    若stack为空，或末尾不是相对应的左符号，则添加，否则消去。


table = {')': '(', ']': '[', '}': '{'}

for _ in range(int(input())):
    stack = []
    for x in input():
        if stack and table.get(x) == stack[-1]:
            stack.pop()
        else:
            stack.append(x)
    print("NO" if stack else "YES")
