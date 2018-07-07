numbers = [int(input()) for _ in range(int(input()))]
_max = max(numbers)
count = list(range(4)) + [-1] * (_max - 3)

for i in range(_max+1):
    if count[i] == -1 or count[i-1] + 1 < count[i]:
        count[i] = count[i-1] + 1
    for j in range(1, i+1):
        if i * j > _max:
            break
        if count[i*j] == -1 or count[i]+1 < count[i*j]:
            count[i*j] = count[i] + 1


for num in numbers:
    print(count[num])