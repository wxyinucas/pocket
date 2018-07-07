# Down to Zero II

numbers = [int(input()) for _ in range(int(input()))]
num_max = max(numbers)

moves = [-1] * (num_max + 1)
moves[:4] = list(range(4))

for i in range(num_max+1):
    if moves[i] == -1 or moves[i] > moves[i-1] + 1:
        moves[i] = moves[i-1] + 1
    for j in range(1, i+1):
        if i*j > num_max:
            break
        if moves[i*j] == -1 or moves[i*j] > moves[i] + 1:
            moves[i*j] = moves[i] + 1

for num in numbers:
    print(moves[num])


