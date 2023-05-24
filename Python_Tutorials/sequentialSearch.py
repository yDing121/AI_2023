import random


def find(arr: list, n):
    for i in range(len(arr)):
        if arr[i] == n:
            return i
    return -1


a = [random.randint(-10,10) for i in range(10)]

print(find(a, 0))
