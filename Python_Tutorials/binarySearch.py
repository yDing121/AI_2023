import random


def find(arr:list, n):
    lo = 0
    hi = len(arr) - 1
    mid = (lo + hi)//2

    while lo <= hi:
        if n < arr[mid]:
            hi = mid - 1
        elif n == arr[mid]:
            return mid
        else:
            lo = mid + 1
        mid = (lo + hi) // 2
    return -1


a = [random.randint(-10, 10) for i in range(10)]
a.sort()
print(a)
print(find(a, 0))
