import random

arr = [random.randint(0, 10) for i in range(10)]

def binarySearch(a:list, n):
    low = 0
    high = len(a) - 1
    while low <= high:
        mid = (low + high) // 2
        if n == a[mid]:
            return mid
        elif n < a[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return -1

print(arr, "\n", binarySearch(arr, 10))