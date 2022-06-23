import random

def quicksort1(array, i, j):
    if i >= j:
        return list
    key = array[i]
    low = i
    high = j
    while i < j:
        while i < j and array[j] >= key:
            j -= 1
        array[i] = array[j]
        while i < j and array[i] <= key:
            i += 1
        array[j] = array[i]
    array[j] = key
    quicksort1(array, low, i - 1)
    quicksort1(array, i + 1, high)
    return array


def quicksort2(array, l, r):
    if l < r:
        q = partition(array, l, r)
        quicksort2(array, l, q - 1)
        quicksort2(array, q + 1, r)

def partition(array, l, r):
    x = array[r]
    i = l - 1
    for j in range(l, r):
        if array[j] <= x:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i+1], array[r] = array[r], array[i+1]
    return i+1


if __name__ == '__main__':

    # 测试代码
    data = [random.randint(-100, 100) for _ in range(1000)]
    data1 = data[:]
    data2 = data[:]
    # print(data)
    # quicksort2(data, 0, len(data) - 1)
    # print(data)
    import time
    start1 = time.time()
    quicksort2(data1, 0, len(data1) - 1)
    end1 = time.time()
    print("quick_sort time: {:.7f}".format(end1 - start1))
    start2 = time.time()
    data2.sort()
    end2 = time.time()
    print("sort time: {:.7f}".format(end2 - start2))

