from ds import Queue


def hot_potato(namelist, num):
    queue = Queue()

    for name in namelist:
        queue.enqueue(name)

    while queue.size() > 1:
        for i in range(num):
            queue.enqueue(queue.dequeue())
        queue.dequeue()

    return queue.dequeue()


if __name__ == '__main__':
    print(hot_potato(["Bill", "David", "Susan", "Jane", "Kent", "Brad"], 7))
