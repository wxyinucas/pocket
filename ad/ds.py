from collections import deque


class Stack:

    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        return self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        try:
            return self.items[-1]
        except IndexError:
            return 'None'

    def size(self):
        return len(self.items)


class Queue:

    def __init__(self):
        self.items = deque([])

    def __repr__(self):
        return ','.join(list(map(str, self.items)))

    def is_empty(self):
        return self.items == deque([])

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        return self.items.popleft()

    def size(self):
        return len(self.items)


class Deque:
    """
        Front is left.
    """

    def __init__(self):
        self.items = deque([])

    def __str__(self):
        return ' '.join(list(map(str, self.items)))

    def __len__(self):
        return len(self.items)

    def add_front(self, item):
        self.items.appendleft(item)

    def add_rear(self, item):
        self.items.append(item)

    def remove_front(self):
        return self.items.popleft()

    def remove_rear(self):
        return self.items.pop()

    def is_empty(self):
        return self.items == deque([])

    def size(self):
        return len(self.items)


################################################
#
# Linked List
#
################################################

class Node:

    def __init__(self, init_data):
        self.data = init_data
        self.next = None

    def __repr__(self):
        return 'N:{}'.format(self.data)

    def get_data(self):
        return self.data

    def get_next(self):
        return self.next

    def set_data(self, data):
        self.data = data

    def set_next(self, next):
        self.next = next


class UnorderedList:

    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def add(self, item):
        tmp = Node(item)
        tmp.set_next(self.head)
        self.head = tmp

    def __len__(self):
        current = self.head
        count = 0

        while current is not None:
            count += 1
            current = current.get_next()

        return count

    def search(self, item):
        current = self.head
        found = False

        while current is not None and not found:
            if current.get_data() == item:
                found = True
            else:
                current = current.get_next()

        return found

    def remove(self, item):
        current = self.head
        previous = None
        found = False

        while current is not None and not found:
            if current.get_data() == item:
                found = True
            else:
                previous = current
                current = current.get_next()

        if current is None:
            return 'Not found.'

        if previous is None:
            self.head = current.get_next()
        else:
            previous.set_next(current.get_next())

    def append(self, item):
        current = self.head
        tmp = Node(item)

        while current.get_next() is not None:
            current = current.get_next()

        current.set_next(tmp)


