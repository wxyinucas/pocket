# https://www.hackerrank.com/challenges/print-the-elements-of-a-linked-list-in-reverse/problem

def reverse_print(head):
    if head:
        reverse_print(head.next)
    print(head.data)


# https: // www.hackerrank.com / challenges / reverse - a - linked - list / forum
# 多看几眼，递归挺烦的。。

def reverse(head):
    if not head or not head.next:
        return head
    newHead = reverse(head.next)
    head.next.next = head
    head.next = None
    return newHead
