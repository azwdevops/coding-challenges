from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# leetcode 143
class Solution143:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head.next or not head.next.next:
            return
        # split the list
        mid = end = head
        while end.next and end.next.next:
            end = end.next.next
            mid = mid.next
        p2 = mid.next
        mid.next = None
        # reverse the list
        prev = None
        while p2 and p2.next:
            p2Next = p2.next
            p2.next = prev
            prev = p2
            p2 = p2Next
        p2.next = prev

        # merge
        p1 = head
        while p1 and p2:
            p1Next = p1.next
            p2Next = p2.next
            p1.next = p2
            p2.next = p1Next
            p1 = p1Next
            p2 = p2Next


# leetcode 237
class Solution237:
    def deleteNode(self, node):
        node.val = node.next.val
        node.next = node.next.next


# leetcode 141
class Solution141:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head is None:
            return False
        single = double = head
        while double and double.next is not None:
            double = double.next.next
            single = single.next
            if double == single:
                return True

        return False


# leetcode 21
class Solution21:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode(None)
        current = head
        while list1 or list2:
            if list1 is None:
                current.next = list2
                return head.next
            if list2 is None:
                current.next = list1
                return head.next
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        return head.next


# leetcode 19
class Solution19:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:

        # solution 1

        dummy = ListNode(0, head)
        p1 = head
        p2 = dummy
        counts = 0
        while p1 != None:
            if counts > n:
                p2 = p2.next
            p1 = p1.next
            counts += 1
        p2.next = p2.next.next
        # return dummy.next

        # solution 2
        index = self.removeNode(head, n)
        if index == n:
            return head.next
        return head

    def removeNode(self, node, n):
        if node is None:
            return 0
        index = self.removeNode(node.next, n) + 1
        if index == n + 1:
            node.next = node.next.next
        return index


# leetcode 206
class Solution206:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # solution 1
        prev = None
        current = head
        while current != None:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        # return prev

        # solution 2
        return self.helper(None, head)

    def helper(self, prev, current):
        if current is None:
            return prev
        next_node = current.next
        current.next = prev
        new_head = self.helper(current, next_node)

        return new_head


# leetcode
