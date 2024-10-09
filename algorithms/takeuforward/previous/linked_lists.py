from typing import Optional


class ListNode:
    def __init__(self, data=0, next=None):
        self.data = data
        self.next = next


def addOne(head: ListNode):
    # reverse list first
    def reverseList(node):
        current = node
        prev = None
        while current:
            temp = current.next
            current.next = prev

            prev = current
            current = temp

        return prev


class SolutionReverseKNodes:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverseList(node, k):
            current = node
            prev = None
            count = 0

            while current and count < k:
                temp = current.next
                current.next = prev

                prev = current
                current = temp
                count += 1

            # the last prev is the new_head, node will be the new_tail, current will represent the next_node if not None
            # note that we stop before reversing the whole list in case where k is greater than remaining nodes

            new_head = prev
            new_tail = node
            next_node = current
            return new_head, new_tail, next_node

        # helper function to check if k nodes are available and thus reverse can be done
        def hasKNodes(node, k):
            count = 0
            current = node
            while current and count < k:
                current = current.next
                count += 1
            return count == k

        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head

        while current:
            if hasKNodes(current, k):
                new_head, new_tail, next_node = reverseList(current, k)
                prev.next = new_head
                new_tail.next = next_node

                prev = new_tail
                current = next_node

            else:
                break

        return dummy.next
