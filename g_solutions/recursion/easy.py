from typing import Optional, List
import math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# leetcode 1763
class Solution1763:
    def longestNiceSubstring(self, s: str) -> str:
        if not s:
            return ""

        def is_nice(sub: str) -> bool:
            char_set = set(sub)
            for char in char_set:
                if char.swapcase() not in char_set:
                    return False
            return True

        def longest_nice_helper(s: str) -> str:
            if is_nice(s):
                return s
            max_nice = ""
            for i in range(len(s)):
                if s[i].swapcase() not in s:
                    left = longest_nice_helper(s[:i])
                    right = longest_nice_helper(s[i + 1 :])
                    max_nice = max(max_nice, right, left)
                    break  # no need to check further as we have already split the string
            return max_nice

        return longest_nice_helper(s)


# solution = Solution1763()
# print(solution.longestNiceSubstring('YazaAay'))
# print(solution.longestNiceSubstring('Bb'))
# print(solution.longestNiceSubstring('c'))


# leetcode 234
class Solution234:
    def isPalidrome(self, head: Optional[ListNode]) -> bool:
        if head is None or head.next is None:
            return True

        # find the middle of the list
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # reverse the second half
        second_half = reverseList(slow)

        def reverseList(head):
            prev = None
            current = head
            while current:
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node
            return prev

        # compare the first half and the reversed second half
        first_half = head
        while second_half:
            if first_half.val != second_half.val:
                return False
            first_half = first_half.next
            second_half = second_half.next

        return True


# leetcode 234
class Solution234:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        # this pointer will move from the front of the list
        global front_pointer
        front_pointer = head

        def recursive_check(current_node: ListNode) -> bool:
            # base case: if the current node is None, we're at the end of the list
            if current_node is None:
                return True
            # recursively check the next node
            is_palidrome_so_far = recursive_check(current_node.next)

            # check the current value with the front pointer's value
            is_palidrome_now = is_palidrome_so_far and (front_pointer.val == current_node.val)

            # move the front pointer to the next node
            global front_pointer
            front_pointer = front_pointer.next

            return is_palidrome_now

        # start the recursive check
        return recursive_check(head)


# leetcode 3211
class Solution3211:
    def validStrings(self, n: int) -> List[str]:
        # base case: when length is 1, the valid strings are 0 and 1
        if n == 1:
            return ["0", "1"]
        # recursively generate strings of length n - 1
        previous_strings = self.validStrings(n - 1)
        valid_strings = []
        # extend each string based on the last character
        for string in previous_strings:
            if string[-1] == "1":
                valid_strings.append(string + "0")
                valid_strings.append(string + "1")
            elif string[-1] == "0":
                valid_strings.append(string + "1")
        return valid_strings
