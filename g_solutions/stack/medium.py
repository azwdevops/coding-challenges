from typing import List, Optional


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# leetcode 1008
class Solution1008:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return None
        # step 1: initialize root and stack
        root = TreeNode(preorder[0])
        stack = [root]
        # step 2 iterate through the preorder list
        for value in preorder[1:]:
            node = stack[-1]
            if value < node.val:
                # the value is less, it is the left child of the last node
                node.left = TreeNode(value)
                stack.append(node.left)
            else:
                # the value is greater, find the right position in the stack
                while stack and stack[-1].val < value:
                    node = stack.pop()
                node.right = TreeNode(value)
                stack.append(node.right)

        return root


# leetcode 2130
class Solution2130:
    def pairSum(self, head: Optional[ListNode]) -> int:
        # step 1 find the middle of the linked list using slow and fast pointers
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # step 2 reverse the second half of the linked list
        prev = None
        while slow:
            temp = slow.next
            slow.next = prev
            prev = slow
            slow = temp

        # step 3 calculate the maximum twin sum
        max_sum = 0
        left, right = head, prev
        while right:
            max_sum = max(max_sum, left.val + right.val)
            left = left.next
            right = right.next

        return max_sum


# leetcode 1441
class Solution1441:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        operations = []
        target_index = 0
        for num in range(1, n + 1):
            # push operation
            operations.append("Push")

            if num == target[target_index]:
                # move to the next number in the target
                target_index += 1
                if target_index == len(target):
                    break
            else:
                # pop operation if not matching
                operations.append("Pop")

        return operations


# leetcode 1472
class BrowserHistory:

    def __init__(self, homepage: str):
        self.history = [homepage]
        self.current = 0

    def visit(self, url: str) -> None:
        # clear forward history by slicing the list to the current + 1
        self.history = self.history[: self.current + 1]
        # add the new url to the history
        self.history.append(url)
        # update the current position
        self.current += 1

    def back(self, steps: int) -> str:
        # move back by steps, but not before the start
        self.current = max(0, self.current - steps)
        # return the current URL
        return self.history[self.current]

    def forward(self, steps: int) -> str:
        # move forward by steps, but not beyond the last visited page
        self.current = min(len(self.history) - 1, self.current + steps)
        # return the current URL
        return self.history[self.current]


# leetcode 173
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        self._push_left(root)

    def _push_left(self, node: TreeNode):
        # push all left nodes onto the stack
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        # the next smallest element is at the top of the stack
        node = self.stack.pop()
        # if this node has a right child, push all left nodes of the right subtree
        if node.right:
            self._push_left(node.right)
        # return the value of the current node
        return node.val

    def hasNext(self) -> bool:
        # if the stack is not empty, there are more nodes to visit
        return len(self.stack) > 0


# leetcode 1111
class Solution1111:
    def maxDepthAfterSplit(self, seq: str) -> List[int]:
        depth = 0
        answer = []
        for char in seq:
            if char == "(":
                depth += 1
                if depth % 2 == 1:
                    answer.append(0)  # assign to A
                else:
                    answer.append(1)  # assign to B
            else:
                if depth % 2 == 1:
                    answer.append(0)  # assign to A
                else:
                    answer.append(1)  # assign to B
                depth -= 1

        return answer


# leetcode 1190
class Solution1190:
    def reverseParentheses(self, s: str) -> str:
        stack = []
        for char in s:
            if char == ")":
                temp = []
                # pop characters until we find the matching (
                while stack and stack[-1] != "(":
                    temp.append(stack.pop())
                # pop the (
                stack.pop()
                # push the reversed substring back onto the stack
                stack.extend(temp)
            else:
                stack.append(char)

        # join the stack into the final string
        return "".join(stack)


# leetcode 946
class Solution946:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        j = 0
        for value in pushed:
            stack.append(value)
            # Check if the current top of the stack matches the popped sequence
            while stack and stack[-1] == popped[j]:
                stack.pop()
                j += 1
        return True if stack == [] else False


# leetcode 1249
class Solution1249:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        to_remove = set()
        # first pass to identity indices to remove
        for i, char in enumerate(s):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    stack.pop()
                else:
                    to_remove.add(i)
        # add any unmatched ( to the removal set
        while stack:
            to_remove.add(stack.pop())
        # second pass to build the valid string
        result = []
        for i, char in enumerate(s):
            if i not in to_remove:
                result.append(char)
        return "".join(result)
