from typing import List, Optional


# leetcode 894
# we define a binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# time complexity O(m * n)
# space complexity O(n)
class SolutionRecursion894:
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        # memoization table to stiore results of subproblems
        memo = {}

        def generateFBT(num_nodes):
            if num_nodes in memo:
                return memo[num_nodes]
            if num_nodes == 1:
                return [TreeNode(0)]

            result = []
            for left_nodes in range(1, num_nodes, 2):
                right_nodes = num_nodes - 1 - left_nodes
                left_trees = generateFBT(left_nodes)
                right_trees = generateFBT(right_nodes)

                for left in left_trees:
                    for right in right_trees:
                        root = TreeNode(0)
                        root.left = left
                        root.right = right
                        result.append(root)
            memo[num_nodes] = result
            return result

        if n % 2 == 0:
            return []  # full binary trees cannot have an even number of nodes
        return generateFBT(n)


# solution = SolutionRecursion894()
# print(solution.allPossibleFBT(7))


# leetcode 894
class Solution894:
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        # base case if n == 1, return a single-node tree
        if n == 1:
            return [TreeNode(0)]
        result = []
        # iterate over all possible sizes for the left subtree
        for left_size in range(1, n, 2):
            right_size = n - 1 - left_size

            # recursively generate all possible left and right subtrees
            left_trees = self.allPossibleFBT(left_size)
            right_trees = self.allPossibleFBT(right_size)

            # combine each left and right subtree
            for left in left_trees:
                for right in right_trees:
                    root = TreeNode(0)
                    root.left = left
                    root.right = right
                    result.append(root)
        return result


# leetcode 1823
class Solution1823:
    def findTheWinner(self, n: int, k: int) -> int:
        def recursive_call(n, k):
            # base case: only one friend left
            if n == 1:
                return 0
            else:
                # recursive case find the winner for n - 1 friends and adjust for the nth friend
                return (recursive_call(n - 1, k) + k) % n

        # since the problem assumes 1-based indexing, we need to adjust the result by adding 1
        return recursive_call(n, k) + 1


# leetcode 2487
class Solution2487:
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def reverse_list(head: ListNode) -> ListNode:
            prev = None
            current = head

            while current:
                next_node = current.next
                current.next = prev
                prev = current
                current = next_node

            return prev

        # step 1 reverse the list
        head = reverse_list(head)

        # step 2 remove nodes that have a smaller value than the maximum seen so far
        max_value = float("-inf")
        prev = None
        current = head

        while current:
            if current.val >= max_value:
                max_value = current.val
                prev = current
            else:
                prev.next = current.next

        # step 3  reverse the list back to restore original order
        head = reverse_list(head)

        return head


# leetcode 241
class Solution241:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        memo = {}

        def compute(exp: str) -> List[int]:
            if exp in memo:
                return memo[exp]

            results = []
            for i, char in enumerate(exp):
                if char in {"+", "-", "*"}:
                    # divide: split expression into left and right parts
                    left_results = compute([exp[i:]])
                    right_results = compute(exp[i + 1 :])

                    # conquer and combine: compute all combinations
                    for left in left_results:
                        for right in right_results:
                            result = apply_operator(left, right, char)
                            results.append(result)

            # base case: exp is a single number
            if not results:
                results.append(int(exp))
            memo[exp] = results
            return results

        def apply_operator(a: int, b: int, operator: str) -> int:
            if operator == "+":
                return a + b
            elif operator == "-":
                return a - b
            elif operator == "*":
                return a * b
            else:
                raise ValueError(f"Invalid operator {operator}")

        return compute(expression)


# leetcode 24
class Solution24:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # base case if the list is empty or has only one node, return the head
        if not head or not head.next:
            return head
        # identity the new head after the swap
        new_head = head.next
        # recursively swap the rest of the list
        head.next = self.swapPairs(new_head.next)

        # complete the swap by linking the new head to the current node
        new_head.next = head

        # return the new head of the list
        return new_head


# leetcode 143
class Solution143:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return
        # step 1 find the middle of the linked list
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # step 2 reverse the second half of the list
        prev, curr = None, slow.next
        slow.next = None  # split the list into two halves
        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp

        # step 3 merge the two halves
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first = tmp1
            second = tmp2


# leetcode 1545
class Solution1545:
    def findKthBit(self, n: int, k: int) -> str:
        # base case: if n is 1 the string is "0" and the first bit is "0"
        if n == 1:
            return "0"
        # calculate the middle position (length of S_{n - 1} + 1)
        mid = 2 ** (n - 1)

        if k == mid:
            return "1"  # the middle bit is always 1
        elif k < mid:
            return self.findKthBit(n - 1, k)  # the kth bit is in the left segment
        else:
            # the kth bit is in the right segment, calculate its position in S_{n - 1}
            return "0" if self.findKthBit(n - 1, 2 * mid - k) == "1" else "1"
