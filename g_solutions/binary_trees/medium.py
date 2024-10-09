from typing import Optional, List
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# leetcode 1038


class Solution1038:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        total_sum = 0

        def reverseInOrder(node):
            nonlocal total_sum
            if node is None:
                return None

            # traverse the right subtree (larger values)
            reverseInOrder(node.right)

            # update the current node's value with the total sum
            total_sum += node.val
            node.val = total_sum

            # traverse the left subtree (smaller values)
            reverseInOrder(node.left)

        reverseInOrder(root)

        return root


# leetcode 1302
class Solution1302:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        queue = deque([root])

        sum_of_deepest_leaves = 0

        while queue:
            sum_of_current_level = 0
            level_size = len(queue)

            for _ in range(level_size):
                node = queue.popleft()
                sum_of_current_level += node.val

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            sum_of_deepest_leaves = sum_of_current_level

        return sum_of_deepest_leaves


# leetcode 2265
class Solution2265:
    def averageOfSubtree(self, root: TreeNode) -> int:
        total_count = 0

        def postOrder(node):
            nonlocal total_count

            if not node:
                return (0, 0)  # (sum, count)
            left_sum, left_count = postOrder(node.left)
            right_sum, right_count = postOrder(node.right)

            subtree_sum = node.val + left_sum + right_sum
            subtree_count = 1 + left_count + right_count

            average = subtree_sum // subtree_count

            if average == node.val:
                total_count += 1
            return subtree_sum, subtree_count

        postOrder(root)

        return total_count


# leetcode 654
class Solution654:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        # find the maximum value and it's index
        max_value = max(nums)
        max_index = nums.index(max_value)

        # create the root node with the maximum value
        root = TreeNode(max_value)

        root.left = self.constructMaximumBinaryTree(nums[:max_index])
        root.right = self.constructMaximumBinaryTree(nums[max_index + 1 :])

        return root


# leetcode 1315
class Solution1315:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        def dfs(node, parent, grandparent):
            if not node:
                return 0
            total = 0
            if grandparent and grandparent.val % 2 == 0:
                total += node.val

            total += dfs(node.left, node, parent)
            total += dfs(node.right, node, parent)

            return total

        return dfs(root, None, None)


# leetcode 1382
class Solution1382:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        # step 1 perform an in-order traversal to get the sortedlist of node values
        def inOrderTraversal(node):
            if not node:
                return []
            return inOrderTraversal(node.left) + [node.val] + inOrderTraversal(node.right)

        # step 2 build a balanced BST from the sorted list
        def buildTree(values, start, end):
            if start > end:
                return None
            mid = (start + end) // 2
            root = TreeNode(values[mid])
            root.left = buildTree(values, start, mid - 1)
            root.right = buildTree(values, mid + 1, end)

            return root

        sorted_values = inOrderTraversal(root)
        return buildTree(sorted_values, 0, len(sorted_values) - 1)


# leetcode 894
class Solution894:
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        if n % 2 == 0:
            return []  # no valid binary trees with an even number of nodes
        if n == 1:
            return [TreeNode(0)]
        result = []
        for left_size in range(1, n, 2):  # left size must be odd and less than n
            right_size = n - 1 - left_size
            left_subtrees = self.allPossibleFBT(left_size)
            right_subtrees = self.allPossibleFBT(right_size)

            for left in left_subtrees:
                for right in right_subtrees:
                    root = TreeNode(0)
                    root.left = left
                    root.right = right
                    result.append(root)

        return result


# leetcode 1008
class Solution1008:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        # option 1 using recursive approach

        if not preorder:
            return None
        root_val = preorder[0]
        root = TreeNode(root_val)

        # elements less than root_val go to the left subtree
        left_subtree = [x for x in preorder if x < root_val]
        # elements greater than root_val fo to the right subtree
        right_subtree = [x for x in preorder if x > root_val]

        root.left = self.bstFromPreorder(left_subtree)
        root.right = self.bstFromPreorder(right_subtree)

        return root

        # option 2 using a stack
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        stack = [root]
        for value in preorder[1:]:
            node, child = stack[-1], TreeNode(value)

            # adjust stack and place the child in the correct spot
            while stack and stack[-1].val < value:
                node = stack.pop()

            if value < node.val:
                node.left = child
            else:
                node.right = child

            stack.append(child)

        return root


# leecode 2196
class Solution2196:
    def createBinaryTree(self, descriptions: List[List[int]]) -> Optional[TreeNode]:
        nodes = {}
        children = set()
        # process each description
        for parent, child, isLeft in descriptions:
            if parent not in nodes:
                nodes[parent] = TreeNode(parent)
            if child not in nodes:
                nodes[child] = TreeNode(child)

            # set the left or right child
            if isLeft == 1:
                nodes[parent].left = nodes[child]
            else:
                nodes[parent].right = nodes[child]
            # track child nodes
            children.add(child)
        # the root is the only node that is not a child
        for node_value in nodes:
            if node_value not in children:
                return nodes[node_value]


# leetcode 1305
class Solution1305:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:

        def inOrderTraversal(root):
            result = []
            stack = []
            current = root

            while stack or current:
                while current:
                    stack.append(current.left)
                    current = current.left
                current = stack.pop()
                result.append(current.val)
                current = current.right
            return result

        list1 = inOrderTraversal(root1)
        list2 = inOrderTraversal(root2)

        def mergeSortedLists(list1, list2):
            result = []
            i, j = 0, 0
            while i < len(list1) and j < len(list2):
                if list1[i] < list2[j]:
                    result.append(list1[i])
                    i += 1
                else:
                    result.append(list2[j])
                    j += 1
            while i < len(list1):
                result.append(list1[i])
                i += 1
            while j < len(list2):
                result.append(list2[j])
                j += 1
            return result

        return mergeSortedLists(list1, list2)


# leetcode 1026
class Solution1026:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        def findMaxAncestorDiff(node, min_val, max_val):
            if node is None:
                return max_val - min_val
            # update the current path's min and max values
            min_val = min(min_val, node.val)
            max_val = max(max_val, node.val)

            # recursively calculate differences for left and right subtrees
            left_diff = findMaxAncestorDiff(node.left, min_val, max_val)
            right_diff = findMaxAncestorDiff(node.right, min_val, max_val)

            # return the maximum difference found
            return max(left_diff, right_diff)

        if root is None:
            return 0

        return findMaxAncestorDiff(root, root.val, root.val)


# leetcode 2415
class Solution2415:
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        queue = deque([root])
        level = 0
        while queue:
            level_size = len(queue)
            current_level_values = []

            # collect values at the current level
            for _ in range(level_size):
                node = queue.popleft()
                current_level_values.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            # reverse values at odd levels
            if level % 2 == 1:
                current_level_values.reverse()
            # put the reversed values back into the tree
            queue = deque([root])
            index = 0

            while queue:
                level_size = len(queue)
                for _ in range(level_size):
                    node = queue.popleft()
                    node.val = current_level_values[index]
                    index += 1
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
            level += 1

        return root


# leetcode 1261
class FindElements1261:

    def __init__(self, root: Optional[TreeNode]):
        self.values = set()
        self.recover(root, 0)

    def recover(self, node: TreeNode, value: int):
        if not node:
            return
        node.val = value
        self.values.add(value)
        if node.left:
            self.recover(node.left, 2 * value + 1)
        if node.right:
            self.recover(node.right, 2 * value + 2)

    def find(self, target: int) -> bool:
        return target in self.values


# leetcode 1325
class Solution1325:
    def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
        if root is None:
            return None
        root.left = self.removeLeafNodes(root.left, target)
        root.right = self.removeLeafNodes(root.right, target)

        if root and root.left is None and root.right is None and root.val == target:
            return None

        return root


# leetcode 979
class Solution979:
    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        moves = 0

        def dfs(node):
            if not node:
                return 0
            # traverse the left and right subtree first (postorder)
            left_balance = dfs(node.left)
            right_balance = dfs(node.right)

            # calculate current balance: coins at this node + balances from subtrees - 1
            current_balance = node.val + left_balance + right_balance - 1

            # update the total moves needed with the absolute value of the current balance
            nonlocal moves
            moves += abs(current_balance)

            # return the current balance
            return current_balance

        dfs(root)

        return moves
