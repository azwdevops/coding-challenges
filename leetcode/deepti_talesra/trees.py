from typing import Optional, List
from collections import defaultdict


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# leetcode 235
class Solution235:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root


# leetcode 98
class Solution98:

    def helper(self, root, left, right):
        if root is None:
            return True
        if not (left < root.val < right):
            return False
        return self.helper(root.left, left, root.val) and self.helper(root.right, root.val, right)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return self.helper(root, float("-inf"), float("inf"))


# leetcode 208
class Trie:

    def __init__(self):
        self.children = {}
        self.word = False

    def insert(self, word: str) -> None:
        root = self
        for char in word:
            if char not in root.children:
                root.children[char] = Trie()
            root = root.children[char]
        root.word = True

    def search(self, word: str) -> bool:
        root = self
        for char in word:
            if char not in root.children:
                return False
            root = root.children[char]
        return root.word

    def startsWith(self, prefix: str) -> bool:
        root = self
        for char in prefix:
            if char not in root.children:
                return False
            root = root.children[char]
        return True


# leetcode 104
class Solution104:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))


# leetcode 1161
class Solution1161:

    def helper(self, root, level):
        if root:
            self.level[level] += root.val
            self.helper(root.left, level + 1)
            self.helper(root.right, level + 1)

    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        # solution 1 recursively
        self.level = defaultdict(int)
        self.helper(root, 1)
        return sorted(self.level.items(), key=lambda x: x[1], reverse=True)[0][0]

        # solution 2 iteratively
        level_num = 1
        max_sum = root.val
        level = [root]
        return_level = 1
        while level != []:
            next_level = []
            level_sum = 0
            level_num += 1
            for roots in level:
                if roots.left:
                    next_level.append(roots.left)
                    level_sum += roots.left.val
                if roots.right:
                    next_level.append(roots.right)
                    level_sum += roots.right.val
            if next_level != [] and level_sum > max_sum:
                max_sum = level_sum
                return_level = level_num
            level = next_level
        return return_level


# leetcode 662
class Solution662:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        max_width = 1
        current_level = [(root, 1)]  # root, position
        while current_level != []:
            next_level = []
            for roots, position in current_level:
                if roots.left:
                    next_level.append((roots.left, position * 2))
                if roots.right:
                    next_level.append((roots.right, position * 2 + 1))
            if next_level != []:
                max_width = max(max_width, next_level[-1][1] - next_level[0][1] + 1)
            current_level = next_level
        return max_width


# leetcode 199
class Solution199:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        level = []
        queue = [root]
        while queue != [] and root is not None:
            for node in queue:
                if node.left:
                    level.append(node.left)
                if node.right:
                    level.append(node.right)
            result.append(node.val)
            queue = level
            level = []
        return result


# leetcode 124
class Solution124:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.maxPathSumSoFar = float("-inf")

        def pathSum(root):
            if not root:
                return 0
            left = max(0, pathSum(root.left))
            right = max(0, pathSum(root.right))

            self.maxPathSumSoFar(self.maxPathSumSoFar, left + right + root.val)
            return max(left, right) + root.val

        pathSum(root)
        return self.maxPathSumSoFar


# leetcode 226
class Solution226:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return
        root.right, root.left = self.invertTree(root.left), self.invertTree(root.right)

        return root


# leetcode 230
class Solution230:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:

        # solution 1
        self.k = k
        self.result = None

        def inOrder(root):
            if root is None or self.result is not None:
                return
            inOrder(root.left)
            self.k -= 1
            if self.k == 0:
                self.result = root.val
                return
            inOrder(root.right)

        inOrder(root)
        return self.result

        # solution 2
        stack = []
        result = None
        while stack or root:
            while root:
                stack.apend(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right


# leetcode 101
class Solution101:

    # solution 1
    def check(self, left, right):
        if left is None and right is None:
            return True
        if left is None or right is None or left.val != right.val:
            return False
        return self.check(left.left, right.right) and self.check(left.right, right.left)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        # solution 1
        # return self.check(root.left, root.right)

        # solution 2
        stack = [(root.left, root.right)]
        while stack:
            left, right = stack.pop()
            if left is None and right is None:
                continue
            if left is None or right is None or left.val != right.val:
                return False
            stack.append((left.left, right.right))
            stack.append((left.right, right.left))

        return True


# leetcode 700
class Solution700:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # solution 1
        if root is None:
            return None

        if root.val == val:
            return root

        if val < root.val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

        # solution 2
        current = root
        while current and current.val != val:
            if val < current.val:
                current = current.left
            else:
                current = current.right
            return current


# leetcode 872
class Solution872:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        leaf1 = []
        leaf2 = []

        def getLeaf(root, result):
            if root.left is None and root.right is None:
                result.append(root.val)
                return
            if root.left:
                getLeaf(root.left, result)
            if root.right:
                getLeaf(root.right, result)
            return

        getLeaf(root1, leaf1)
        getLeaf(root2, leaf2)

        return leaf1 == leaf2
