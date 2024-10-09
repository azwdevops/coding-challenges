from typing import Optional, List
from collections import defaultdict, deque


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
    def helper(self, node, left, right):
        if node is None:
            return True
        if not (left < node.val < right):
            return False
        return self.helper(node.left, left, node.val) and self.helper(node.right, node.val, right)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return self.helper(root, float("-inf"), float("float"))


# leetcode 208


class Trie28:

    def __init__(self):
        self.children = {}
        self.word = False

    def insert(self, word: str) -> None:
        root = self
        for char in word:
            if char not in root.children:
                root.children[char] = Trie28()
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
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        # solution 1
        level_num = 1
        max_sum = root.val
        level = [root]
        ret = 1
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
                    level_sum += roots.left.val
            if next_level != [] and level_sum > max_sum:
                max_sum = level_sum
                ret = level_num
            level = next_level

        # return ret

        # solution 2
        self.level = defaultdict(int)
        self.helper(root, 1)
        return sorted(self.level.items(), key=lambda x: x[1], reverse=True)[0][0]

    def helper(self, node, level):
        if node:
            self.level[level] += node.val
            self.helper(node.left, level + 1)
            self.helper(node.right, level + 1)


# leetcode 662
class Solution662:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        max_width = 1
        current_level = [(root, 1)]  # root, position
        while current_level != []:
            next_level = []
            for node, position in current_level:
                if node.left:
                    next_level.append((node.left, position * 2))
                if node.right:
                    next_level.append((node.right, position * 2 + 1))
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


# leetcode 103
class Solution103:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        s1 = [root]
        s2 = []
        level = []
        result = []
        while s1 or s2:
            while s1:
                node = s1.pop()
                level.append(node.val)
                if node.left:
                    s2.append(node.left)
                if node.right:
                    s2.append(node.right)
            result.append(level)
            level = []
            while s2:
                node = s2.pop()
                level.append(node.val)
                if node.right:
                    s1.append(node.right)
                if node.left:
                    s1.append(node.left)
            if level != []:
                result.append(level)
                level = []

        return result


# leetcode 102
class Solution102:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        queue = [root]
        next_queue = []
        level = []
        result = []
        while queue != []:
            for node in queue:
                level.append(node.val)
                if node.left is not None:
                    next_queue.append(node.left)
                if node.right is not None:
                    next_queue.append(node.right)
            result.append(level)
            level = []
            queue = next_queue
            next_queue = []

        return result


# leetcode 144
class Solution144:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # solution 1
        if root is None:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

        # solution 2
        if root is None:
            return []
        stack = [root]
        result = []
        while stack != []:
            node = stack.pop()
            result.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return result


# leetcode 94
class Solution94:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # solution 1
        if not root:
            return []
        # return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

        # solution 2
        stack = []
        result = []
        while root or stack:
            while root is not None:
                stack.append(root)
                root = root.left
            root = stack.pop()
            result.append(root.val)
            root = root.right

        return result


# leetcode 100
class Solution100:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # solution 1
        if p is None and q is None:
            return True
        if (p and q and p.val != q.val) or (p is None or q is None):
            return False
        return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)


# leetcode 572
class Solution572:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if subRoot is None:
            return True
        if root is None:
            return False
        if self.sameTree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def sameTree(self, p, q):
        if p is None and q is None:
            return True
        if p is None or q is None or p.val != q.val:
            return False
        return self.sameTree(p.left, q.left) and self.sameTree(p.right, q.right)


# leetcode 124
class Solution124:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.maxSum = float("-inf")

        def pathSum(node):
            if not node:
                return 0
            left = max(0, pathSum(node.left))
            right = max(0, pathSum(node.right))
            self.maxSum = max(self.maxSum, left + right + node.val)

            return max(left, right) + node.val

        return self.maxSum


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

        def inOrder(node):
            if not node or self.result is not None:
                return
            inOrder(node.left)
            self.k -= 1
            if self.k == 0:
                self.result = node.val
                return
            inOrder(node.right)

        inOrder(root)
        # return self.result

        # solution 2
        stack = []
        while stack or root:
            while root:
                stack.append(root.left)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right


# leetcode 101
class Solution101:
    def check(self, left, right):
        if left is None and right is None:
            return True
        if left is None or right is None or left.val != right.val:
            return False
        return self.check(left.left, right.right) and self.check(left.right, right.left)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        # solution 1
        return self.check(root.left, root.right)

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
            if val > current.val:
                current = current.right
        return current


# leetcode 872
class Solution872:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        leaf1 = []
        leaf2 = []

        def getLeaf(node, result):
            if node.left is None and node.right is None:
                result.append(node.val)
                return
            if node.left:
                getLeaf(node.left, result)
            if node.right:
                getLeaf(node.right, result)
            return

        getLeaf(root1, leaf1)
        getLeaf(root2, leaf2)

        return leaf1 == leaf2
