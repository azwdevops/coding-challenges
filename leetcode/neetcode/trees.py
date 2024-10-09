from typing import Optional, List
from collections import deque
from functools import reduce


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

    def addWord(self, word):
        current = self
        for c in word:
            if c not in current.children:
                current.children[c] = TrieNode()
            current = current.children[c]
        current.endOfWord = True


class Node:
    def __init__(self, val: int = 0, left: "Node" = None, right: "Node" = None, next: "Node" = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


# leetcode 110


class Solution110:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def dfs(root):
            if not root:
                return [True, 0]
            left, right = dfs(root.left), dfs(root.right)
            balanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1

            return [balanced, 1 + max(left[1], right[1])]

        return dfs(root)[0]


# leetcode 1448
class Solution1448:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node, maxVal):
            if not node:
                return 0
            res = 1 if node.val >= maxVal else 0
            maxVal = max(maxVal, node.val)
            res += dfs(node.left, maxVal)
            res += dfs(node.right, maxVal)
            return res

        return dfs(root, root.val)


# leetcode 226
class Solution226:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        # swap the children
        temp = root.left
        root.left = root.right
        root.right = temp
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root


# leetcode 617
class Solution617:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1 and not root2:
            return None
        v1 = root1.val if root1 else 0
        v2 = root2.val if root2 else 0
        root = TreeNode(v1 + v2)

        root.left = self.mergeTrees(root1.left if root1 else None, root2.left if root2 else None)
        root.right = self.mergeTrees(root1.right if root1 else None, root2.right if root2 else None)

        return root


# leetcode 108
class Solution108:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def helper(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)

            return root

        return helper(0, len(nums) - 1)


# leetcode 98
class Solution98:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def valid(node, left, right):
            if not node:
                return True
            if not (node.val < right and node.val > left):
                return False
            return valid(node.left, left, node.val) and valid(node.right, node.val, right)

        return valid(root, float("-inf"), float("inf"))


# leetcode 120
class Solution120:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp = [0] * (len(triangle) + 1)
        for row in triangle[::-1]:
            for i, n in enumerate(row):
                dp[i] = n + min(dp[i], dp[i + 1])

        return dp[0]


# leetcode 129
class Solution129:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(current, num):
            if not current:
                return 0
            num = num * 10 + current.val
            if not current.left and not current.right:
                return num
            return dfs(current.left, num) + dfs(current.right, num)

        return dfs(root, 0)


# leetcode 96
class Solution96:
    def numTrees(self, n: int) -> int:
        # numTrees[4] = numTrees[0] * numTrees[3] + numTrees[1] * numTrees[2] + numTrees[2] * numTrees[1] + numTrees[3] + numTrees[1]
        dp = [1] * (n + 1)
        # 0 node = 1 tree
        # 1 node = 1 tree
        for nodes in range(2, n + 1):
            total = 0
            for root in range(1, nodes + 1):
                left = root - 1
                right = nodes - root
                total += dp[left] * dp[right]
            dp[nodes] = total
        return dp[n]


# leetcode 199
class Solution199:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        queue = deque([root])

        while queue:
            rightSide = None
            queue_length = len(queue)
            for i in range(queue_length):
                node = queue.popleft()
                if node:
                    rightSide = node
                    queue.append(node.left)
                    queue.append(node.right)
            if rightSide:
                result.append(rightSide.val)

        return result


# leetcode 230
class Solution230:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        n = 0
        stack = []
        current = root
        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            n += 1
            if n == k:
                return current.val
            current = current.right


# leetcode 543
class Solution543:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        result = 0

        def dfs(root):
            if not root:
                return -1
            left = dfs(root.left)
            right = dfs(root.right)
            result = max(result, 2 + left + right)

            return 1 + max(left, right)

        dfs(root)

        return result


# leetcode 337
class Solution337:
    def rob(self, root: Optional[TreeNode]) -> int:

        # returns pair [withRoot, withoutRoot]
        def dfs(root):
            if not root:
                return [0, 0]
            leftPair = dfs(root.left)
            rightPair = dfs(root.right)

            withRoot = root.val + leftPair[1] + rightPair[1]
            withoutRoot = max(leftPair) + max(rightPair)

            return [withRoot, withoutRoot]

        return max(dfs(root))


# leetcode 102
class Solution102:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        queue = deque()
        queue.append(root)
        while queue:
            queueLength = len(queue)
            level = []
            for i in range(queueLength):
                node = queue.popleft()
                if node:
                    level.append(node.val)
                    queue.append(node.left)
                    queue.append(node.right)

        if level:
            result.append(level)


# leetcode 105
class Solution105:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1 :], inorder[mid + 1 :])

        return root


# leetcode 208
class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.endOfWord = True

    def search(self, word: str) -> bool:
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.endOfWord

    def startsWith(self, prefix: str) -> bool:
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True


# leetcode 100
class Solution100:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)


# leetcode 297
class Codec297:

    def serialize(self, root):
        result = []

        def dfs(node):
            if not node:
                result.append("N")
                return
            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(result)

    def deserialize(self, data):
        vals = data.split(",")
        self.i = 0

        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()


# leetcode 235
class Solution235:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        current = root
        while current:
            if p.val > current.val and q.val > current.val:
                current = current.right
            elif p.val < current.val and q.val < current.val:
                current = current.left
            else:
                return current


# leetcode 211
class WordDictionary211:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.endOfWord = True

    def search(self, word: str) -> bool:
        def dfs(j, root):
            current = root
            for i in range(j, len(word)):
                char = word[i]
                if char == ".":
                    for child in current.children.values():
                        if dfs(i + 1, child):
                            return True
                    return False
                else:
                    if char not in current.children:
                        return False
                    current = current.children[char]

        return dfs(0, self.root)


# leetcode 124
class Solution124:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        result = [root.val]

        def dfs(root):
            if not root:
                return 0
            leftMax = dfs(root.left)
            rightMax = dfs(root.right)

            leftMax = max(leftMax, 0)
            rightMax = max(rightMax, 0)

            result[0] = max(result[0], root.val + leftMax + rightMax)

            return root.val + max(leftMax, rightMax)

        dfs(root)
        return result[0]


# leetcode 104
class Solution104:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # solution 1
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

        # solution 2
        if not root:
            return 0
        level = 0
        queue = deque([root])
        while queue:
            for i in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            level += 1
        return level

        # solution 3
        stack = [[root, 1]]
        result = 0
        while stack:
            node, depth = stack.pop()
            if node:
                result = max(result, depth)
                stack.append([node.left, depth + 1])
                stack.append([node.right, depth + 1])
        return result


# leetcode 951
class Solution951:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        if not root1 or not root2:
            return not root1 and not root2
        if root1.val != root2.val:
            return False
        a = self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right)
        return a or self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left)


# leetcode 5848
class Solution5848:
    def __init__(self, parent: List[int]):
        self.parent = parent
        self.locked = [None] * len(parent)
        self.child = {i: [] for i in range(len(parent))}
        for i in range(1, len(parent)):
            self.child[parent[i]].append(i)

    def lock(self, num: int, user: int) -> bool:
        if self.locked[num]:
            return False
        self.locked[num] = user
        return True

    def unlock(self, num: int, user: int) -> bool:
        if self.locked[num] != user:
            return False
        self.locked[num] = None
        return True

    def upgrade(self, num: int, user: int) -> bool:
        i = num
        while i != -1:
            if self.locked[i]:
                return False
            i = self.parent[i]

        lockedCount, q = 0, deque([num])
        while q:
            n = q.popleft()
            if self.locked[n]:
                self.locked[n] = None
                lockedCount += 1
            q.extend(self.child[n])
        if lockedCount > 0:
            self.locked[num] = user
        return lockedCount > 0


# leetcode 114
class Solution114:
    def flatten(self, root: Optional[TreeNode]) -> None:
        # flatten the root tree and return the list tail
        def dfs(node):
            if not node:
                return None
            leftTail = dfs(node.left)
            rightTail = dfs(node.right)

            if node.left:
                leftTail.right = node.right
                node.right = root.left
                node.left = None
            last = rightTail or leftTail or node

            return last

        dfs(root)


# leetcode 894
class Solution894:
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        dp = {0: [], 1: [TreeNode()]}  # map n : list of FBT

        # returns the list of fbt with n nodes
        def backtrack(n):
            if n in dp:
                return dp[n]
            result = []
            for left in range(n):
                right = n - 1 - left
                leftTrees, rightTrees = backtrack(left), backtrack(right)
                for t1 in leftTrees:
                    for t2 in rightTrees:
                        result.append(TreeNode(0, t1, t2))
            dp[n] = result
            return result

        return backtrack(n)


# leetcode 572
class Solution572:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not subRoot:
            return True
        if not root:
            return False
        if self.sameTree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def sameTree(self, mainTree, subTree):
        if not mainTree and not subTree:
            return True
        if mainTree and subTree and mainTree.val == subTree.val:
            return self.sameTree(mainTree.left, subTree.left) and self.sameTree(mainTree.right, subTree.right)
        return False


# leetcode 513
class Solution513:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node.right:
                queue.append(node.right)
            if node.left:
                queue.append(node.left)
        return node.val


# leetcode 669
class Solution669:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if not root:
            return None
        if root.val > high:
            return self.trimBST(root.left, low, high)
        if root.val < low:
            return self.trimBST(root.right, low, high)

        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)

        return root


# leetcode 112
class Solution112:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        def dfs(node, currentSum):
            if not node:
                return False
            currentSum += node.val
            if not node.left and not node.right:
                return currentSum == targetSum
            return dfs(node.left, currentSum) or dfs(node.right, currentSum)

        return dfs(root, 0)


# leetcode 212
class Solution212:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        for w in words:
            root.addWord(w)
        ROWS, COLS = len(board), len(board[0])
        result, visit = set(), set()

        def dfs(row, col, node, word):
            if row < 0 or col < 0 or row == ROWS or col == COLS or (row, col) in visit or board[row][col] not in node.children:
                return
            visit.add((row, col))
            node = node.children(board[row][col])
            word += board[row][col]
            if node.endOfWord:
                result.add(word)
            dfs(row - 1, col, node, word)
            dfs(row + 1, col, node, word)
            dfs(row, col - 1, node, word)
            dfs(row, col + 1, node, word)

            visit.remove((row, col))

        for r in range(ROWS):
            for c in range(COLS):
                dfs(r, c, root, "")
        return list(result)


# leetcode 94
class Solution94:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # solution 1
        result = []

        def helper(node):
            if not node:
                return
            helper(node.left)
            result.append(node.val)
            helper(node.right)

        helper(root)
        return result

        # solution 2
        result = []
        stack = []
        current = root
        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            result.append(result.val)
            current = current.right
        return result


# leetcode 173
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        result = self.stack.pop()
        current = result.right
        while current:
            self.stack.append(current)
            current = current.left
        return result.val

    def hasNext(self) -> bool:
        return self.stack != []


# leetcode 116
class Solution116:
    def connect(self, root: Optional[Node]) -> Optional[Node]:
        current, next_node = root, root.left if root else None
        while current and next_node:
            current.left.next = current.right
            if current.next:
                current.right.next = current.next.left
            current = current.next
            if not current:
                current = next_node
                next_node = current.left
        return root


# leetcode 538
class Solution538:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        currentSum = 0

        def dfs(node):
            nonlocal currentSum
            if not node:
                return
            dfs(node.right)
            temp = node.val
            node.val += currentSum
            currentSum += temp
            dfs(node.left)

        dfs(root)
        return root


# leetcode 606
class Solution606:
    def tree2str(self, root: Optional[TreeNode]) -> str:
        result = []

        def preorder(node):
            if not node:
                return
            result.append("(")
            result.append(str(root.val))

            if not root.left and root.right:
                result.append("()")
            preorder(root.left)
            preorder(root.right)
            result.append("(")

        preorder(root)

        return "".join(result)[1:-1]  # to get rid of the opening ( and closing ) on the root
