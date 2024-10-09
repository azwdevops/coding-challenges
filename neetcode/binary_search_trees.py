from typing import Optional

class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right=right



# leetcode 230
class Solution230:
  def kthSmallest(self, root: TreeNode, k: int) -> int:
    n = 0
    stack = []
    current = root
    while current and stack:
      while current:
        stack.append(current)
        current = current.left
      current = stack.pop()
      n += 1
      if n == k:
        return current.val
      current = current.right

# leetcode 98
class Solution98:
  def isValidBST(self, root: TreeNode) -> bool:
    def valid(node, left, right):
      if not node:
        return True
      if not (node.val < right and node.val > left):
        return False
      return (valid(node.left, left, node.val) and valid(node.right, node.val, right))
    return valid(root, float('-inf'), float('inf'))
    

# leetcode 538
class Solution538:
  def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    currentSum = 0
    def dfs(node):
      if not node:
        return
      nonlocal currentSum
      dfs(node.right)

      tmp = node.val
      node.val += currentSum

      currentSum += tmp 

      dfs(node.left)

    dfs(root)

    return root
  

# leetcode 96
class Solution96:
  def numTrees(self, n: int) -> int:
    numTree = [1] * len(n + 1)
    for nodes in range(2, n + 1):
      total = 0
      for root in range(1, nodes + 1):
        left = root - 1
        right = nodes - root
        total += numTree[left] * numTree[right]
      numTree[nodes] = total
    return numTree[n]

