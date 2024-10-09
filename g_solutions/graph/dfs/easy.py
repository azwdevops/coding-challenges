from typing import Optional, List
from collections import deque

# leetcode 617

class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.right = right
    self.left = left

  def __repr__(self):
    return f'{self.val}, {self.left}, {self.right}'

class Solution617:
  def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root1 and not root2:
      return None
    elif not root1:
      return root2
    elif not root2:
      return root1
    merged = TreeNode(root1.val + root2.val)
    merged.left = self.mergeTrees(root1.left, root1.left)
    merged.right = self.mergeTrees(root1.right, root1.right)

    return merged
  

# leetcode 897
class Solution897:
  def increasingBST(self, root: TreeNode) -> TreeNode:
    
    def inOrderTraversal(node, prev):
      if not node:
        return prev
      prev = inOrderTraversal(node.left, prev)
      node.left = None
      prev.right = node
      prev = node
      return inOrderTraversal(node.right, prev)
    
    dummy = TreeNode(0)
    prev = dummy
    inOrderTraversal(root, prev)
    
    return dummy.right
    
# node1 = TreeNode(1)
# node2 = TreeNode(2)
# node3 = TreeNode(3)
# node4 = TreeNode(4)
# node5 = TreeNode(5)
# node6 = TreeNode(6)
# node7 = TreeNode(7)
# node8 = TreeNode(8)
# node9 = TreeNode(9)

# node5.right = node6
# node5.left = node3
# node3.left = node2
# node3.right = node4
# node2.left = node1
# node6.right = node8
# node8.left = node7
# node8.right = node9

# solution = Solution897()
# print(solution.increasingBST(node5))

# leetcode 590
class Node:
  def __init__(self, val=None, children=None):
    self.val = val
    self.children = children

class Solution590:
  # using recursion
  def postorder(self, root: 'Node') -> List[int]:
    def dfs(node, result):
      if not node:
        return
      for child in node.children:
        dfs(child, result)
      result.append(node.val)
    result = []
    dfs(root, result)

    return result

  # using iterative approach
  def postorderIterative(self, root: Node) -> List[int]:
    if not root:
      return []
    stack = [root]
    result = []
    while stack:
      node = stack.pop()
      result.insert(0, node.val)
      for child in node.children:
        stack.append(child)
    return result


# leetcode 1022
class Solution1022:
  def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
    def dfs(node, current_number):
      if node is None:
        return 0
      current_number = current_number * 2 + node.val
      if node.left is None and node.right is None:
        return current_number
      return dfs(node.left, current_number) + dfs(node.right, current_number)
    
    return dfs(root, 0)
  
# leetcode 463
class Solution463:
  def islandPerimeter(self, grid: List[List[int]]) -> int:
    perimeter = 0
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
      for j in range(cols):
        if grid[i][j] == 1:
          # check top
          if i == 0 or grid[i - 1][j] == 0:
            perimeter += 1
          # check bottom
          if i == rows - 1 or grid[i + 1][j] == 0:
            perimeter += 1
          # check left
          if j == 0 or grid[i][j - 1] == 0:
            perimeter += 1
          # check right
          if j == cols - 1 or grid[i][j + 1] == 0:
            perimeter += 1
    return perimeter
  
# solution = Solution463()
# print(solution.islandPerimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]))

# leetcode 733
class Solution733:
  def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    original_color = image[sr][sc]
    if original_color == color:
      return image
    rows = len(image)
    cols = len(image[0])
    def dfs(x, y):
      if x < 0 or y < 0 or x >= rows or y >= cols or image[x][y] != original_color:
        return
      image[x][y] = color
      dfs(x + 1, y)
      dfs(x - 1, y)
      dfs(x, y + 1)
      dfs(x, y - 1)

    dfs(sr, sc)

    return image
  
# solution = Solution733()
# print(solution.floodFill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2))

# leetcode 257
class Solution257:
  def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
    if not root:
      return []
    result = []
    def dfs(node, path):
      if not node:
        return
      if not node.left and not node.right:
        result.append(path + str(node.val))
      if node.left:
        dfs(node.left, path + str(node.val) + '->')
      if node.right:
        dfs(node.right, path + str(node.val) + '->')
    dfs(root, '')

    return result
  

# leetcode 563
class Solution563:
  def findTilt(self, root: Optional[TreeNode]) -> int:
    total_tilt = 0
    def dfs(node):
      nonlocal total_tilt
      if not node:
        return 0
      left_sum = dfs(node.left)
      right_sum = dfs(node.right)
      node_tilt = abs(left_sum - right_sum)
      total_tilt += node_tilt
      return node.val + left_sum + right_sum
    
    dfs(root)

    return total_tilt

# leetcode 543
class Solution543:
  def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
    max_diameter = 0
    def dfs(node):
      nonlocal max_diameter
      if not node:
        return 0
      left_depth = dfs(node.left)
      right_depth = dfs(node.right)
      max_diameter = max(max_diameter, left_depth + right_depth)
      return 1 + max(left_depth, right_depth)
    dfs(root)

    return max_diameter

# leetcode 783
class Solution783:
  def minDiffInBST(self, root: Optional[TreeNode]) -> int:
    prev = None
    min_diff = float('inf')
    def inOrderTraversal(node):
      nonlocal prev, min_diff
      if not node:
        return
      inOrderTraversal(node.left)
      if prev is not None:
        min_diff = min(min_diff, node.val - prev)
      prev = node.val
      inOrderTraversal(node.right)
    inOrderTraversal(root)

    return min_diff
  
# leetcode 530
class Solution530:
  def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
    prev = None
    min_diff = float('inf')
    def inOrderTraversal(node):
      nonlocal prev, min_diff
      if not node:
        return
      inOrderTraversal(node.left)
      if prev is not None:
        min_diff = min(min_diff, abs(node.val - prev))
      prev = node.val
      inOrderTraversal(node.right)

    inOrderTraversal(root)

    return min_diff

# leetcode 101
class Solution101:
  def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    if not root:
      return True
    queue = deque([(root.left, root.right)])
    while queue:
      node1, node2 = queue.popleft()
      if not node1 and not node2:
        continue
      if not node1 or not node2 or node1.val != node2.val:
        return False
      queue.append((node1.left, node2.right))
      queue.append((node1.right, node2.left))
    return True
  

# leetcode 110
class Solution110:
  def isBalanced(self, root: Optional[TreeNode]) -> bool:
    if not root:
      return True
    stack = [(root, 0, False)] # node, height, processed
    height = {} # to store height of nodes
    while stack:
      node, current_height, processed = stack.pop()
      if not node:
        continue
      if not processed:
        # push node back onto the stack as processed
        stack.append((node, current_height, True))
        # push the children onto the stack
        stack.append((node.left, current_height + 1, False))
        stack.append((node.right, current_height + 1, False))
      else:
        # calculate the height of the node
        left_height = height.get(node.left, -1)
        right_height = height.get(node.right, -1)
        if abs(left_height - right_height) > 1:
          return False
        height[node] = 1 + max(left_height, right_height)

    return True


# leetcode 112
class Solution112:
  def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
    if not root:
      return False
    stack = [(root, root.val)]
    while stack:
      node, current_sum = stack.pop()
      # check if it's a leaf node and the sum equals targetSum
      if not node.left and not node.right and current_sum == targetSum:
        return True
      # add the right child to the stack
      if node.right:
        stack.append((node.right, current_sum + node.right.val))
      # add the left child to the stack
      if node.left:
        stack.append((node.left, current_sum + node.left.val))

    return False