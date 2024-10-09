from typing import List, Optional
from collections import deque, defaultdict

# leetcode 637

class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.right = right
    self.left = left

  def __repr__(self):
    return f'{self.val}, {self.left}, {self.right}'
  
class Solution637:
  def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
    if not root:
      return []
    averages = []
    queue = deque([root])

    while queue:
      level_sum = 0
      level_count = len(queue)

      for _ in range(level_count):
        node = queue.popleft()
        level_sum += node.val
        if node.left:
          queue.append(node.left)
        if node.right:
          queue.append(node.right)

      level_average = level_sum / level_count
      averages.append(level_average)

    return averages

# leetcode 501
class Solution501:
  def findMode(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
      return []
    queue = [root]
    frequency = defaultdict(int)

    while queue:
      node = queue.pop(0)
      frequency[node.val] += 1
      if node.left:
        queue.append(node.left)
      if node.right:
        queue.append(node.right)
    max_frequency = max(frequency.values())
    modes = [key for key, value in frequency.items() if value == max_frequency]

    return modes

# leetcode 993
class Solution993:
  def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
    if not root:
      return False
    queue = [(root, 0, None)] # node, depth, parent
    x_info = (None, None) # depth, parent
    y_info = (None, None) # depth, parent

    while queue:
      node, depth, parent = queue.pop(0)
      if node.val == x:
        x_info = (depth, parent)
      elif node.val == y:
        y_info = (depth, parent)
      if node.left:
        queue.append((node.left, depth + 1, node))
      if node.right:
        queue.append((node.right, depth + 1, node))
    x_depth, x_parent = x_info
    y_depth, y_parent = y_info

    return x_depth == y_depth and x_parent != y_parent
  

# leetcode 1971
class Solution1971:
  def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
    if source == destination:
      return True
    # build adjacency list
    graph = defaultdict(list)
    for start, end in edges:
      graph[start].append(end)
      graph[end].append(start)
    # initialize BFS
    queue = [source]
    visited = set([source])
    while queue:
      node = queue.pop(0)
      if node == destination:
        return True
      for neighbor in graph[node]:
        if neighbor not in visited:
          visited.add(neighbor)
          queue.append(neighbor)

    return False