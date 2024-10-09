from typing import List

class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

# leetcode 2824
class Solution2824:
  def countPairs(self, nums: List[int], target: int) -> int:
    nums.sort()
    count = 0
    n = len(nums)
    for i in range(n - 1):
      left, right = i + 1, n - 1
      while left <= right:
        mid = (left + right) // 2
        if nums[i] + nums[mid] < target:
          left = mid + 1
        else:
          right = mid - 1
      count += left - (i + 1)

    return count
  

# solution = Solution2824()
# print(solution.countPairs([-1,1,2,3,1], 2))

# leetcode 1351
class Solution1351:
  def countNegatives(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    count = 0
    row = m - 1
    col = 0
    
    while row >= 0 and col < n:
      if grid[row][col] < 0:
        # All elements above grid[row][col] are negative
        count += (n - col)
        row -= 1
      else:
        # Move upwards
        col += 1
    
    return count
  

# solution = Solution1351()
# print(solution.countNegatives([[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]))


# leetcode 2089
class Solution2089:
  def targetIndices(self, nums: List[int], target: int) -> List[int]:
    nums.sort()
    result = []
    for i in range(len(nums)):
      if nums[i] == target:
        result.append(i)
      elif nums[i] > target:
        break
    return result
  
# solution = Solution2089()
# print(solution.targetIndices([1,2,5,2,3], 2))

# leetcode 1337
class Solution1337:
  def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
    m = len(mat)
    n = len(mat[0])
    # create a list to store (number of soldiers, row index) for each row
    soldier_count = []
    for i in range(m):
      # count the number of soldiers (1's) in each row
      count = sum(mat[i])
      soldier_count.append((count, i))

    # sort the list by number of soldiers then by row index
    soldier_count.sort()
    # extract the indices of the first k rows from the sorted list
    result = [soldier_count[i][1] for i in range(k)]

    return result
  
# solution = Solution1337()
# print(solution.kWeakestRows([[1,1,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,0,0,0],[1,1,1,1,1]], 3))

# leetcode 1608
class Solution1608:
  def specialArray(self, nums: List[int]) -> int:
    nums.sort()
    n = len(nums)
    for i in range(n):
      # number of elements that are greater than or equal to nums[i]
      count = n - i

      # Check if nums[i] >= count and the previous element is less than count
      if nums[i] >= count and (i == 0 or nums[i - 1] < count):
        return count
      
    return -1
  
# solution = Solution1608()
# print(solution.specialArray([3,5]))


# leetcode 897
class Solution897:
  def increasingBST(self, root: TreeNode) -> TreeNode:
    dummy = TreeNode(-1)
    current = dummy

    def in_order_traversal(node):
      nonlocal current
      if not node:
        return
      # traverse the left subtree
      in_order_traversal(node.left)
      # visit the node
      current.right = node # attach node to the right of the current node
      node.left = None # ensure the left child is None
      current = current.right # move current to the next node

      # traverse the right subtree
      in_order_traversal(node.right)
    in_order_traversal(root)
    return dummy.right
  
  def increasingBSTUsingStack(self, root: TreeNode) -> TreeNode:
    dummy = TreeNode(-1)
    current = dummy
    stack = []
    node = root
    while stack or node:
      # traverse to the leftmost node
      while node:
        stack.append(node)
        node = node.left
      # process the node
      node = stack.pop()
      current.right = node # attach node to the right of the current node
      node.left = None
      current = current.right # move the current to the next node

      # move to the right subtree
      node = node.right
    return dummy.right
