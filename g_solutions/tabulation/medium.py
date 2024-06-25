from typing import List, Optional


# leetcode 894 
# we first define a binary tree
class TreeNode:
  def __init__(self, val = 0, left = None, right=None):
    self.val = val
    self.left = left
    self.right = right 
# time complexity O(2^n)
# space complexity O(2^n)
class SolutionTabulation894:
  def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
    if n % 2 == 0:
      return [] # full binary trees must have an odd number of nodes
    # initialize dp table with base case
    dp = [[] for _ in range(n + 1)]
    dp[1] = [TreeNode(0)]
    # fill the dp table
    for nodes in range(3, n + 1, 2):
      for left_nodes in range(1, nodes, 2):
        right_nodes = nodes - 1 - left_nodes
        for left_tree in dp[left_nodes]:
          for right_tree in dp[right_nodes]:
            root = TreeNode(0)
            root.left = left_tree
            root.right = right_tree
            dp[nodes].append(root)
    return dp[n]
  

# solution = SolutionTabulation894()
# print(solution.allPossibleFBT(7))
# print(solution.allPossibleFBT(3))


# leetcode 1641 
# time complexity O()
# space complexity O()
class SolutionTabulation1641:
  def countVowelStrings(self, n: int) -> int:
    # initialize the dp table
    dp = [[0] * 5 for _ in range(n + 1)]
    # base case for length 1, there is exactly one way to have each vowel
    for j in range(5):
      dp[1][j] = 1
    # fill the dp table
    for i in range(2, n + 1):
      for j in range(5):
        dp[i][j] = sum(dp[i - 1][k] for k in range(j + 1))
    # the result is the sum of all dp[n][j] for j from 0 to 4
    return sum(dp[n][j] for j in range(5))
  
# solution = SolutionTabulation1641()
# print(solution.countVowelStrings(1))
# print(solution.countVowelStrings(2))
# print(solution.countVowelStrings(33))