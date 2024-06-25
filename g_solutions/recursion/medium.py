from typing import List, Optional

# leetcode 894
# we define a binary tree node
class TreeNode:
    def __init__(self, val = 0, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right


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
       return [] # full binary trees cannot have an even number of nodes
    return generateFBT(n)



solution = SolutionRecursion894()
print(solution.allPossibleFBT(7))