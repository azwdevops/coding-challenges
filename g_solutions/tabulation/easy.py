from typing import List, Optional

# leetcode 3032
# tabulation
class SolutionTabulation3032:
  def count_unique_numbers_tabulation(self, a, b):
    def has_unique_digits(num):
      num_str = str(num)
      return len(set(num_str)) == len(num_str)
    
    count = 0
    for num in range(a, b + 1):
      if has_unique_digits(num):
        count += 1
    return count
  
# solution = SolutionTabulation3032()
# print(solution.count_unique_numbers_tabulation(10,15))

# leetcode 338 
# time complecity O(n)
# space complexity O(n)
class SolutionTabulation338:
  def countBits(self, n: int) -> List[int]:
    # Initialize the result array with zeros, length is n + 1 to include 0 to n
    ans = [0] * (n + 1)
    # Loop through each number from 1 to n
    for i in range(1, n + 1):
      # Calculate number of 1's in binary representation of i
      # ans[i >> 1] gives the number of 1's in i // 2
      # (i & 1) adds 1 if the last bit of i is 1 (i is odd), else adds 0
      ans[i] = ans[i >> 1] + (i&1)

    return ans
  
# solution = SolutionTabulation338()
# print(solution.countBits(2))
# print(solution.countBits(5))

# leetcode 118 
class SolutionTabulation118:
  def generate(self, numRows: int) -> List[List[int]]:
    # Initialize the triangle list of lists
    triangle = []
    for i in range(numRows):
      # start each row with a 1
      row = [1] * (i + 1)
      # compute the inner elements of the row if i > 1
      for j in range(1, i):
        # each element is the sum of the two elements directly above it
        row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
      triangle.append(row)

    return triangle
  
# solution = SolutionTabulation118()
# print(solution.generate(1))
# print(solution.generate(2))
# print(solution.generate(3))


# leetcode 509 
# time complecity O(n)
# space complexity O(n)
class SolutionTabulation509:
  def fib(self, n: int) -> int:
    if n <= 1:
      return n
    # initialize an array to store the fibonacci numbers upto n
    dp = [0] * (n + 1)
    # base cases
    dp[0] = 0
    dp[1] = 1
    # computer each fibonacci number from 2 to n
    for i in range(2, n + 1):
      dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]
  
# solution = SolutionTabulation509()
# print(solution.fib(10))


# leetcode 1025 
# time complecity O(n^2)
# space complexity O(n)
class SolutionTabulation1025:
  def divisorGame(self, n: int) -> bool:
    # initialize a list to store the results for each number from 0 to n
    dp = [False] * (n + 1)
    # base case if n == 1, the player loses (since there are no moves)
    dp[1] = False
    # fill the dp table for each number from 2 to n
    for i in range(2, n + 1):
      # check all possible moves
      for x in range(1, i):
        if i % x == 0:
          # if the opponent loses with (i - x), the current player wins with i
          if not dp[i - x]:
            dp[i] = True
            break

    # the result for the initial number n is stored in dp[n]
    return dp[n]
  
# solution = SolutionTabulation1025()
# print(solution.divisorGame(2))
# print(solution.divisorGame(3))


# leetcode 746 
# time complecity O(n)
# space complexity O(n), However, this can be optimized to 𝑂(1) since we only need the last two values at any point in time.
class SolutionTabulation746:
  def minCostClimbingStairs(self, cost: List[int]) -> int:
    n = len(cost)
    if n == 0:
      return 0
    if n == 1:
      return cost[0]
    # initialize dp array
    dp = [0] * (n)
    dp[0] = cost[0]
    dp[1] = cost[1]
    # fill the dp array
    for i in range(2, n):
      dp[i] = cost[i] + min(cost[i - 1], cost[i - 2])
    
    # the minimum cost to reach the top is either from the last step or the second last step
    return min(dp[n - 2], dp[n - 1])
  
  def minCostClimbingStairsOptimized(self, cost: List[int]) -> int:
    n = len(cost)
    if n == 0:
      return 0
    if n == 1:
      return cost[0]
    # initialize the first two cases
    prev2 = cost[0]
    prev1 = cost[1]
    for i in range(2, n):
      current = cost[i] + min(prev1, prev2)
      prev2 = prev1
      prev1 = current
    # the minimum cost to reach at the top is either from the last step or the second last step
    return min(prev1, prev2)
  
# solution = SolutionTabulation746()
# print(solution.minCostClimbingStairs([10,15,20]))
# print(solution.minCostClimbingStairsOptimized([10,15,20]))


# leetcode 119 
# time complexity O(n^2)
# space complexity O(n)
class SolutionTabulation119:
  def getRow(self, rowIndex: int) -> List[int]:
    # initialize the first row of pascal's triangle
    row = [1]
    # generate each subsequent row up to rowIndex
    for i in range(1, rowIndex + 1):
      # compute each element in the current row
      # we build the row backwards to avoid overwriting values prematurely
      for j in range(i, 0, -1):
        if j == i:
          row.append(1) # last element in each row is 1
        else:
          row[j] = row[j] + row[j - 1] # compute the current element

    return row


# solution = SolutionTabulation119()
# print(solution.getRow(3))


# leetcode 1137 
# time complexity O(n)
# space complexity O(n)
class SolutionTabulation1137:
  def tribonacci(self, n: int) -> int:
    if n == 0:
      return 0
    if n == 1 or n == 2:
      return 1
    # initialize dp array
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    dp[2] = 1
    # fill the dp array
    for i in range(3, n + 1):
      dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
    return dp[n]
  

# solution = SolutionTabulation1137()
# print(solution.tribonacci(4))
# print(solution.tribonacci(25))

# leetcode 2900 
# time complexity O(n^2)
# space complexity O(n)
class SolutionTabulation2900:
  def getLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
    n = len(words)
    if n == 0:
      return []
    dp = [1] * n # dp[i] will store the length of the longest alternating subsequence at index i
    prev = [-1] * n # prev[i] will store the index of the previous element in the subsequence

    # iterate over all pairs (i, j) where i < j
    for i in range(n):
      for j in range(i + 1, n):
        if groups[i] != groups[j]: # check if groups differ
          if dp[j] < dp[i] + 1:
            dp[j] = dp[i] + 1
            prev[j] = i

    # find the index of the element with maximum dp value
    max_length = max(dp)
    max_index = dp.index(max_length)

    # reconstruct the longest alternating subsequence
    result = []
    while max_index != -1:
      result.append(words[max_index])
      max_index = prev[max_index]

    # reverse the result to get the correct order
    result.reverse()
    return result
  
# solution = SolutionTabulation2900()
# print(solution.getLongestSubsequence(["e","a","b"], [0,0,1]))



# leetcode 121 
# time complexity O(n)
# space complexity O(1)
class SolutionTabulation121:
  def maxProfit(self, prices: List[int]) -> int:
    if not prices:
      return 0
    # initialize variables
    min_price = float('inf')
    max_profit = 0
    # iterate through the prices
    for price in prices:
      if price < min_price:
        min_price = price
      elif price - min_price > max_profit:
        max_profit = price - min_price

    return max_profit
  
# solution = SolutionTabulation121()
# print(solution.maxProfit([7,1,5,3,6,4]))
# print(solution.maxProfit([7,6,4,3,1]))



# leetcode 70 
# time complexity O(n)
# space complexity O(n)
class SolutionTabulation70:
  def climbStairs(self, n: int) -> int:
    if n == 0:
      return 1 # base case: 1 way to stay on the ground (do nothing)
    dp = [0] * (n + 1)
    dp[0] = 1 # 1 way to stay on the ground (do nothing)
    if n >= 1:
      dp[1] = 1 # 1 way to reach the 1st step
    if n >= 2:
      dp[2] = 2 # 2 ways to reach the 2nd step (1 + 1, or 2)
    for i in range(3, n + 1):
      dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
  
# solution = SolutionTabulation70()
# print(solution.climbStairs(2))
# print(solution.climbStairs(3))



# leetcode 392 
# time complexity O(m x n)
# space complexity O(m x n)
class SolutionTabulation392:
  def isSubsequence(self, s: str, t: str) -> bool:
    m, n = len(s), len(t)
    # dp[i][j] will be True if the first i characters of s is a subsequence of the first j characters of t
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    # an empty string is a subsequence of any string
    for j in range(n + 1):
      dp[0][j] = True
    # fill the dp table
    for i in range(1, m + 1):
      for j in range(1, n + 1):
        if s[i - 1] == t[j - 1]:
          dp[i][j] = dp[i - 1][j - 1]
        else:
          dp[i][j] = dp[i][j - 1]
    return dp[m][n]
  

# solution = SolutionTabulation392()
# print(solution.isSubsequence('abc','ahbgdc'))
# print(solution.isSubsequence('axc','ahbgdc'))


# leetcode 1668 
# time complexity O(n x k)
# space complexity O(1)
class SolutionTabulation1668:
  def maxRepeating(self, sequence: str, word: str) -> int:
    max_k = 0
    k = 1
    # check for increasing values of k
    while word * k in sequence:
      max_k = k
      k += 1
    return max_k
  
# solution = SolutionTabulation1668()
# print(solution.maxRepeating('ababc', 'ab'))
# print(solution.maxRepeating('ababc', 'ba'))
# print(solution.maxRepeating('ababc', 'ac'))


