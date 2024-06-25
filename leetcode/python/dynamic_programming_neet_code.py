from typing import List

# =================== START OF LEETCODE 62. Unique Paths ================================================
class UniquePathSolution:
  def uniquePaths(self, m: int, n: int) -> int:
    row = [1] * n
    for i in range(m-1):
      newRow = [1] * n
      for j in range(n - 2, -1, -1):
        newRow[j] = newRow[j + 1] + row[j]
      row = newRow
    return row[0]

   
# =================== END OF LEETCODE 62. Unique Paths ================================================


# =================== START OF LEETCODE 1143. Longest Common Subsequence ================================================
class LongestCommonSubsequenceSolution:
  def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    dp = [[0 for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]
    for i in range(len(text1) - 1, -1, -1):
      for j in range(len(text2) - 1, -1, -1):
        if text1[i] == text2[j]:
          dp[i][j] = 1 + dp[i + 1][j + 1]
        else:
          dp[i][j] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[0][0]
# =================== END OF LEETCODE 1143. Longest Common Subsequence ================================================


# =================== START OF LEETCODE 309. Best Time to Buy and Sell Stock with Cooldown ================================================
class BestTimeToBuyAndSellStockWithCooldownSolution:
  def maxProfit(self, prices: List[int]) -> int:
    # state: Buying or Selling
    # If Buy -> i + 1
    # If Sell -> i + 2
    dp = {} # key=(i, buying) val=max_profit

    
    def dfs(i, buying):
      if i >= len(prices):
        return 0
      if (i, buying) in dp:
        return dp[(i, buying)]
      if buying:
        buy = dfs(i + 1, not buying) - prices[i]
        cooldown = dfs(i + 1, buying)
        dp[(i, buying)] = max(buy, cooldown)
      else:
        sell = dfs(i + 2, not buying) + prices[i]
        cooldown = dfs(i + 1, buying)
        dp[(i, buying)] = max(sell, cooldown)

      return dp[(i, buying)]
    
    return dfs(0, True)


# =================== END OF LEETCODE 309. Best Time to Buy and Sell Stock with Cooldown ================================================


# =================== START OF LEETCODE 518. Coin Change II ================================================
class CoinChange2Solution:
  def change(self, amount: int, coins: List[int]) -> int:
    cache = {}
    def dfs(i, a):
      if a == amount:
        return 1
      if a > amount:
        return 0
      if i == len(coins):
        return 0
      if (i, a) in cache:
        return cache[(i, a)]
      
      cache[(i, a)] = dfs(i, a + coins[i]) + dfs(i + 1, a)

      return cache[(i, a)]
    
    return dfs(0,0)
# =================== END OF LEETCODE 518. Coin Change II ================================================


# =================== START OF LEETCODE 494. Target Sum ================================================
class TargetSumSolution:
  def findTargetSumWays(self, nums: List[int], target: int) -> int:
    dp = {} # (index, total) -> # of ways
    def backtrack(index, total):
      if index == len(nums):
        return 1 if total == target else 0
      if (index, total) in dp:
        dp[(index, total)]

      dp[(index, total)] = backtrack(index + 1, total + nums[index]) + backtrack(index + 1, total - nums[index])

      return dp[(index, total)]
    
    return backtrack(0, 0)

# =================== END OF LEETCODE 494. Target Sum ================================================


# =================== START OF LEETCODE 97. Interleaving String ================================================
class InterleavingStringSolution:
  def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
    # solution 1
    if len(s1) + len(s2) != len(s3):
      return False
    
    dp = [[False] * (len(s2) + 1) for i in range(len(s1) + 1)]
    dp[len(s1)][len(s2)] = True

    for i in range(len(s1), -1, -1):
      for j in range(len(s2), -1, -1):
        if i < len(s1) and s1[i] == s3[i + j] and dp[i + 1][j]:
          dp[i][j] = True
        if j < len(s2) and s2[j] == s3[i + j] and dp[i][j + 1]:
          dp[i][j] = True

    return dp[0][0]

    # solution 2
    dp = {}
    # k = i + j
    def dfs(i, j):
      if i == len(s1) and  len(s2):
        return True
      if (i, j) in dp:
        return dp[(i, j)]
      
      if i < len(s1) and s1[i] == s3[i + j] and dfs(i + 1, j):
        return True
      if j < len(s2) and s2[j] == s3[i + j] and dfs(i, j + 1):
        return True
      dp[(i, j)] = False
      return False
    
    return dfs(0,0)
# =================== END OF LEETCODE 97. Interleaving String ================================================


# =================== START OF LEETCODE 329. Longest Increasing Path in a Matrix ================================================
class LongestIncreasingPathInAMatrix:
  def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
    rows, cols = len(matrix), len(matrix[0])
    dp = {} # (r, c) -> LIP
    
    def dfs(r, c, prevVal):
      if (r < 0 or r == rows or c < 0 or c == cols or matrix[r][c] <= prevVal):
        return 0
      if (r, c) in dp:
        return dp[(r, c)]
      
      result = 1
      result = max(result, 1 + dfs(r+1, c, matrix[r][c]))
      result = max(result, 1 + dfs(r-1, c, matrix[r][c]))
      result = max(result, 1 + dfs(r, c + 1, matrix[r][c]))
      result = max(result, 1 + dfs(r, c-1, matrix[r][c]))

      dp[(r, c)] = result
      return result
    
    for r in range(rows):
      for c in range(cols):
        dfs(r, c, -1)

    return max(dp.values())

# =================== END OF LEETCODE 329. Longest Increasing Path in a Matrix ================================================


# =================== START OF LEETCODE 115. Distinct Subsequences ================================================
class DistinctSubsequenceSolution:
  def numDistinct(self, s: str, t: str) -> int:
    cache = {}
    def dfs(i, j):
      if j == len(t):
        return 1
      if i == len(s):
        return 0
      if (i, j) in cache:
        return cache[(i, j)]
      if s[i] == t[j]:
        cache[(i, j)] = dfs(i + 1, j + 1) + dfs(i + i, j)
      else:
        cache[(i, j)] = dfs(i + i, j)
      return cache[(i, j)]
    
    return dfs(0,0)

# =================== END OF LEETCODE 115. Distinct Subsequences ================================================


# =================== START OF LEETCODE 72. Edit Distance ================================================
class EditDistanceSolution:
  def minDistance(self, word1: str, word2: str) -> int:
    cache = [[float('inf')] * (len(word2) + 1) for i in range(len(word1) + 1)]
    for j in range(len(word2) + 1):
      cache[len(word1)][j] = len(word2) - j

    for i in range(len(word1) + 1):
      cache[i][len(word2)] = len(word1) -  i

    for i in range(len(word1) - 1, -1, -1):
      for j in range(len(word2) - 1, -1, -1):
        if word1[i] == word2[j]:
          cache[i][j] = cache[i + 1][j + 1]
        else:
          cache[i][j] = 1 + min(cache[i + 1][j], cache[i][j + 1], cache[i + 1][j + 1])

    return cache[0][0]
# =================== END OF LEETCODE 72. Edit Distance ================================================


# =================== START OF LEETCODE 312. Burst Balloons ================================================
class BurstBalloonSolution:
  def maxCoins(self, nums: List[int]) -> int:
    nums = [1] + nums + [1]
    dp = {}

    def dfs(left, right):
      if left > right:
        return 0
      if (left, right) in dp:
        return dp[(left, right)]
      dp[(left, right)] = 0
      for i in range(left, right + 1):
        coins = nums[left - 1] * nums[i] * nums[right + 1]
        coins += dfs(left, i - 1) + dfs(i + 1, right)
        dp[(left, right)] = max(dp[(left, right)], coins)
      return dp[(left, right)]

    return dfs(1, len(nums) - 2)
# =================== END OF LEETCODE 312. Burst Balloons ================================================