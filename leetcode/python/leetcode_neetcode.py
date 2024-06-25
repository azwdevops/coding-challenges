from typing import List

# =================== START OF LEETCODE 494. Target Sum ========================================
class TargetSumSolution:
  def findTargetSumWays(self, nums: List[int], target: int) -> int:
    dp = {} # (index, total) -> number of ways

    def backtrack(index, total):
      if index == len(nums):
        return 1 if total == target else 0
      if (index, total) in dp:
        dp[(index, total)]
      dp[(index, total)] = (backtrack(index + 1, total + nums[index])) + (backtrack(index + 1, total - nums[index]))

      return dp[(index, total)]
    
    return backtrack(0, 0)
  
# =================== END OF LEETCODE 494. Target Sum ========================================


# =================== END OF LEETCODE 70. Climbing Stairs ========================================
class ClimbingStairSolution:
  def climbStairs(self, n: int) -> int:
    one, two = 1,1
    for i in range(n-1):
      temp = one
      one = one + two
      two = temp
    return one

# =================== END OF LEETCODE 70. Climbing Stairs ========================================

# =================== START OF LEETCODE 983. Minimum Cost For Tickets ========================================
class MinimumCostForTicketSolution:
  def minCostTickets(self, days: List[int], costs: List[int]) -> int:
    dp = {}
    def dfs(index):
      if index == len(days):
        return 0
      if index in dp:
        return dp[index]
      dp[index] = float('inf')
      for day, cost in zip([1,7,30], costs):
        j = index
        while j < len(days) and days[j] < days[index] + day:
          j += 1
        dp[index] = min(dp[index], cost, dfs(j))

      return dp[index]
    
    return dfs(0)

# =================== END OF LEETCODE 983. Minimum Cost For Tickets ========================================


# =================== START OF LEETCODE 518. Coin Change II ========================================
class CoinChange2Solution:
  def change(self, amount: int, coins: List[int]) -> int:
    dp = {}
    def dfs(index, a):
      if a == amount:
        return 1
      if a > amount:
        return 0
      if index == len(coins):
        return 0
      if (index, a) in dp:
        return dp[(index, a)]
      dp[(index, a)] = dfs(index, a + coins[index]) + dfs(index + 1, a)
      return dp[(index, a)]
    return dfs(0,0)

# =================== END OF LEETCODE 518. Coin Change II ========================================


# =================== START OF LEETCODE 139. Word Break ========================================

class WordBreakSolution:
  def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    dp = [False]  * (len(s) + 1)
    dp[len(s)] = True
    for i in range(len(s) - 1, -1, -1):
      for w in wordDict:
        if (i + w) <= len(s) and s[i:i+len(w)] == w:
          dp[i] = dp[i+len(w)]
        if dp[i]:
          break
    return dp[0]

# =================== END OF LEETCODE 139. Word Break ========================================


# =================== START OF LEETCODE 5. Longest Palindromic Substring ========================================
class LongestPalindromicSubstring:
  def longestPalidrome(self, s: str) -> str:
    result = ''
    resultLength = 0
    for i in range(len(s)):
      # odd length palidromes
      left, right = i, i
      while left >= 0 and right < len(s) and s[left] == s[right]:
        if (right - left + 1) > resultLength:
          result = s[left:right+1]
          resultLength = right - left + 1
        left -= 1
        right += 1

      # even length palidromes
      left, right = i, i + 1
      while left >= 0 and right < len(s) and s[left] == s[right]:
        if (right - left + 1) > resultLength:
          result = s[left:right+1]
          resultLength = right - left + 1
        left -= 1
        right += 1

    return result
# =================== END OF LEETCODE 5. Longest Palindromic Substring ========================================


# =================== START OF LEETCODE 91. Decode Ways ========================================
class DecodeWaySolution:
  def numDecodings(self, s: str) -> int:
    dp = {len(s): 1}
    
    def dfs(i):
      if i in dp:
        return dp[i]
      if s[i] == '0':
        return 0
      result = dfs(i+1)
      if (i + 1 < len(s) and s[i] == '1') or (s[i] == '2' and s[i+1] in '0123456'):
        result += dfs(i + 2)
      dp[i] = result

      return result
    return dfs(0)


# =================== END OF LEETCODE 91. Decode Ways ========================================

# =================== START OF LEETCODE 198. House Robber ========================================
class HouseRobberSolution:
  def rob(self, nums: List[int]) -> int:
    rob1, rob2 = 0,0
    # [rob1, rob2, n, n+1 .....]
    for n in nums:
      temp = max(n + rob1, rob2)
      rob1 = rob2
      rob2 = temp
    return rob2
# =================== END OF LEETCODE 198. House Robber ========================================


# =================== START OF LEETCODE 97. Interleaving String ========================================
class InterleavingStringSolution:
  def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
    if len(s1) + len(s2) != len(s3):
      return False
    dp = [ [False] * len(s2) + 1 for i in range(len(s1) + 1)]
    dp[len(s1)][len(s2)] = True
    for i in range(len(s1), -1, -1):
      for j in range(len(s2), -1, -1):
        if i < len(s1) and s1[i] == s3[i+j] and dp[i+1][j]:
          dp[i][j] = True
        if j < len(s2) and s2[j] == s3[i+j] and dp[i][j+1]:
          dp[i][j] = True
    return dp[0][0]
    
    # alternative solution below
    dp = {}
    # k = i + j
    def dfs(i, j):
      if i == len(s1) and j == len(s2):
        return True
      if (i, j) in dp:
        return dp[(i, j)]
      if i < len(s1) and s1[i] == s3[i+j] and dfs(i+1, j):
        return True
      if j < len(s2) and s2[j] == s3[i+j] and dfs(i, j+1):
        return True
      dp[(i, j)] = False
      return False
    return dfs(0,0)


# =================== END OF LEETCODE 97. Interleaving String ========================================

# =================== START OF LEETCODE 213. House Robber II ========================================
class HouseRobber2Solution:
  def rob(self, nums: List[int]) -> int:

    return max(nums[0], self.helper(nums[1:]), self.helper(nums[:-1]))

  def helper(self, nums):
    rob1, rob2 = 0, 0
    for n in nums:
      newRob = max(rob1 + n, rob2)
      rob1 = rob2
      rob2 = newRob
    return rob2

# =================== END OF LEETCODE 213. House Robber II ========================================


# =================== START OF LEETCODE 256. Paint House ========================================
class PaintHouseSolution:
  def minCost(self, costs: List[List[int]]) -> int:
    # costs[i][j] i is house, j is color
    dp = [0,0,0]
    for i in range(len(costs)):
      dp0 = costs[i][0] + min(dp[1], dp[2])
      dp1 = costs[i][1] + min(dp[0], dp[2])
      dp2 = costs[i][2] + min(dp[0], dp[1])
      dp = [dp0, dp1, dp2]
    return min(dp)

# =================== END OF LEETCODE 256. Paint House ==========================================


# =================== START OF LEETCODE 300. Longest Increasing Subsequence ========================================
class LongestIncreasingSubsequenceSolution:
  def lengthOfLIS(self, nums: List[int]) -> int:
    LIS = [1] * len(nums)
    for i in range(len(nums), -1, -1, -1):
      for j in range(i + 1, len(nums)):
        if nums[i] < nums[j]:
          LIS[i] = max(LIS[i], 1 + LIS[j])
    return max(LIS)

# =================== END OF LEETCODE 300. Longest Increasing Subsequence ==========================================


# =================== START OF LEETCODE 1143. Longest Common Subsequence ==========================================
class LongestCommonSubsequenceSolution:
  def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    dp = [[0 for j in range(text2) + 1] for i in range(text1) + 1]
    for i in range(len(text1) - 1, -1, -1):
      for j in range(len(text2) - 1, -1, -1):
        if text1[i] == text2[j]:
          dp[i][j] = 1 + dp[i + 1][j + 1]
        else:
          dp[i][j] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[0][0]
  
# =================== END OF LEETCODE 1143. Longest Common Subsequence ============================================


# =================== START OF LEETCODE 152. Maximum Product Subarray ============================================
class MaximumProductSubarraySolution:
  def maxProduct(self, nums: List[int]) -> int:
    result = max(nums)
    currentMin, currentMax = 1,1
    for n in nums:
      if n == 0:
        currentMin,currentMax = 1,1
        continue
      temp = currentMax * n
      currentMax = max(n * currentMax, n * currentMin, n)
      currentMin = min(temp, n * currentMin, n)

      result = max(result, currentMax, currentMin)

    return result

# =================== END OF LEETCODE 152. Maximum Product Subarray ==============================================


# =================== START OF LEETCODE 322. Coin Change ==============================================
class CoinChangeSolution:
  def coinChange(self, coins: List[int], amount: int) -> int:
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
      for coin in coins:
        if a - coin >= 0:
          dp[a] = min(dp[a], 1 + dp[a - coin])
      
    return dp[amount] if dp[amount] != amount + 1 else -1

# =================== END OF LEETCODE 322. Coin Change ================================================


# =================== START OF LEETCODE 221. Maximal Square ================================================
class MaximalSquareSolution:
  def maximalSquare(self, matrix: List[List[str]]) -> int:
    # option 1 - dynamic programming bottom up
    # option 2 - recursive : top down
    ROWS, COLS = len(matrix), len(matrix[0])
    cache = {} # map each (r, c) -> maxLength of square

    def helper(r, c):
      if r >= ROWS or c >= COLS:
        return 0
      if (r, c) not in cache:
        down = helper(r + 1, c)
        right = helper(r, c + 1)
        diagonal = helper(r + 1, c + 1)

        cache[(r, c)] = 0
        if matrix[r][c] == '1':
          cache[(r, c)] = 1 + min(down, right, diagonal)
        
      return cache[(r, c)]
    helper(0,0)

    return max(cache.values()) ** 2

# =================== END OF LEETCODE 221. Maximal Square ================================================


# =================== START OF LEETCODE 120. Triangle ================================================
class TriangleSolution:
  def minimumTotal(self, triangle: List[List[int]]) -> int:
    dp = [0] * (len(triangle) + 1)
    for row in triangle[::-1]:
      for i, n in enumerate(row):
        dp[i] = n + min(dp[i], dp[i+1])
      
    return dp[0]

# =================== END OF LEETCODE 120. Triangle ================================================


# =================== START OF LEETCODE 96. Unique Binary Search Trees ================================================
class UniqueBinarySearchTreeSolution:
  def numTrees(self, n: int) -> int:
    numTree = [] * (n + 1)
    for nodes in range(2, n + 1):
      total = 0
      for root in range(1, nodes + 1):
        left = root - 1
        right = nodes - root
        total += numTree[left] * numTree[right]
      numTree[nodes] = total

    return numTree[n]

# =================== END OF LEETCODE 96. Unique Binary Search Trees ================================================


# =================== START OF LEETCODE 10. Regular Expression Matching ================================================
class RegularExpressionMatchingSolution:
  def isMatch(self, s:str, p:str) -> bool:
    cache = {}
    def dfs(i, j):
      if (i, j) in cache:
        return cache[(i,j)]
      if i >= len(s) and j >= len(p):
        return True
      if j >= len(p):
        return False
      new_match = i < len(s) and (s[i] == p[j] or p[j] == '.')
      if (j+1) < len(p) and p[j + 1] == '*':
        # dont use *      use * here
        cache[(i, j)] = dfs(i, j + 2) or (new_match and dfs(i + 1, j))
        return cache[(i, j)]
      if new_match:
        cache[(i, j)] = dfs(i+1, j+1)
        return cache[(i, j)]
      cache[(i, j)] = False
      return cache[(i, j)]
    
    return dfs(0,0)
# =================== END OF LEETCODE 10. Regular Expression Matching ================================================


# =================== START OF LEETCODE 1. Two Sum ================================================
class TwoSumSolution:
  def twoSum(self, nums: List[int], target: int) -> List[int]:
    prevMap = {} # val: index
    for index, value in enumerate(nums):
      diff = target - value
      if diff in prevMap:
        return [prevMap[diff], index]
      prevMap[value] = index

    
# =================== END OF LEETCODE 1. Two Sum ================================================


# =================== START OF LEETCODE 121. Best Time to Buy and Sell Stock ================================================
class BestTimeToBuyAndSellStockSolution:
  def maxProfit(self, prices: List[int]) -> int:
    left, right = 0, 1 # left = buy, right = sell
    maxP = 0

    while right < len(prices):
      if prices[left] < prices[right]:
        profit = prices[right] - prices[left]
        maxP = max(maxP, profit)
      else:
        left = right
      right += 1

    return maxP

# =================== END OF LEETCODE 121. Best Time to Buy and Sell Stock ================================================


# =================== START OF LEETCODE 217. Contains Duplicate ================================================
class ContainsDuplicateSolution:
  def containsDuplicate(self, nums: List[int]) -> bool:
    hashset = set()
    for n in nums:
      if n in hashset:
        return True
      hashset.add(n)

    return False
# =================== END OF LEETCODE 217. Contains Duplicate ================================================


# =================== START OF LEETCODE 238. Product of Array Except Self ================================================
class ProductofArrayExceptSelfSolution:
  def productExceptSelf(self, nums: List[int]) -> List[int]:
    result = [1] * len(nums)
    prefix = 1
    for i in range(len(nums)):
      result[i] = prefix
      prefix *= nums[i]
    postfix = 1
    for i in range(len(nums)-1, -1, -1):
      result[i] *= postfix
      postfix *= nums[i]

    return result 

# =================== END OF LEETCODE 238. Product of Array Except Self ================================================

# =================== START OF LEETCODE 53. Maximum Subarray ================================================
class MaximumSubarraySolution:
  def maxSubArray(self, nums: List[int]) -> int:
    max_sub = nums[0]
    current_sum = 0
    for n in nums:
      if current_sum < 0:
        current_sum = 0
      current_sum += n
      max_sub = max(max_sub, current_sum)
    return max_sub

# =================== END OF LEETCODE 53. Maximum Subarray ================================================


# =================== START OF LEETCODE 152. Maximum Product Subarray ================================================
class MaximumProductSubarraySolution:
  def maxProduct(self, nums: List[int]) -> int:
    result = max(nums)
    current_min, current_max = 1, 1
    for n in nums:
      if n == 0:
        current_min, current_max = 1, 1
        continue
      temp = current_max * n
      current_max = max(n * current_max, n * current_min, n)
      current_min = min(temp, n * current_min, n)
      result = max(result, current_max, current_min)

    return result

# =================== END OF LEETCODE 152. Maximum Product Subarray ================================================


# =================== START OF LEETCODE 153. Find Minimum in Rotated Sorted Array ================================================
class MinimumRotatedSortedArraySolution:
  def findMin(self, nums: List[int]) -> int:
    result = nums[0]
    left, right = 0, len(nums) - 1
    while left <= right:
      if nums[left] < nums[right]:
        result = min(result, nums[left])
        break
      mid = (left + right) // 2
      result = min(result, nums[mid])
      if nums[mid] >= nums[left]:
        left = mid + 1
      else:
        right = mid - 1
    return result
# =================== END OF LEETCODE 153. Find Minimum in Rotated Sorted Array ================================================


# =================== START OF LEETCODE 33. Search in Rotated Sorted Array ================================================
class SearchInRotatedSortedArraySolution:
  def search(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left < right:
      mid = (left + right) // 2
      if target == nums[mid]:
        return mid
      
      # left sorted portion
      if nums[left] <= nums[mid]:
        if target > nums[mid] or target < nums[left]:
          left = mid + 1
        else:
          right = mid - 1
      # right sorted portion
      else:
        if target < nums[mid] or target > nums[right]:
          right = mid - 1
        else:
          left = mid + 1
    return -1
        
# =================== END OF LEETCODE 33. Search in Rotated Sorted Array ================================================

# =================== END OF LEETCODE 15. 3Sum ================================================
class ThreeSumSolution:
  def threeSum(self, nums: List[int]) -> List[List[int]]:
    result = []
    nums.sort()
    for index, value in enumerate(nums):
      if index > 0 and value == nums[i - 1]:
        continue
      left, right = index + 1, len(nums) - 1
      while left < right:
        current_sum = value + nums[left] + nums[right]
        if current_sum > 0:
          right -= 1
        elif current_sum < 0:
          left += 1
        else:
          result.append([value, nums[left], nums[right]])
          left += 1
          while nums[left] == nums[left - 1] and left < right:
            left += 1

    return result

# =================== END OF LEETCODE 15. 3Sum ================================================

# =================== START OF LEETCODE 11. Container With Most Water ================================================
class ContainerWithMostWaterSolution:
  def maxArea(self, height: List[int]) -> int:
    result = 0
    left, right = 0, len(height) - 1

    while left < right:
      area = (right - left) * min(height[left], height[right])
      result = max(result, area)

      if height[left] < height[right]:
        left += 1
      else:
        right -= 1

    return result

# =================== END OF LEETCODE 11. Container With Most Water ================================================


# =================== START OF LEETCODE 191. Number of 1 Bits ================================================
class Numberof1BitsSolution:
  def hammingWeight(self, n: int) -> int:
    result = 0
    while n > 0:
      result += n % 2
      n = n >> 1
    return result
  
    # solution 2
    result = 0
    while n > 0:
      n &= (n-1)
      result += 1

    return result
# =================== END OF LEETCODE 191. Number of 1 Bits ================================================


# =================== START OF LEETCODE 338. Counting Bits ================================================
class CountingBitsSolution:
  def countBits(self, n: int) -> List[int]:
    dp = [0] * (n + 1)
    offset = 1
    for i in range(1, n+1):
      if offset * 2 == i:
        offset = i
      dp[i] = 1 + dp[i - offset]
    return dp

# =================== END OF LEETCODE 338. Counting Bits ================================================


# =================== START OF LEETCODE 268. Missing Number ================================================
class MissingNumberSolution:
  def missingNumber(self, nums: List[int]) -> int:
    result = len(nums)
    for i in range(len(nums)):
      result += (i - nums[i])
    return result
# =================== END OF LEETCODE 268. Missing Number ================================================


# =================== START OF LEETCODE 190. Reverse Bits ================================================
class ReverseBitsSolution:
  def reverseBis(self, n: int) -> int:
    result = 0
    for i in range(32):
      bit = (n >> i) & 1
      result = result | (bit << (31 - i))

    return result
# =================== END OF LEETCODE 190. Reverse Bits ================================================


# =================== START OF LEETCODE 70. Climbing Stairs ================================================
class ClimbingStairsSolution:
  def climbStairs(self, n: int) -> int:
    one, two = 1, 1
    for i in range(n - 1):
      temp = one
      one = one + two
      two = temp

    return one
# =================== END OF LEETCODE 70. Climbing Stairs ================================================

# =================== START OF LEETCODE 322. Coin Change ================================================
class CoinChangeSolution:
  def coinChange(self, coins: List[int], amount: int) -> int:
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
      for coin in coins:
        if a - coin >= 0:
          dp[a] = min(dp[a], 1 + dp[a - coin])
    
    return dp[amount] if dp[amount] != amount + 1 else -1
# =================== END OF LEETCODE 322. Coin Change ================================================


# =================== START OF LEETCODE 300. Longest Increasing Subsequence ================================================
class LongestIncreasingSubsequenceSolution:
  def lengthOfLIS(self, nums: List[int]) -> int:
    LIS = [1] * len(nums)
    for i in range(len(nums) -1, -1, -1):
      for j in range(i + 1, len(nums)):
        if nums[i] < nums[j]:
          LIS[i] = max(LIS[i], 1 + LIS[j])
    return max(LIS)

# =================== END OF LEETCODE 300. Longest Increasing Subsequence ================================================

# =================== START OF LEETCODE 1143. Longest Common Subsequence ================================================
class LongestCommonSubsequenceSolution:
  def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    dp = [[0 for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]
    for i in range(len(text1) - 1, -1, -1):
      for j in range(len(text2) - 1, -1, -1):
        if text1[i] == text1[j]:
          dp[i][j] = 1 + dp[i + 1][j + 1]
        else:
          dp[i][j] = max(dp[i][j + 1], dp[i + 1], [j])

    return dp[0][0]
# =================== END OF LEETCODE 1143. Longest Common Subsequence ================================================


# =================== START OF LEETCODE 139. Word Break ================================================
class WordBreakSolution:
  def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    dp = [False] * (len(s) + 1)
    dp[len(s)] = True
    for i in range(len(s) - 1, -1, -1):
      for w in wordDict:
        if (i + len(w)) <= len(s) and s[i:i+len(w)] == w:
          dp[i] = dp[i + len(w)]
        if dp[i]:
          break
    return dp[0]
# =================== END OF LEETCODE 139. Word Break ================================================


# =================== START OF LEETCODE 39. Combination Sum ================================================
class CombinationSumSolution:
  def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    result = []
    def dfs(i, current, total):
      if total == target:
        result.append(current.copy())
        return 
      
      if i >= len(candidates) or total > target:
        return
      
      current.append(candidates[i])
      dfs(i, current, total + candidates[i])
      current.pop()
      dfs(i + 1, current, total)

    dfs(0, [], 0)

    return result
# =================== END OF LEETCODE 39. Combination Sum ================================================


# =================== START OF LEETCODE 198. House Robber ================================================
class HouseRobberSolution:
  def rob(self, nums: List[int]) -> int:
    rob1, rob2 = 0, 0

    for n in nums:
      temp = max(n + rob1, rob2)
      rob1 = rob2
      rob2 = temp

    return rob2

# =================== END OF LEETCODE 198. House Robber ================================================


# =================== START OF LEETCODE 213. House Robber II ================================================
class HouseRobber2Solution:
  def rob(self, nums: List[int]) -> int:
    return max(nums[0], self.helper(nums[1:]), self.helper(nums[:-1]))


  def helper(self, nums):
    rob1, rob2 = 0, 0
    for n in nums:
      newRob = max(rob1 + n, rob2)
      rob1 = rob2
      rob2 = newRob

    return rob2
  
# =================== END OF LEETCODE 213. House Robber II ================================================


# =================== START OF LEETCODE 91. Decode Ways ================================================
class DecodeWaySolution:
  def numDecodings(self, s: str) -> int:
    dp = {len(s): 1}

    def dfs(i):
      if i in dp:
        return dp[i]
      if s[i] == '0':
        return 0
      
      result = dfs(i + 1)
      if (i + 1 < len(s) and (s[i] == '1' or (s[i] == '2' and s[i+1] in '0123456'))):
        result += dfs[i + 2]

      dp[i] = result

      return result
# =================== END OF LEETCODE 91. Decode Ways ================================================


# =================== START OF LEETCODE 62. Unique Paths ================================================
class UniquePathSolution:
  def uniquePaths(self, m: int, n: int) -> int:
    row = [1] * n
    for i in range(m - 1):
      newRow = [1] * n
      for j in range(n - 2, -1, -1):
        newRow[j] = newRow[j + 1] + row[j]
      row = newRow

    return row[0]
# =================== END OF LEETCODE 62. Unique Paths ================================================