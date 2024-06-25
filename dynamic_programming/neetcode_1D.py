from typing import List


# leetcode 70

class Solution70:
  def climbStairs(self, n: int) -> int:
    one, two = 1, 1
    for i in range(n - 1):
      temp = one
      one = one + two
      two = temp
    
    return one
  

# leetcode 746
class Solution746:
  def minCostClimbStairs(self, cost: List[int]) -> int:
    cost.append(0)
    for i in range(len(cost) - 3, -1, -1):
      cost[i] += min(cost[i + 1], cost[i + 2])
    
    return min(cost[0], cost[1])


# leetcode 198
class Solution198:
  def rob(self, nums: List[int]) -> int:
    rob1, rob2 = 0,0
    for house in nums:
      temp = max(house + rob1, rob2)
      rob1 = rob2
      rob2 = temp
    return rob2



# solution = Solution198()
# print(solution.rob([1,2,3,1]))
# print(solution.rob([2,7,9,3,1]))

# leetcode 213
class Solution213:
  def rob(self, nums:list[int]) -> int:
    return max(nums[0], self.helper(nums[1:]), self.helper(nums[:-1]))

  def helper(self, nums):
    rob1, rob2 = 0,0
    for n in nums:
      newRob = max(n + rob1, rob2)
      rob1 = rob2
      rob2 = newRob
    return rob2


# leetcode 5
class Solution5:
  def longestPalidrome(self, s: str) -> str:
    result = ''
    resultLength = 0
    for i in range(len(s)):
      # odd length palidromes
      left, right = i, i
      while left >= 0 and right < len(s) and s[left] == s[right]:
        if (right - left + 1) > resultLength:
          result = s[left:right + 1]
          resultLength = right - left + 1
        left -= 1
        right += 1
      
      # even length
      left, right = i, i + 1
      while left >= 0 and right < len(s) and s[left] == s[right]:
        if (right - left + 1) > resultLength:
          result = s[left:right + 1]
          resultLength = right - left + 1 
        left -= 1
        right += 1

    return result
  
# leetcode 647
class Solution647:
  def countSubstrings(self, s: str) -> int:
    count = 0
    for i in range(len(s)):
      # if s is of odd length
      left, right = i, i
      while left >= 0 and right < len(s) and s[left] == s[right]:
        count += 1
        left -= 1
        right += 1

      # if s is of even length
      left, right = i, i + 1
      while left >= 0 and right < len(s) and s[left] == s[right]:
        count += 1
        left -= 1
        right += 1

    return count
  

# leetcode 91
class Solution91:
  def numDecodings(self, s: str) -> int:
    dp = {len(s): 1}
    def dfs(i):
      if i in dp:
        return dp[i]
      if s[i] == '0':
        return 0
      result = dfs(i + 1)
      if (i + 1 < len(s) and (s[i] == '1' or s[i] == '2' and s[i + 1] in '0123456')):
        result += dfs(i + 2)
      dp[i] = result
      return result
    return dfs(0)
