
# ============================================== START OF QUESTION 15 3SUM ==================================================

# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

# Notice that the solution set must not contain duplicate triplets.

class Solution(object):
  def threeSum(self, nums):
    triplets_list = []
    nums.sort()
    length = len(nums)
    for i in range(length - 2):
      if i > 0 and nums[i] == nums[i-1]:
        continue
      left_index = i + 1
      right_index = length - 1
      while left_index < right_index:
        total = nums[i] + nums[left_index] + nums[right_index]
        if total < 0:
          left_index += 1
        elif total > 0:
          right_index -= 1
        else:
          triplets_list.append([nums[i], nums[left_index], nums[right_index]])
          while left_index < right_index and nums[left_index] == nums[left_index + 1]:
            left_index += 1
          while left_index < right_index and nums[right_index] == nums[right_index - 1]:
            right_index -= 1
          left_index += 1
          right_index -= 1
    return triplets_list
  

# new_solution = Solution()
# print(new_solution.threeSum([-1,0,1,2,-1,-4]))
# print(new_solution.threeSum([0,1,1]))
# print(new_solution.threeSum([0,0,0]))
# print(new_solution.threeSum([1,-1,-1,0]))
        

# ============================================== END OF QUESTION 15 3SUM ==================================================


# ============================================== START OF QUESTION 15 3SUM ==================================================
# 3. Longest Substring Without Repeating Characters
# Given a string s, find the length of the longest substring without repeating characters.
class Solution:
  def lengthOfLongestSubstring(self, s: str) -> int:
    if len(s) == 0 or len(s) == 1:
      return len(s)
    longest_str = ''
    i = 0
    while i < len(s):
      new_str = ''
      for j in range(i, len(s)):
        if s[j] in new_str:
          new_str = s[i:j]
          break
        else:
          new_str += s[j]

      if len(new_str) > len(longest_str):
        longest_str = new_str
      i += 1
    return longest_str, len(longest_str)
  
new_solution = Solution()
# print(new_solution.lengthOfLongestSubstring('abcabcbb'))
# print(new_solution.lengthOfLongestSubstring('bbbbb'))
print(new_solution.lengthOfLongestSubstring('pwwkew'))
# ============================================== END OF QUESTION 15 3SUM ====================================================
