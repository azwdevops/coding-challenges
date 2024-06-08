from typing import List



# ============================================== START OF QUESTION 15 3SUM ==================================================

# Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

# Notice that the solution set must not contain duplicate triplets.

class ThreeSumClass(object):
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
  

# new_solution = ThreeSumClass()
# print(new_solution.threeSum([-1,0,1,2,-1,-4]))
# print(new_solution.threeSum([0,1,1]))
# print(new_solution.threeSum([0,0,0]))
# print(new_solution.threeSum([1,-1,-1,0]))
        

# ============================================== END OF QUESTION 15 3SUM ==================================================


# ============================================== START OF QUESTION 15 3SUM ==================================================
# 3. Longest Substring Without Repeating Characters
# Given a string s, find the length of the longest substring without repeating characters.
class LongestSubstringClass1:
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
  
# new_solution = LongestSubstringClass1()
# print(new_solution.lengthOfLongestSubstring('abcabcbb'))
# print(new_solution.lengthOfLongestSubstring('bbbbb'))
# print(new_solution.lengthOfLongestSubstring('pwwkew'))

class LongestSubstringClass2:
  def lengthOfLongestSubstring(self, s: str) -> int:
    if len(s) == 0:
      return 0
    map = {}
    max_length = start = 0
    for i in range(len(s)):
      if s[i] in map and start <= map[s[i]]:
        start = map[s[i]] + 1
      else: 
        max_length = max(max_length, i - start + 1)
      map[s[i]] = i
    return (max_length)
  
# new_solution = LongestSubstringClass2()
# print(new_solution.lengthOfLongestSubstring('abcabcbb'))
# print(new_solution.lengthOfLongestSubstring('bbbbb'))
# print(new_solution.lengthOfLongestSubstring('pwwkew'))

# ============================================== END OF QUESTION 15 3SUM ====================================================


# ============================================== START OF QUESTION 560. Subarray Sum Equals K ====================================================
# Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
# A subarray is a contiguous non-empty sequence of elements within an array.

class SubArraySum1:
  def subarraySum(self, nums: list[int], k: int) -> int:
    sum_dict = {0:1}
    count = 0
    current_sum = 0
    for item in nums:
      current_sum += item
      if current_sum - k in sum_dict:
        count += sum_dict[current_sum - k]
      if current_sum in sum_dict:
        sum_dict[current_sum] += 1
      else:
        sum_dict[current_sum] = 1
    return count


# new_solution = SubArraySum1()
# print(new_solution.subarraySum([1,1,1], 2))
# print(new_solution.subarraySum([1,2,3], 3))
# print(new_solution.subarraySum([-1,-1,1], 0))


# ============================================== END OF QUESTION 560. Subarray Sum Equals K ====================================================


# ============================================== START OF QUESTION 8. String to Integer (atoi) ====================================================
# Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer.

# The algorithm for myAtoi(string s) is as follows:

# Whitespace: Ignore any leading whitespace (" ").
# Signedness: Determine the sign by checking if the next character is '-' or '+', assuming positivity is neither present.
# Conversion: Read the integer by skipping leading zeros until a non-digit character is encountered or the end of the string is reached. If no digits were read, then the result is 0.
# Rounding: If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then round the integer to remain in the range. Specifically, integers less than -231 should be rounded to -231, and integers greater than 231 - 1 should be rounded to 231 - 1.
# Return the integer as the final result.
class StringToSignedIntegerSolution:
    def myAtoi(self, s: str) -> int:
      signed_int = ''
      stripped_str = s.strip()
      for index, value in enumerate(stripped_str):
        if index == 0:
          if value in ['-', '+']:
            signed_int += value
          elif not value.isdigit():
            return 0
          else:
            signed_int += value
        else:
          if not value.isdigit():
            break
          else:
            signed_int += value
      if len(signed_int) <= 1 and not signed_int.isdigit():
        return 0
      elif int(signed_int) > 2**31 -1:
        return 2**31 - 1
      elif int(signed_int) < -2**31:
        return -2**31
      else:
        return int(signed_int)
        
# new_solution = StringToSignedIntegerSolution()
# print(new_solution.myAtoi('+-12'))
# print(new_solution.myAtoi('-042'))

# ============================================== END OF QUESTION 8. String to Integer (atoi) ====================================================


# ============================================== START OF QUESTION 13. Roman to Integer ====================================================
# Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

# Symbol       Value
# I             1
# V             5
# X             10
# L             50
# C             100
# D             500
# M             1000
# For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

# Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

# I can be placed before V (5) and X (10) to make 4 and 9. 
# X can be placed before L (50) and C (100) to make 40 and 90. 
# C can be placed before D (500) and M (1000) to make 400 and 900.
# Given a roman numeral, convert it to an integer.

class RomanToIntegerSolution:
  def romanToInt(self, s:str) -> int:
    roman_dict = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000, 'IV':4, 'IX':9,'XL':40,'XC':90, 'CD':400,'CM':900}
    int_list=[]
    i=0
    while i < len(s):
      if i+1 < len(s) and s[i] + s[i+1] in roman_dict:
          int_list.append(roman_dict[s[i] + s[i+1]])
          i += 2
      elif s[i] in roman_dict:
          int_list.append(roman_dict[s[i]])
          i+=1
    return sum(int_list)
  
# new_solution = RomanToIntegerSolution()
# print(new_solution.romanToInt('III'))
# print(new_solution.romanToInt('LVIII'))
# print(new_solution.romanToInt('MCMXCIV'))

# ============================================== END OF QUESTION 13. Roman to Integer ====================================================


# ============================================== START OF QUESTION 937. Reorder Data in Log Files ====================================================
# You are given an array of logs. Each log is a space-delimited string of words, where the first word is the identifier.

# There are two types of logs:

# Letter-logs: All words (except the identifier) consist of lowercase English letters.
# Digit-logs: All words (except the identifier) consist of digits.
# Reorder these logs so that:

# The letter-logs come before all digit-logs.
# The letter-logs are sorted lexicographically by their contents. If their contents are the same, then sort them lexicographically by their identifiers.
# The digit-logs maintain their relative ordering.
# Return the final order of the logs.

class ReorderDataInLogFilesSolution:
  def reorderLogFiles(self, logs: list[str]) -> list[str]:
    letter_logs = []
    digit_logs = []
    for log_item in logs:
        if log_item.split(' ')[1].isdigit():
            digit_logs.append(log_item)
        else:
            letter_logs.append(log_item)
    letter_logs.sort(key=self.sort_letter_logs)
    return letter_logs + digit_logs


  def sort_letter_logs(self, log_item):
      log_item_identifier, *log_item_value = log_item.split(' ')
      return (log_item_value, log_item_identifier)
  
# new_solution = ReorderDataInLogFilesSolution()
# print(new_solution.reorderLogFiles(["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let4 art zero", "let3 art zero"]))

# ============================================== END OF QUESTION 937. Reorder Data in Log Files ====================================================


# ============================================== START OF QUESTION 53. Maximum Subarray ====================================================
# Given an integer array nums, find the subarray with the largest sum, and return its sum.
class MaximumSubarraySolution:
  def maxSubArray(self, nums: list[int]) -> int:
    total_sum = max_sum = nums[0]
    for current_value in nums[1:]:
      total_sum = max(current_value, total_sum + current_value)
      max_sum = max(max_sum, total_sum)
    return max_sum

# new_solution = MaximumSubarraySolution()
# print(new_solution.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))
# print(new_solution.maxSubArray([5,4,-1,7,8]))
# print(new_solution.maxSubArray([-2,1]))
# print(new_solution.maxSubArray([1,-1,1]))

# ============================================== END OF QUESTION 53. Maximum Subarray ====================================================


# ============================================== START OF QUESTION 14. Longest Common Prefix ====================================================
# Write a function to find the longest common prefix string amongst an array of strings. If there is no common prefix, return an empty string "".

class LongestCommonPrefixSolution:
    def longestCommonPrefix(self, strs: list[str]) -> str:
        common_prefix = ''
        strs.sort(key=self.sort_by_length)
        shortest_str = strs[0]
        break_loop = False
        for index, char in enumerate(shortest_str):
          for item in strs:
              if char != item[index]:
                break_loop = True
                break
          if break_loop:
            break
          common_prefix += char
        return common_prefix
    
    def sort_by_length(self, item):
      return len(item)
    
# new_solution = LongestCommonPrefixSolution()
# print(new_solution.longestCommonPrefix(["flower","flow","flight"]))
# print(new_solution.longestCommonPrefix(["dog","racecar","car"]))

# ============================================== END OF QUESTION 14. Longest Common Prefix ====================================================


# ============================================== START OF QUESTION 16. 3Sum Closest ====================================================
# Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers.
# You may assume that each input would have exactly one solution.
class ThreeSumClosestSolution:
  def threeSumClosest(self, nums: list[int], target: int) -> int:
    nums.sort()
    closest_sum = sum(nums[:3])
    nums_length = len(nums)
    for i in range(nums_length - 2):
      start = i + 1
      end = nums_length - 1
      while start < end:
        current_sum = nums[i] + nums[start] + nums[end]
        if abs(current_sum - target) < abs(closest_sum - target):
          closest_sum = current_sum
        if current_sum < target:
          start +=1
        else:
          end -= 1
    return closest_sum

# new_solution = ThreeSumClosestSolution()
# print(new_solution.threeSumClosest([-1,2,1,-4], 1))
# print(new_solution.threeSumClosest([0,0,0], 1))
# print(new_solution.threeSumClosest([1,1,1,1], 0))

# ============================================== END OF QUESTION 16. 3Sum Closest ====================================================


# ============================================== START OF QUESTION 572. Subtree of Another Tree ====================================================
# Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.
# A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

### commented the class since Optional[TreeNode] are not recognized in my system at the moment
# class SubtreeSolution:
#   def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
#     if root is None and subRoot is None:
#         return True
#     if subRoot is None:
#         return True
#     if root is None and subRoot is not None:
#         return False
#     return self.isSame(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

#   def isSame(self, root, subRoot):
#     if root is None and subRoot is None:
#         return True
#     if root is None or subRoot is None:
#         return False
#     return root.val == subRoot.val and self.isSame(root.left, subRoot.left) and self.isSame(root.right, subRoot.right)

# new_solution = SubtreeSolution()
# print(new_solution.isSubtree([3,4,5,1,2,'null','null','null','null', 0], [4,1,2]))
# print(new_solution.isSubtree([3,4,5,1,2], [4,1,2]))


# ============================================== END OF QUESTION 572. Subtree of Another Tree ====================================================


# ============================================== START OF QUESTION 54. Spiral Matrix ====================================================
# Given an m x n matrix, return all elements of the matrix in spiral order.
class SpiralOrderSolution:
  def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
    if not matrix:
      return []
    row_begin = 0
    row_end = len(matrix)
    column_begin = 0
    column_end = len(matrix[0])
    spiral_array = []
    while row_end > row_begin and column_end > column_begin:
      for i in range(column_begin, column_end):
        spiral_array.append(matrix[row_begin][i])

      for j in range(row_begin + 1, row_end - 1):
        spiral_array.append(matrix[j][column_end - 1])

      if row_end != row_begin + 1:
        for m in range(column_end - 1, column_begin - 1, -1):
          spiral_array.append(matrix[row_end - 1][m])

      if column_begin != column_end - 1:
        for n in range(row_end - 2, row_begin, -1):
          spiral_array.append(matrix[n][column_begin])

      row_begin += 1
      row_end -= 1
      column_begin += 1
      column_end -= 1

    return spiral_array



  

# new_solution = SpiralOrderSolution()
# print(new_solution.spiralOrder([[1,2,3],[4,5,6],[7,8,9]]))
# print(new_solution.spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))

# ============================================== END OF QUESTION 54. Spiral Matrix ====================================================


# ============================================== START OF QUESTION 202. Happy Number ====================================================
# Write an algorithm to determine if a number n is happy. A happy number is a number defined by the following process: Starting with any positive integer, 
# replace the number by the sum of the squares of its digits.Repeat the process until the number equals 1 (where it will stay), or it loops endlessly 
# in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy. Return true if n is a happy number, and false if not.

class HappyNumberSolution:
  def isHappy(self, n: int) -> bool:
    seen_numbers = set()
    while self.squareSum(n) not in seen_numbers:
      current_sum = self.squareSum(n)
      if current_sum == 1:
        return True
      else:
        seen_numbers.add(current_sum)
        n = current_sum
    return False

  def squareSum(self, num):
    result = 0
    while num > 0:
      remainder = num % 10
      result = result + remainder ** 2
      num = num // 10
    return result


# new_solution = HappyNumberSolution()
# print(new_solution.isHappy(19))
# print(new_solution.isHappy(2))

# ============================================== END OF QUESTION 202. Happy Number ====================================================


# ============================================== START OF QUESTION 101. Symmetric Tree ====================================================
# Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).
# class SymmetricTreeSolution:
#     def isSymmetric(self, root: Optional[TreeNode]) -> bool:
#       if root is None:
#         return True
#       return self.isMirror(root.left, root.right)
    
#     def isMirror(self, leftroot, rightroot):
#       if leftroot and rightroot:
#         return leftroot.val == rightroot.val and self.isMirror(leftroot.left, rightroot.right) and self.isMirror(leftroot.right, rightroot.left)
#       return leftroot == rightroot


# ============================================== END OF QUESTION 101. Symmetric Tree ====================================================


# ============================================== START OF QUESTION 5. Longest Palindromic Substring ====================================================
# Given a string s, return the longest palindromic substring in s.

class LongestPalindromicSubstringSolution:
    def longestPalindrome(self, s: str) -> str:
      pass

# ============================================== END OF QUESTION 5. Longest Palindromic Substring ====================================================

