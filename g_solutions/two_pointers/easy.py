from typing import List


# leetcode 392 
# time complexity O(n + m)
# space complexity O(1)
class Solution392:
  def isSubsequence(self, s: str, t: str) -> bool:
    index_s = 0 # pointer for string s
    index_t = 0 # pointer for string t
    while index_s < len(s) and index_t < len(t):
      if s[index_s] == t[index_t]:
        index_s += 1 # move to the next character in s
      index_t += 1 # always move to the next character in t

    # after loop, check if index_s reached the end of s
    return index_s == len(s)
  
# solution = Solution392()
# print(solution.isSubsequence('abc', 'ahbgdc'))


# leetcode 2932 
# time complexity O(n^2)
# space complexity O(1)
class Solution2932:
  def maximumStrongPairXor(self, nums: List[int]) -> int:
    # step 1 sort the array
    nums.sort()
    # step 2 initialize the maximum xor value to 0
    max_xor = 0
    # step 3 use a sliding window to find the maximum XOR of strong pairs
    for i in range(len(nums)):
      for j in range(i, len(nums)):
        x = nums[i]
        y = nums[j]
        # step 4 check if the pair (x, y) satisfies the strong pair condition
        if abs(x - y) <= min(x, y):
          # step 5 calculate the XOR of the pair
          current_xor = x ^ y
          # step 6 update the maximum XOR value if the current XOR is greater than existing max_xor
          max_xor = max(current_xor, max_xor)
    return max_xor

# solution = Solution2932()
# print(solution.maximumStrongPairXor([1,2,3,4,5]))
# print(solution.maximumStrongPairXor([10,100]))
# print(solution.maximumStrongPairXor([5,6,25,30]))


# leetcode 2367
class Solution2367:
  def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
    seen = set()
    count = 0
    for num in nums:
      if num - diff in seen and num - 2 * diff in seen:
        count += 1
      seen.add(num)
    return count
  
# solution = Solution2367()
# print(solution.arithmeticTriplets([0,1,4,6,7,10], 3))


# leetcode 2200
class Solution2200:
  def findKDistantIndices(self, nums: List[int], key: int, k: int) -> List[int]:
    result = set()
    n = len(nums)
    for j in range(n):
      if nums[j] == key:
        for i in range(n):
          if abs(i - j) <= k:
            result.add(i)
    return sorted(result)
  

# solution = Solution2200()
# print(solution.findKDistantIndices([3,4,9,1,3,9,5], 9, 1))
# print(solution.findKDistantIndices([2,2,2,2,2], 2, 2))

# leetcode 696
class Solution696:
  def countBinarySubstrings(self, s: str) -> int:
    n = len(s)
    # group bits together in terms of consecutive
    groupings = []
    current_count = 1
    for i in range(1, n):
      if s[i] == s[i - 1]:
        current_count += 1
      else:
        groupings.append(current_count)
        current_count = 1
    groupings.append(current_count)
    result = 0
    for i in range(len(groupings) - 1):
      result += min(groupings[i], groupings[i + 1])
    return result
  
# solution = Solution696()
# print(solution.countBinarySubstrings('00110011'))

# leetcode 917
class Solution917:
  def reverseOnlyLetters(self, s: str) -> str:
    s_list = list(s)
    left = 0
    right = len(s) - 1
    while left < right:
      if not s_list[left].isalpha():
        left += 1
      elif not s_list[right].isalpha():
        right -= 1
      else:
        s_list[left], s_list[right] = s_list[right], s_list[left]
        left += 1
        right -= 1
    return ''.join(s_list)
  
# solution = Solution917()
# print(solution.reverseOnlyLetters("a-bC-dEf-ghIj"))

# leetcode 1455
class Solution1455:
  def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
    words = sentence.split()
    for index, word in enumerate(words):
      if word.startswith(searchWord):
        return index + 1
    return -1