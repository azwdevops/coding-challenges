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