from typing import List

# leetcode 1863
class Solution1863:
  def subsetXORSum(self, nums: List[int]) -> int:
    # initialize result to 0
    result = 0
    # get the number of elements in the array
    n = len(nums)

    # iterate over each element in the array
    for num in nums:
      """
        for each element, its contribution to the final result is its value multiplied by 2^(n-1), where n is the number of elements in nums. 
        This is because each element appears in exactly 2^(n-1) subsets
      """ 
      result += num * (1 << (n - 1))
    return result

# arr = sorted([0,1,2,4,8,3,5,6,7], key=lambda x: [bin(x).count('1'), x])
# print(arr)

# leetcode 2917
class Solution:
  def findKOr(self, nums: List[int], k: int) -> int:
    # determine the maximum bits length which is the number of bits for max value in nums
      max_bits = max(nums).bit_length()
      result = 0

      # iterate over each bit position
      for i in range(max_bits):
        count = 0
        # count the number of 1s at the ith bit position in all numbers
        for num in nums:
          if num & (1<<i):
            count += 1
        # if at least k numbers have 1 at this bit position, set this bit in the result
        if count >= k:
          result |= (1<<i)
      return result

