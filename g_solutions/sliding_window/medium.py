from typing import List
from collections import defaultdict

# leetcode 2743
class Solution2743:
  def numberOfSpecialSubstrings(self, s: str) -> int:
    n = len(s)
    left = 0
    count = 0
    char_set = set()
    for right in range(n):
      while s[right] in char_set:
        char_set.remove(s[left])
        left += 1
      char_set.add(s[right])
      count += right - left + 1
    print(char_set)
    return count

  
# solution = Solution2743()
# print(solution.numberOfSpecialSubstrings('abcd'))
# print(solution.numberOfSpecialSubstrings('ooo'))

# leetcode 1100
class Solution1100:
  def numberOfSpecialKLengthSubstrings(self, s: str, k: int) -> int:
    if k > len(s):
      return 0
    left = 0
    count = 0
    char_set = set()
    for right in range(len(s)):
      # ensure that the window contains unique characters
      while s[right] in char_set:
        char_set.remove(s[left])
        left += 1
      char_set.add(s[right])
      # check if we have a valid window of size k
      if right - left + 1 == k:
        count += 1
        # move left to shrink the window for next iteration
        char_set.remove(s[left])
        left += 1

    return count 
  
# solution = Solution1100()
# print(solution.numberOfSpecialKLengthSubstrings('havefunonleetcode', 5))
# print(solution.numberOfSpecialKLengthSubstrings('home', 5))

# leetcode 1852 
class Solution1852:
  def distinctNumbers(self, nums: List[int], k: int) -> List[int]:
    n = len(nums)
    left = 0
    right = k
    ans = []
    while left < n and right <= n:
      sub = nums[left:right]
      ans.append(len(set(sub)))
      left += 1
      right += 1
    return ans
  
# solution = Solution1852()
# print(solution.distinctNumbers([1,2,3,2,2,1,3], 3))
# print(solution.distinctNumbers([1,1,1,1,2,3,4], 4))
      

# leetcode 1248
class Solution1248:
  def numberOfSubarrays(self, nums: List[int], k: int) -> int:
    prefix_sums = {}
    prefix_sums[0] = 1
    current_old_count = 0
    count = 0
    for num in nums:
      if num % 2 != 0:
        current_old_count += 1
      if current_old_count - k in prefix_sums:
        count += prefix_sums[current_old_count - k]
      prefix_sums[current_old_count] = prefix_sums.get(current_old_count, 0) + 1
    return count
  
# solution  = Solution1248()
# print(solution.numberOfSubarrays([1,1,2,1,1], 3))
# print(solution.numberOfSubarrays([2,2,2,1,2,2,1,2,2,2], 2))

# leetcode 3191
class Solution3191:
  def minOperations(self, nums: List[int]) -> int:
    def flip_triplet(start):
      for i in range(3):
        nums[start + i] = 1 - nums[start + i]
    n = len(nums)
    operations = 0
    for i in range(n - 2): # we only need to check until n - 3
      if nums[i] == 0:
        flip_triplet(i)
        operations += 1
    # after processing check if all elements are 1
    for num in nums:
      if num == 0:
        return -1
    return operations
  
# solution = Solution3191()
# print(solution.minOperations([0,1,1,1,0,0]))
# print(solution.minOperations([0,1,1,1]))


# leetcode 1343
class Solution1343:
   def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
    n = len(arr)
    required_sum = k * threshold
    current_sum = sum(arr[:k])
    count = 0
    # check the sum of the first window
    if current_sum >= required_sum:
      count += 1
    # slide the window over the array
    for i in range(k, n):
      current_sum += arr[i] - arr[i - k]
      if current_sum >= required_sum:
        count += 1
    return count
   
  
# solution = Solution1343()
# print(solution.numOfSubarrays([2,2,2,2,5,5,5,8], 3, 4))

# leetcode 1493
class Solution1493:
  def longestSubarray(self, nums: List[int]) -> int:
    left = 0
    zero_count = 0
    max_length = 0
    for right in range(len(nums)):
      if nums[right] == 0:
        zero_count += 1
      while zero_count > 1:
        if nums[left] == 0:
          zero_count -= 1
        left += 1

      # calculate the length of the current window and subtract one (because we should delete one element per instructions) that's why we do not do right - left + 1
      max_length = max(max_length, right - left)

    # if the array is all ones, we have to delete one, so return len(nums) - 1
    return max_length
  
# solution = Solution1493()
# print(solution.longestSubarray([1,1,0,1]))
# print(solution.longestSubarray([0,1,1,1,0,1,1,0,1]))
# print(solution.longestSubarray([1,1,1]))

# leetcode 2024
class Solution2024:
  def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
    def maxConsecutiveChar(answerKey, k, char):
      left = 0
      max_length = 0
      count = 0
      for right in range(len(answerKey)):
        if answerKey[right] != char:
          count += 1
        while count > k:
          if answerKey[left] != char:
            count -= 1
          left += 1
        max_length = max(max_length, right - left + 1)
      return max_length
    
    # calculate the max consecutive T's and F's
    max_T = maxConsecutiveChar(answerKey, k, 'T')
    max_F = maxConsecutiveChar(answerKey, k, 'F')

    return max(max_T, max_F)

# solution = Solution2024()
# print(solution.maxConsecutiveAnswers('TTFF', 2))
# print(solution.maxConsecutiveAnswers('TFFT', 1))
# print(solution.maxConsecutiveAnswers('TTFTTFTT', 1))


# leetcode 1358
class Solution1358:
  def numberOfSubstrings(self, s: str) -> int:
    n = len(s)
    count = 0
    left = 0
    my_dict = {'a': 0, 'b': 0, 'c': 0}
    for right in range(n):
      my_dict[s[right]] += 1
      while all(my_dict[char] > 0 for char in 'abc'):
        count += n - right
        my_dict[s[left]] -= 1
        left += 1
    return count
  
# solution = Solution1358()
# print(solution.numberOfSubstrings('abcabc'))
# print(solution.numberOfSubstrings('aaacb'))
# print(solution.numberOfSubstrings('abc'))

# leetcode 2799
class Solution2799:
  def countCompleteSubarrays(self, nums: List[int]) -> int:
    # fine the number of distinct elements in the entire array
    total_distinct = len(set(nums))
    count = 0
    left = 0
    my_dict = defaultdict(int)
    current_distinct = 0
    for right in range(len(nums)):
      if my_dict[nums[right]] == 0:
        current_distinct += 1
      my_dict[nums[right]] += 1

      # check if the window is valid (contains all distinct elements)
      while current_distinct == total_distinct:
        count += len(nums) - right
        my_dict[nums[left]] -= 1
        if my_dict[nums[left]] == 0:
          current_distinct -= 1
        left += 1
    return count
  
# solution = Solution2799()
# print(solution.countCompleteSubarrays([1,3,1,2,2]))
# print(solution.countCompleteSubarrays([5,5,5,5]))

# leetcode 1052
class Solution1052:
  def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
    n = len(customers)
    initial_satisfaction = 0
    # calculate the initial satisfaction without using the technique
    for i in range(n):
      if grumpy[i] == 0:
        initial_satisfaction += customers[i]
    # sliding window to find the maximum additional satisfaction
    max_gain = 0
    current_gain = 0
    # initial window calculation
    for i in range(minutes):
      if grumpy[i] == 1:
        current_gain += customers[i]

    max_gain = current_gain
    # sliding window across the rest of the array
    for i in range(minutes, n):
      if grumpy[i] == 1:
        current_gain += customers[i]
      if grumpy[i - minutes] == 1:
        current_gain -= customers[i - minutes]
      max_gain = max(max_gain, current_gain)

    return initial_satisfaction + max_gain
  

# solution = Solution1052()
# print(solution.maxSatisfied([1,0,1,2,1,1,7,5], [0,1,0,1,0,1,0,1], 3))
# print(solution.maxSatisfied([1],[0],1))

# leetcode 1004
class Solution1004:
  def longestOnes(self, nums: List[int], k: int) -> int:
    left = 0
    max_len = 0
    zero_count = 0
    for right in range(len(nums)):
      if nums[right] == 0:
        zero_count += 1
        while zero_count > k:
          if nums[left] == 0:
            zero_count -= 1
          left += 1
        max_len = max(max_len, right - left + 1)
    return max_len

# solution = Solution1004()
# print(solution.longestOnes([1,1,1,0,0,0,1,1,1,1,0], 2))
# print(solution.longestOnes([0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], 3))

# leetcode 930
class Solution930:
  def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
    prefix_sum_count = defaultdict(int)
    prefix_sum_count[0] = 1 # initialize with zero prefix sum
    current_prefix_sum = 0
    count = 0
    for num in nums:
      current_prefix_sum += num
      if (current_prefix_sum - goal) in prefix_sum_count:
        count += prefix_sum_count[current_prefix_sum - goal]
      prefix_sum_count[current_prefix_sum] += 1
    return count
  
# solution = Solution930()
# print(solution.numSubarraysWithSum([1,0,1,0,1], 2))
# print(solution.numSubarraysWithSum([0,0,0,0,0], 0))