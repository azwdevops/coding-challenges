from typing import List
from collections import defaultdict


# leetcode 1876
class Solution1876:
    def countGoodSubstrings(self, s: str) -> int:
        left = 0
        right = 3
        count = 0
        while right <= len(s):
            current_str = s[left:right]
            if len(current_str) == len(set(current_str)):
                count += 1
            left += 1
            right += 1
        return count
    
# solution = Solution1876()
# print(solution.countGoodSubstrings('xyzzaz'))
# print(solution.countGoodSubstrings('aababcabc'))

# leetcode 1652
class Solution1652:
    def decrypt(self, code: List[int], k: int) -> List[int]:
      n = len(code)
      # special case when k == 0
      if k == 0:
        return [0] * n
      # extend the array to handle the circular nature
      extended_code = code + code
      
      # initialize the result array
      result = [0] * n
      # define the sliding window range
      start, end = (1, k) if k > 0 else (n + k, n - 1)
      # calculate initial window sum
      window_sum = sum(extended_code[start:end + 1])
      for i in range(n):
        result[i] = window_sum
        # slide the window
        window_sum -= extended_code[start]
        start += 1
        end += 1
        window_sum += extended_code[end]
      return result
        
    
# solution = Solution1652()
# print(solution.decrypt([5,7,1,4], 3))

# leetcode 3090
# time O(n)
# space O(1)
class Solution3090:
	def maximumLengthSubstring(self, s: str) -> int:
		# initialize the pointers and the character count dictionary
		left = 0
		max_length = 0
		char_count = {}
		for right in range(len(s)):
			# expand window by including the character at the right pointer
			char_count[s[right]] = char_count.get(s[right], 0) + 1
			# shrink the window from the left if any character occurs more than twice
			while any(count > 2 for count in char_count.values()):
				char_count[s[left]] -= 1
				if char_count[s[left]] == 0:
					del char_count[s[left]]
				left += 1

			# update the maximum length of valid windows
			max_length = max(max_length, right - left + 1)
      
		return max_length
      
# solution = Solution3090()
# print(solution.maximumLengthSubstring('bcbbbcba'))
# print(solution.maximumLengthSubstring('aaaa'))

# leetcode 2269
class Solution2269:
  def divisorSubstrings(self, num: int, k: int) -> int:
    # convert num to string
    num_str = str(num)
    length = len(num_str)
    k_beauty_count = 0
    # iterate through possible substrings of length k
    for i in range(length - k + 1):
      substring = num_str[i:i+k]
      substring_int = int(substring)

      # check if the substring is a divisor of num and not zero
      if substring_int != 0 and num % substring_int == 0:
        k_beauty_count += 1
    return k_beauty_count
  
# solution = Solution2269()
# print(solution.divisorSubstrings(240,2))
# print(solution.divisorSubstrings(430043, 2))
    

# leetcode 2379
class Solution2379:
  def minimumRecolors(self, blocks: str, k: int) -> int:
    n = len(blocks)
    # initial window of size k
    min_operations = float('inf')
    current_white_blocks = sum(1 for i in range(k) if blocks[i] == 'W')
    # initial minimum operations is the white blocks in the first window
    min_operations = current_white_blocks
    # slide the window from left to right
    for i in range(1, n - k + 1):
      # remove the effect of the leftmost block of the previous window
      if blocks[i - 1] == 'W':
        current_white_blocks -= 1
      # add the effect of the new rightmost block of the current window
      if blocks[i + k - 1] == 'W':
        current_white_blocks += 1
      # update the minimum operations 
      min_operations = min(min_operations, current_white_blocks)
    return min_operations

# solution = Solution2379()
# print(solution.minimumRecolors('WBBWWBBWBW', 7))  
# print(solution.minimumRecolors('WBWBBBW', 2))  

# leetcode 1984
class Solution1984:
	def minimumDifference(self, nums: List[int], k: int) -> int:
		if k == 1:
			return 0 # if k is 1 the difference is always 0 because we only have one element
		nums.sort() # step 1 sort the array
		min_diff = float('inf') # initialize the minimum difference to infinity
    # step 2 use a sliding window to find the minimum difference in a sorted array
		for i in range(len(nums) - k + 1):
			current_diff = nums[i + k - 1] - nums[i]
			min_diff = min(min_diff, current_diff)
		return min_diff

# solution = Solution1984()
# print(solution.minimumDifference([9,4,1,7], 2))

# leetcode 594
class Solution594:
  def findLHS(self, nums: List[int]) -> int:
    # step 1 count the frequency of each element
    frequency = {}
    for num in nums:
      frequency[num] = frequency.get(num, 0) + 1 
    # step 2 find the longest harmonious subsequence
    max_length = 0
    for num in frequency:
      if num + 1 in frequency:
        current_length = frequency[num] + frequency[num + 1]
        max_length = max(max_length, current_length)
    return max_length
  
# solution = Solution594()
# print(solution.findLHS([1,3,2,2,5,2,3,7]))
# print(solution.findLHS([1,2,3,4]))
# print(solution.findLHS([1,1,1,1]))

# leetcode 1176
# time complexity O(k)
# space complexity O(1)
class Solution1176:
    def dieterPoints(self, calories, k, lower, upper):
      n = len(calories)
      if n < k:
        return 0 # if there are fewer than k days, no points can be gained or lost
      # initialize the sum of the first window
      current_sum = sum(calories[:k])
      points = 0
      # initial points calculation for the first window
      if current_sum < lower:
        points -= 1
      elif current_sum > upper:
        points += 1
      # slide the window from start to end
      for i in range(1, n - k + 1):
        # slide the window: subtract the element that's sliding out and add the new element
        current_sum = current_sum - calories[i - 1] + calories[i + k - 1]
        # calculate points for the new window
        if current_sum < lower:
          points -= 1
        elif current_sum > upper:
          points += 1
      return points

    
# solution = Solution1176()
# print(solution.dieterPoints([1,2,3,4,5], 1, 3, 3))
# print(solution.dieterPoints([3,2], 2, 0, 1))


# leetcode 219
class Solution219:
  def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    # dictionary to store the most recent index at each element
    index_map = {}
    # iterate through the array
    for i, num in enumerate(nums):
      if num in index_map and abs(i - index_map[num]) <= k:
        return True
      # update the dictionary with the current index of the element
      index_map[num] = i
    return False
  
# solution = Solution219()
# print(solution.containsNearbyDuplicate([1,2,3,1], 3))
# print(solution.containsNearbyDuplicate([1,0,1,1], 1))
# print(solution.containsNearbyDuplicate([1,2,3,1,2,3], 2))

# leetcode 643
class Solution643:
  def findMaxAverage(self, nums: List[int], k: int) -> float:
    n = len(nums)
    # compute the sum of the first k elements
    max_sum = current_sum = sum(nums[:k])
    # slide the window from start to end
    for i in range(k, n):
      current_sum = current_sum - nums[i - k] + nums[i]
      max_sum = max(max_sum, current_sum)
    # calculate the maximum average
    return max_sum / k
  
# solution = Solution643()
# print(solution.findMaxAverage([1,12,-5,-6,50,3], 4))
# print(solution.findMaxAverage([5], 1))

# leetcode 3095
class Solution3095:
  def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
    n = len(nums)
    min_length = float('inf') # same as infinity since the largest subarray is of length n + 1
    current_or = 0
    left = 0
    for right in range(n):
      current_or |= nums[right]
      while current_or >= k and left <= right:
        min_length = min(min_length, right - left + 1)
        current_or ^= nums[left]
        left += 1
    return min_length if min_length != n + 1 else -1
  
# solution = Solution3095()
# print(solution.minimumSubarrayLength([1,2,3], 2))
# print(solution.minimumSubarrayLength([2,1,8], 10))
# print(solution.minimumSubarrayLength([1,2], 0))


# leetcode 2760
class Solution2760:
  def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
    n = len(nums)
    max_length = 0 # initialize the maximum length to 0
    left = 0
    while left < n:
      # find the starting point of the subarray which must be an even number
      if nums[left] % 2 != 0 or nums[left] > threshold:
        left += 1
        continue
      
      # initialize the right pointer and the length of the current subarray
      right = left
      current_length = 0
      while right < n and nums[right] <= threshold:
        if right > left and nums[right] % 2 == nums[right - 1] % 2:
          break
        current_length += 1
        right += 1
      max_length = max(max_length, current_length)
      left += 1
    return max_length
  
# solution = Solution2760()
# print(solution.longestAlternatingSubarray([3,2,5,4], 5))
# print(solution.longestAlternatingSubarray([1,2], 2))
# print(solution.longestAlternatingSubarray([2, 3, 4, 5], 4))

