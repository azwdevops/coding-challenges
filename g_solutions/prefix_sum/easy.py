from typing import List
from bisect import bisect_right
from collections import defaultdict

# leetcode 2485

class Solution2485:
  def pivotInteger(self, n: int) -> int:
    # calculate the total sum of numbers from 1 to n
    total_sum = n * (n + 1) // 2
    prefix_sum = 0
    for x in range(1, n + 1):
      # update prefix_sum to include x
      prefix_sum += x
      # calculate suffix_sum as total_sum - prefix_sum + x because x is included in both prefix and suffix
      suffix_sum = total_sum - prefix_sum + x
      # check if prefix_sum is equal to suffix_sum
      if prefix_sum == suffix_sum:
        return x
      
    # if no pivot is found, return -1
    return -1
  
# solution = Solution2485()
# print(solution.pivotInteger(8))

# leetcode 1732
class Solution1732:
  def largestAltitude(self, gain: List[int]) -> int:
    current_altitude = 0
    max_altitude = 0
    for g in gain:
      current_altitude += g
      max_altitude = max(max_altitude, current_altitude)
    return max_altitude
  
# solution = Solution1732()
# print(solution.largestAltitude([-5,1,5,0,-7]))
# print(solution.largestAltitude([-4,-3,-2,-1,4,3,2]))

# leetcode 1588
class Solution1588:
  def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    total_sum = 0
    n = len(arr)
    for i in range(n):
      current_sum = 0
      for j in range(i, n):
        current_sum += arr[j]
        if (j - i + 1) % 2 != 0:
          total_sum += current_sum
    return total_sum

# solution = Solution1588()
# print(solution.sumOddLengthSubarrays([1,4,2,5,3]))

# leetcode 2848
class Solution2848:
  def numberOfPoints(self, nums: List[List[int]]) -> int:
    covered_points = set()
    for start, end in nums:
      for point in range(start, end + 1):
        covered_points.add(point)
    return len(covered_points)
  
# solution = Solution2848()
# print(solution.numberOfPoints([[3,6],[1,5],[4,7]]))
# print(solution.numberOfPoints([[1,3],[5,8]]))


# leetcode 3028
class Solution3028:
  def returnToBoundaryCount(self, nums: List[int]) -> int:
    position = 0
    boundary_count = 0
    for value in nums:
      position += value
      if position == 0:
        boundary_count += 1
    return boundary_count
  

# solution = Solution3028()
# print(solution.returnToBoundaryCount([2,3,-5]))
# print(solution.returnToBoundaryCount([3,2,-3,-4]))


# leetcode 2389
class Solution2389:
  def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
    nums.sort()
    n = len(nums)
    prefix_sums = [0] * n
    # compute t prefix sums
    prefix_sums[0] = nums[0]
    for i in range(1, n):
      prefix_sums[i] = prefix_sums[i - 1] + nums[i]
    answer = []
    for query in queries:
      # use binary search to find the maximum length of subsequence
      index = bisect_right(prefix_sums, query)
      answer.append(index)
    return answer
      

# solution = Solution2389()
# print(solution.answerQueries([4,5,2,1], [3,10,21]))


# leetcode 1991
class Solution1991:
  def findMiddleIndex(self, nums: List[int]) -> int:
    total_sum = sum(nums)
    left_sum = 0
    for i in range(len(nums)):
      right_sum = total_sum - left_sum - nums[i]
      if right_sum == left_sum:
        return i
      left_sum += nums[i]
      
    return -1
  
# solution = Solution1991()
# print(solution.findMiddleIndex([2,3,-1,8,4]))
# print(solution.findMiddleIndex([1,-1,4]))

# leetcode 1413
class Solution1413:
  def minStartValue(self, nums: List[int]) -> int:
    min_cumulative_sum = float('inf')
    cumulative_sum = 0
    for num in nums:
      cumulative_sum += num
      min_cumulative_sum = min(min_cumulative_sum, cumulative_sum)
    if min_cumulative_sum < 0:
      return 1 - min_cumulative_sum
    return 1
  

# solution = Solution1413()
# print(solution.minStartValue([-3,2,-3,4,2]))
# print(solution.minStartValue([1,2]))
# print(solution.minStartValue([1, -2, -3]))

# leetcode 303
class Solution303:
  def __init__(self, nums):
    """
      Initializes the object with the integer array nums.
      Preprocess the array to compute prefix sums.
    """
    n = len(nums)
    self.prefix_sums = [0] * (n + 1)
    for i in range(n):
      self.prefix_sums[i + 1] = self.prefix_sums[i] + nums[i]


  def sumRange(self, left: int, right: int) -> int:
    """
      returns the sum of the elements of nums between indices left amd right inclusive
    """
    return self.prefix_sums[right + 1] - self.prefix_sums[left]
  
# solution = Solution303([-2, 0, 3, -5, 2, -1])
# print(solution.sumRange(0,2))
# print(solution.sumRange(2, 5))
# print(solution.sumRange(0,5))

# leetcode 1422
class Solution1422:
  def maxScore(self, s: str) -> int:
    total_ones = s.count('1')
    left_zeros = 0
    right_ones = total_ones
    max_score = 0
    for i in range(len(s) - 1):
      if s[i] == '0':
        left_zeros += 1
      else:
        right_ones -= 1
      current_score = left_zeros + right_ones
      max_score = max(max_score, current_score)

    return max_score
  
# solution = Solution1422()
# print(solution.maxScore('0100'))


# leetcode 1854
class Solution1854:
  def maximumPopulation(self, logs: List[List[int]]) -> int:
    # initialize a dictionary to track changes in population
    population_changes = defaultdict(int)
    # process each log entry
    for birth, death in logs:
      population_changes[birth] += 1
      population_changes[death] -= 1

    # initialize variables to track the current population and maximum population
    current_population = 0
    max_population = 0
    max_year = 0

    # iterate through the years in sorted order
    for year in sorted(population_changes.keys()):
      current_population += population_changes[year]
      if current_population > max_population:
        max_population = current_population
        max_year = year
    return max_year

# solution = Solution1854()
# print(solution.maximumPopulation([[1993,1999],[2000,2010]]))

# leetcode 1893
class Solution1893:
  def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
    # initialize the coverage array
    covered = [False] * (right - left + 1)

    # mark covered intervals
    for start, end in ranges:
      for i in range(max(start, left), min(end, right) + 1):
        covered[i - left] = True
    # check if all values in the range [left, right] are covered
    return all(covered)

# solution = Solution1893()
# print(solution.isCovered([[1,2],[3,4],[5,6]], 2, 5))
# print(solution.isCovered([[1,10],[10,20]], 21,21))

# leetcode 2391
class Solution2391:
  def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
    minutes = 0
    for i in range(len(garbage)):
      if i > 0:
        minutes += travel[i - 1] * len(garbage[i])
      for item in garbage[i]:
        minutes += 1
    return minutes

# solution = Solution2391()
# print(solution.garbageCollection(["G","P","GP","GG"], [2,4,3]))

# leetcode 2391
class Solution2391:
  def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
    total_time = 0
    last_house = {'M':0, 'P':0, 'G':0}
    # calculate total garbage count and track the last house for each type
    for i in range(len(garbage)):
      for char in garbage[i]:
        total_time += 1
        last_house[char] = i

    # calculate the travel time for each truck
    travel_time = {'M':0, 'P':0, 'G':0}
    for i in range(1, len(garbage)):
      if i <= last_house['M']:
        travel_time['M'] += travel[i - 1]
      if i <= last_house['G']:
        travel_time['G'] += travel[i - 1]
      if i <= last_house['P']:
        travel_time['P'] += travel[i - 1]
    total_time += sum(travel_time.values())
    return total_time

# solution = Solution2391()
# print(solution.garbageCollection(["G","P","GP","GG"], [2,4,3]))

# leetcode 1442
class Solution1442:
  def countTriplets(self, arr: List[int]) -> int:
    n = len(arr)
    prefix_xor = [0] * (n + 1)
    # compute prefix xor
    for i in range(n):
      prefix_xor[i + 1] = prefix_xor[i] ^ arr[i]
    count = 0

    # iterate over all possible (i, j, k) combinations
    for k in range(1, n):
      for j in range(k + 1):
        if prefix_xor[j] == prefix_xor[k+1]:
          count += k - j

    return count

# solution = Solution1442()
# print(solution.countTriplets([2,3,1,6,7]))

# leetcode 1829
class Solution1829:
  def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
    n = len(nums)
    max_value = (1 << maximumBit) - 1 # 2^maximumBit - 1
    # compute the cumulative XOR of the entire array
    cumulative_xor = 0
    for num in nums:
      cumulative_xor ^= num
    # initialize the array answer
    answer = [0] * n
    # iterate from the last element to the first
    for i in range(n - 1, -1, -1):
      answer[n - 1 - i] = max_value ^ cumulative_xor
      cumulative_xor ^= nums[i]

# solution = Solution1829()
# print(solution.getMaximumXor([0,1,1,3], 2))