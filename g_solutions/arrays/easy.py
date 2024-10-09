from typing import List
from collections import defaultdict
# leetcode 1512
class Solution1512:
  def numIdenticalPairs(self, nums: List[int]) -> int:
    # dict to store the count of each number
    count_map = defaultdict(int)
    # variable to store the number of good pairs
    good_pairs_count = 0
    # iterate through each number in the array
    for num in nums:
      # if the num in count_map it means we can form good pairs
      if num in count_map:
        good_pairs_count += count_map[num]
      
      # increment the count of the number in the dictionary
      count_map[num] += 1
    return good_pairs_count
  
# solution = Solution1512()
# print(solution.numIdenticalPairs([1,2,3,1,1,3]))


# leetcode 2011
class Solution2011:
  def finalValueAfterOperations(self, operations: List[str]) -> int:
    value = 0
    for op in operations:
      if '--' in op:
        value -= 1
      elif '++' in op:
        value += 1
    return value

# solution = Solution2011()
# print(solution.finalValueAfterOperations(["--X","X++","X++"]))

# leetcode 1470
class Solution1470:
  def shuffle(self, nums: List[int], n: int) -> List[int]:
    result = []
    for i in range(n):
      result.append(nums[i])
      result.append(nums[i + n])
    return result
  
# solution = Solution1470()
# print(solution.shuffle([2,5,1,3,4,7], 3))


# leetcode 2942
class Solution2942:
  def findWordsContaining(self, words: List[str], x: str) -> List[int]:
    indices = []
    for i in range(len(words)):
      if x in words[i]:
        indices.append(i)
    return indices
  
# solution = Solution2942()
# print(solution.findWordsContaining(["leet","code"], 'e'))

# leetcode 3190
class Solution3190:
  def minimumOperations(self, nums: List[int]) -> int:
    operations = 0
    for num in nums:
      remainder = num % 3
      if remainder != 0:
        operations += 1
    return operations
  
# solution = Solution3190()
# print(solution.minimumOperations([1,2,3,4]))
# print(solution.minimumOperations([1,2,3,4,5]))

# leetcode 2373
class Solution2373:
  def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
    pass