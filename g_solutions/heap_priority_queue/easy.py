import heapq
from typing import List
# leetcode 506
class Solution506:
  def findRelativeRanks(self, score: List[int]) -> List[str]:
    # note python heapq library only supports min-heap be default, we can simulate a max-heap by storing negative values of the scores
    # max-heap using negative values
    max_heap = []
    # push all scores with their indices as tuples into the heap
    for i, s in enumerate(score):
      heapq.heappush(max_heap, (-s, i))
    result = ["" for _ in score]

    rank = 1
    while max_heap:
      _, index = heapq.heappop(max_heap)
      if rank == 1:
        result[index] = 'Gold Medal'
      elif rank == 2:
        result[index] = 'Silver Medal'
      elif rank == 3:
        result[index] = 'Bronze Medal'
      else:
        result[index] = str(rank)
      rank += 1
    return result


arr = [1,2,3,4,5,6]
print(arr[0:-2])