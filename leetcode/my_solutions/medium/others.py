from typing import List
import random

# =================== START OF LEETCODE 49. Group Anagrams ========================================

class GroupAnagramSolution:
  def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    my_dict = {}
    for item in strs:
      sorted_item = ''.join(sorted(item))
      if my_dict.get(sorted_item):
        my_dict[sorted_item].append(item)
      else:
        my_dict[sorted_item] = [item]
    return list(my_dict.values())

# new_solution = GroupAnagramSolution()
# print(new_solution.groupAnagrams(["eat","tea","tan","ate","nat","bat"]))

# =================== END OF LEETCODE 49. Group Anagrams ========================================


# =================== END OF LEETCODE 146. LRU Cache ========================================

class LRUCache:

    def __init__(self, capacity: int):
      self.capacity = capacity
      self.cache = {}
       
    def get(self, key: int) -> int:
      value = self.cache.get(key)
      if value is not None:
        self.cache.pop(key)
        self.cache[key] = value
        return value
      else:
          return -1

    def put(self, key: int, value: int) -> None:
      if not self.cache.get(key) and len(self.cache.keys()) + 1 > self.capacity:
        index = list(self.cache.keys())[0]
        self.cache.pop(index)
      if self.cache.get(key):
        self.cache.pop(key)
      
      self.cache[key]= value

    def __repr__(self):
      return f'{self.cache}'


# new_solution = LRUCache(4)
# print(new_solution.put(1,1))
# print(new_solution.put(2,2))
# print(new_solution)
# print(new_solution.get(1))
# print(new_solution.put(2,6))
# print(new_solution.put(3,3))
# print(new_solution.get(2))
# print(new_solution.get(1))
# print(new_solution)


# =================== END OF LEETCODE 146. LRU Cache ========================================






  