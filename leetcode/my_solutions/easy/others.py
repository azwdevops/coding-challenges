from typing import List

# =================== START OF LEETCODE 202. Happy Number ======================================
class HappyNumberSolution:
  def isHappy(self, n: int) -> bool:
    if n == 1:
      return True
    computed_nums = []
    num = abs(n)

    while num != 1:
      current_total = 0
      for item in str(num):
          current_total += int(item) ** 2
      if current_total in computed_nums:
          return False
      if current_total == 1:
          return True
      computed_nums.append(current_total)
      num = current_total

# =================== END OF LEETCODE 202. Happy Number ======================================


# =================== START OF LEETCODE 1. Two Sum ======================================
class TwoSumSolution:
  def twoSum(self, nums: List[int], target: int) -> List[int]:
    my_dict = {}
    for i in range(len(nums)):
        if my_dict.get(nums[i]) is not None:
          return [i, my_dict[nums[i]]]
        diff = target - nums[i]
        my_dict[diff] = i
    return []

# new_solution = TwoSumSolution()
# print(new_solution.twoSum([2,7,11,15], 9))

# =================== END OF LEETCODE 1. Two Sum ======================================


