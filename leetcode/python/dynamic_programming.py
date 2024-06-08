from typing import List

# 494. Target Sum
# You are given an integer array nums and an integer target.
# You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.
# For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1". Return the number of different 
# expressions that you can build, which evaluates to target.
class TargetSumSolution:
  def findTargetSumWays(self, nums: List[int], target: int) -> int:
    dp = {} # (index, total) -> # of ways
    def backtrack(index, total):
      if index == len(nums):
        return 1 if total == target else 0
      if (index, total) in dp:
        return dp[(index, total)]
      dp[(index, total)] = (backtrack(index + 1, total + nums[index]) + backtrack(index + 1, total-nums[index]))

      return dp[(index, total)]
    
    return backtrack(0,0)


# new_solution = TargetSumSolution()
# print(new_solution.findTargetSumWays([1,1,1,1,1], 3))


# 70. Climbing Stairs
# You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
class ClimbingStairs:
  def climbingStairs(self, n:int) -> int:
    one, two = 1,1
    for i in range(n-1):
      temp = one
      one = one + two
      two = temp
    return one
  
# new_solution = ClimbingStairs()
# print(new_solution.climbingStairs(3))


i'm thinking of sliding window, what do you think?