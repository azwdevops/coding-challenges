
from typing import List, Optional

# =========================== START OF QUESTION 121 ==========================
# 121. Best Time to Buy and Sell Stock
# reference https://www.youtube.com/watch?v=1pkOgXD63yU&list=PLot-Xpze53leOBgcVsJBEGrHPd_7x_koV&ab_channel=NeetCode

class BuySellStockSolution:
  def maxProfit(self, prices: List[int]) -> int:
    left, right = 0, 1 # left - buy right sell
    maxProfit = 0
    while right < len(prices):
      # check if profitable
      if prices[left] < prices[right]:
        profit = prices[right] - prices[left]
        maxProfit = max(maxProfit, profit)
      else:
        left = right
      right += 1

    return maxProfit
  
# new_solution = BuySellStockSolution()
# print(new_solution.maxProfit([7,1,5,3,6,4]))

# =========================== END OF QUESTION 121 ==========================


# =========================== START OF QUESTION 350 ==================================
# items should appear as many times as they appear on both arrays
class ArrayIntersectionSolution2:
  def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    i = 0
    j = 0
    result = []
    nums1.sort()
    nums2.sort()
    while i < len(nums1) and j < len(nums2):
      if nums1[i] == nums2[j]:
        result.append(nums1[i])
        i+=1
        j+=1
      elif nums1[i] < nums2[j]:
        i+=1
      else:
        j+=1

    return result
# =========================== END OF QUESTION 350 ==================================



