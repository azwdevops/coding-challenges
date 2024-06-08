
from typing import List

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



