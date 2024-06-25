from typing import List

# LEETCODE 518
class Solution518:
  def change(self, amount: int, coins: List[int]) -> int:
    table = [[] for _ in range(amount + 1)]
    table[0] = [[]]

    for i in range(amount + 1):
      if table[i]:
        for num in coins:
          if i + num <= amount:
            for combination in table[i]:
              table[i + num].append(combination + [num])

    print(table)


solution518 = Solution518()
print(solution518.change(5, [1,2,5]))