

# =================== START OF LEETCODE 3032 - Count Numbers With Unique Digits II ================================================
class CountNumbersWithUniqueDigits2Solution:
  def countNumbers(self, a:int, b: int) -> int:
    count = 0
    memo = {}
    for num in range(a, b+1):
      if self.has_unique_digits(num, memo):
        count +=1
    return count
            
  def has_unique_digits(self, num, memo):
    if num in memo:
      return memo[num]
    digits = set()
    original_num = num
    while num > 0:
      current_digit = num % 10
      if current_digit in digits:
        memo[original_num] = False
        return False
      digits.add(current_digit)
      num = num // 10

    memo[original_num] = True
    return True
  
# solution = CountNumbersWithUniqueDigits2Solution()
# print(solution.countNumbers(1,20))
# print(solution.countNumbers(9,19))
# print(solution.countNumbers(80,120))

# =================== END OF LEETCODE 3032 - Count Numbers With Unique Digits II ================================================