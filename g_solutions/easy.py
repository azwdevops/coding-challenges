# leetcode 3032
# recursive without memoization
class SolutionRecursive3032:
  def count_unique_numbers(self, a,b):
    def helper_has_unique_digits(num):
      num_str = str(num)
      return len(set(num_str)) == len(num_str)
    
    def helper_count_recursive(num):
      if num > b:
        return 0
      return (1 if helper_has_unique_digits(num) else 0) + helper_count_recursive(num + 1)
    return helper_count_recursive(a)
  
# solution = SolutionRecursive3032()
# print(solution.count_unique_numbers(10,15))


# leetcode 3032
# recursive with memoization

class SolutionRecursiveMemoization3032:
  def count_unique_numbers_memo(self, a, b):
    def has_unique_digits(num):
      num_str = str(num)
      return len(set(num_str)) == len(num_str)
    
    memo = {}
    def count_unique_recursive(num):
      if num > b:
        return 0
      if num in memo:
        return memo[num]
      result = (1 if has_unique_digits(num) else 0) + count_unique_recursive(num + 1)
      memo[num] = result
      return result
    
    return count_unique_recursive(a)

# solution = SolutionRecursiveMemoization3032()
# print(solution.count_unique_numbers_memo(10,15))

# leetcode 3032
# tabulation
class SolutionTabulation3032:
  def count_unique_numbers_tabulation(self, a, b):
    def has_unique_digits(num):
      num_str = str(num)
      return len(set(num_str)) == len(num_str)
    
    count = 0
    for num in range(a, b + 1):
      if has_unique_digits(num):
        count += 1
    return count
  
# solution = SolutionTabulation3032()
# print(solution.count_unique_numbers_tabulation(10,15))