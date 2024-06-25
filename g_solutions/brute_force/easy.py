# leetcode 1668 
# time complexity O(m * n)
# space complexity O(n)
class SolutionTabulation1668:
  def maxRepeating(self, sequence: str, word: str) -> int:
    k = 1
    while True:
      concat_word = word * k
      if concat_word in sequence:
        k += 1
      else:
        break
    return k - 1
  
# solution = SolutionTabulation1668()
# print(solution.maxRepeating('ababc', 'ab'))