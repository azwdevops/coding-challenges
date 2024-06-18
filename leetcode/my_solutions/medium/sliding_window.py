from typing import List

# =========================== START OF QUESTION 567 ==================================
class StringPermutationSolution:
  def checkInclusion(self, s1: str, s2: str) -> bool:
    s1_length ,s2_length = len(s1), len(s2)
    if s1_length > s2_length or s2_length == 0 or s1_length == 0:
      return False
    s1_sorted = ''.join(sorted(s1))
    left = 0
    right = s1_length
    while right <= s2_length:
      current_substring = ''.join(sorted(s2[left:right]))
      if s1_sorted == current_substring:
        return True
      right += 1
      left += 1
    return False
  

# new_solution = StringPermutationSolution()
# print(new_solution.checkInclusion('ab', 'eidbaooo'))
# print(new_solution.checkInclusion('ab', 'eidboaoo'))
      


# =========================== END OF QUESTION 567 ==================================

# =========================== START OF QUESTION 209 ==================================
class MinimumSubarraySumSolution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
      nums_length = len(nums)
      left = 0
      minimum_length = float('inf')
      current_total = 0
      for right in range(nums_length):
        current_total += nums[right]
        while current_total >= target:
          # print(current_total, right, left, nums[left])
          if current_total >= target:
            minimum_length = min(minimum_length, right - left + 1)
            current_total -= nums[left]
            left += 1
      return minimum_length if minimum_length != float('inf')  else 0 
    

# new_solution = MinimumSubarraySumSolution()
# print(new_solution.minSubArrayLen(7, [2,3,1,2,4,3]))
# print(new_solution.minSubArrayLen(11, [1,1,1,1,1,1,1,1]))

# =========================== END OF QUESTION 209 ==================================


# =========================== START OF QUESTION 187 ==================================
class RepeatedDNASolution:
  def findRepeatedDnaSequences(self, s: str) -> List[str]:
    my_dict = {}
    repeated_dna = []

    for index in range(len(s) - 9):
      current_dna = s[index:index + 10]
      my_dict[current_dna] = my_dict.get(current_dna, 0) + 1
      if my_dict[current_dna] > 1 and current_dna not in repeated_dna:
        repeated_dna.append(current_dna)

    return repeated_dna

# new_solution = RepeatedDNASolution()
# print(new_solution.findRepeatedDnaSequences('AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT'))
# print(new_solution.findRepeatedDnaSequences('AAAAAAAAAAAAA'))
# print(new_solution.findRepeatedDnaSequences('AAAAAAAAAAA'))

# =========================== END OF QUESTION 187 ==================================


# =========================== START OF QUESTION 438. Find All Anagrams in a String ==================================
class StringAnagramSolution:
  def findAnagrams(self, s: str, p: str) -> List[int]:
    s_length = len(s)
    p_length = len(p)
    if p_length > s_length or p_length == 0:
        return []
    sorted_p = ''.join(sorted(p))
    start_indeces = []

    for i in range(s_length - p_length + 1):
        current = s[i:i+p_length]
        sorted_current = ''.join(sorted(current))
        if sorted_current == sorted_p:
            start_indeces.append(i)
    return start_indeces

# =========================== END OF QUESTION 438. Find All Anagrams in a String ====================================
