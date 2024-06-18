from typing import List
import collections


# ======================== START OF QUESTION 1 ===========================
def maximum_sum(nums, k):
  max_sum = float('-inf')
  for i in range(len(nums) - k):
    current_sum = sum(nums[i:k+1])
    if current_sum > max_sum:
      max_sum = current_sum
  return max_sum


# print(maximum_sum([-1,2,3,1,-3,2], 2))


# ========================= END OF QUESTION 1 ============================

# =========================== START OF QUESTION 3 ==========================
class LongestSubString:
  def lengthOfLongestSubstring(self, s: str) -> int:
    charSet = set()
    left = 0
    result = 0
    for right in range(len(s)):
      while s[right] in charSet:
        charSet.remove(s[left])
        left += 1
      charSet.add(s[right])
      result = max(result, right - left + 1)
    return result
  
# new_solution = LongestSubString()
# print(new_solution.lengthOfLongestSubstring('abcabcbb'))
# print(new_solution.lengthOfLongestSubstring('bbbbb'))
# print(new_solution.lengthOfLongestSubstring('pwwkew'))

# =========================== END OF QUESTION 3 ============================

# =========================== START OF QUESTION 76 ============================
class MinimumWindowSubstringSolution:
  def minimumWindow(self, s: str, t:str) -> str:
    if t == '':
      return ''
    countT, window = {}, {}
    for c in t:
      countT[c] = 1+ countT.get(c, 0)
    have, need = 0, len(countT)
    result, resultLength = [-1,-1], float('infinity')
    left = 0

    for right in range(len(s)):
      c = s[right]
      window[c] = 1 + window.get(c, 0)

      if c in countT and window[c] == countT[c]:
        have += 1
      while have == need:
        # update our result
        if (right - left + 1) < resultLength:
          result = [left, right]
          resultLength = right - left + 1
        # pop from the left of our window to try and minimize our window
        window[s[left]] -= 1
        if s[left] in countT and window[s[left]] < countT[s[left]]:
          have -= 1
        left += 1
    left, right = result
    return s[left:right+1] if resultLength != float('infinity') else ""
  

# new_solution = MinimumWindowSubstringSolution()
# print(new_solution.minimumWindow("ADOBECODEBANC", "ABC"))
# print(new_solution.minimumWindow("a", "aa"))

# =========================== END OF QUESTION 76 ============================


# =========================== START OF QUESTION 424 ============================
class LongestRepeatingCharacterReplacement:
  def characterReplacement(self, s: str, k: int) -> int:
    count = {}
    result = 0
    left = 0
    maxFrequency = 0
    for right in range(len(s)):
      count[s[right]] = 1 + count.get(s[right], 0)
      maxFrequency = max(maxFrequency, count[s[right]])

      while (right - left + 1) - maxFrequency > k:
        count[s[left]] -= 1
        left += 1
      result = max(result, right - left + 1)
    return result

# new_solution = LongestRepeatingCharacterReplacement()
# print(new_solution.characterReplacement('ABAB', 2))
# print(new_solution.characterReplacement('AABABBA', 1))

# =========================== END OF QUESTION 424 ==============================


# =========================== START OF QUESTION 438 ==============================
class StringAnagramsSolution:
  def findAnagrams(self, s: str, p: str) -> List[int]:
    if len(p) > len(s):
      return []
    pCount, sCount = {}, {}
    for i in range(len(p)):
      pCount[p[i]] = 1 + pCount.get(p[i], 0)
      sCount[s[i]] = 1 + sCount.get(s[i], 0)

    result = [0] if sCount == pCount else []
    left = 0
    for right in range(len(p), len(s)):
      sCount[s[right]] = 1 + sCount.get(s[right], 0)
      sCount[s[left]] -= 1

      if sCount[s[left]] == 0:
        sCount.pop(s[left])
      left += 1
      if sCount == pCount:
        result.append(left)

    return result
  
# new_solution = StringAnagramsSolution()
# print(new_solution.findAnagrams("cbaebabacd", "abc"))

# =========================== END OF QUESTION 438 ==============================


# =========================== START OF QUESTION 239 ==============================
class SlidingWindowMaximumSolution:
  def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    output = []
    left= right = 0
    queue = collections.deque() # contains indeces

    while right < len(nums):
      while queue and nums[queue[-1]] < nums[right]:
        queue.pop()
      queue.append(right)

      # remove left value from window
      if left > queue[0]:
        queue.popleft()
      if (right + 1) >= k:
        output.append(nums[queue[0]])
        left += 1
      right += 1

    return output

# new_solution = SlidingWindowMaximumSolution()
# print(new_solution.maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3))
# print(new_solution.maxSlidingWindow([1], 1))

# =========================== END OF QUESTION 239 ================================



# =========================== START OF QUESTION 1838 ================================
class MostFrequentElementSolution:
  def maxFrequency(self, nums: List[int], k: int) -> int:
    nums.sort()
    left = right = result = total = 0

    while right < len(nums):
      total += nums[right]
      while nums[right] * (right - left + 1) > total + k:
        total -= nums[left]
        left += 1
      result = max(result, right - left + 1)
      right += 1

    return result
  

# new_solution = MostFrequentElementSolution()
# print(new_solution.maxFrequency([1,2,4], 5))
# print(new_solution.maxFrequency([1,4,8,13], 5))



# =========================== END OF QUESTION 1838 ==================================


# =========================== END OF QUESTION 1888 ==================================
class MinimumBinaryStringFlipSolution:
  def minFlips(self, s: str) -> int:
    n = len(s)
    s = s + s
    alt1, alt2 = "", ""
    for i in range(len(s)):
      alt1 += "0" if i % 2 else "1"
      alt2 += "1" if i % 2 else "0"

    result = len(s)
    diff1, diff2 = 0, 0
    left = 0
    for right in range(len(s)):
      if s[right] != alt1[right]:
        diff1 += 1
      if s[right] != alt2[right]:
        diff2 += 1
      if (right - left + 1) > n:
        if s[left] != alt1[left]:
          diff1 -= 1
        if s[left] != alt2[left]:
          diff2 -= 1
        left += 1
      if right - left + 1 == n:
        result = min(result, diff1, diff2)

    return result
  
# new_solution = MinimumBinaryStringFlipSolution()
# print(new_solution.minFlips("111000"))


# =========================== END OF QUESTION 1888 ==================================


# =========================== START OF QUESTION 567 ==================================
class StringPermutationSolution:
  def checkInclusion(self, s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
      return False
    s1Count, s2Count = [0] * 26, [0] * 26
    for i in range(len(s1)):
      s1Count[ord(s1[i]) - ord('a')] += 1
      s2Count[ord(s2[i]) - ord('a')] += 1

    matches = 0
    for j in range(26):
      matches += 1 if s1Count[j] == s2Count[j] else 0
    left = 0
    for right in range(len(s1), len(s2)):
      if matches == 26:
        return True
      index = ord(s2[right]) - ord('a')
      s2Count[index] += 1
      if s1Count[index] == s2Count[index]:
        matches +=1
      elif s1Count[index] + 1 == s2Count[index]:
        matches -= 1

      index = ord(s2[left]) - ord('a')
      s2Count[index] -= 1
      if s1Count[index] == s2Count[index]:
        matches +=1
      elif s1Count[index] - 1 == s2Count[index]:
        matches -= 1
      left += 1

    return matches == 26



# new_solution = StringPermutationSolution()
# print(new_solution.checkInclusion("ab", "eidbaooo"))
# print(new_solution.checkInclusion("ab", "eidboaoo"))


# =========================== END OF QUESTION 567 ==================================

# =========================== START OF QUESTION 209 ==================================
# the solution below is not working
class MinimumSubarraySumSolution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
      left, total = 0, 0
      minimum_length = float('inf')
      for right in range(len(nums)):
        total += nums[right]
        while total >= target:
          minimum_length = min(minimum_length, right - left + 1)
          total -= nums[left]
          left += 1
      return 0 if minimum_length == float('inf') else minimum_length


# =========================== END OF QUESTION 209 ==================================


# =========================== START OF QUESTION 187 ==================================
class RepeatedDNASolution:
  def findRepeatedDnaSequences(self, s: str) -> List[str]:
    seen, result = set(), set()
    for left in range(len(s) - 9):
      current = s[left:left+10]
      if current in seen:
        result.add(current)
      seen.add(current)
    return list(result)


# =========================== END OF QUESTION 187 ==================================