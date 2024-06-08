from typing import List

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
  
new_solution = LongestSubString()
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
  

new_solution = MinimumWindowSubstringSolution()
# print(new_solution.minimumWindow("ADOBECODEBANC", "ABC"))
print(new_solution.minimumWindow("a", "aa"))


# =========================== END OF QUESTION 76 ============================