# leetcode 1763
class Solution1763:
  def longestNiceSubstring(self, s: str) -> str:
    if not s:
      return ''
    def is_nice(sub: str) -> bool:
      char_set = set(sub)
      for char in char_set:
        if char.swapcase() not in char_set:
          return False
      return True
    
    def longest_nice_helper(s: str) -> str:
      if is_nice(s):
        return s
      max_nice = ''
      for i in range(len(s)):
        if s[i].swapcase() not in s:
          left = longest_nice_helper(s[:i])
          right = longest_nice_helper(s[i+1:])
          max_nice = max(max_nice, right, left)
          break # no need to check further as we have already split the string
      return max_nice
    
    return longest_nice_helper(s)
  
# solution = Solution1763()
# print(solution.longestNiceSubstring('YazaAay'))
# print(solution.longestNiceSubstring('Bb'))
# print(solution.longestNiceSubstring('c'))


