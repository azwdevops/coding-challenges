from typing import List


# =================== START OF LEETCODE 14. Longest Common Prefix ====================================

class LongestCommonPrefixSolution:
  def longestCommonPrefix(self, strs: List[str]) -> str:
    min_str = min(strs, key=lambda x : len(x))
    common_prefix = ''
    left = 0

    while left < len(min_str):
        char = min_str[left]
        for item in strs:
            if char != item[left]:
                return common_prefix
        common_prefix += char
        left += 1
    return common_prefix

# new_solution = LongestCommonPrefixSolution()
# print(new_solution.longestCommonPrefix(["flower","flow","flight"]))
# =================== END OF LEETCODE 14. Longest Common Prefix ====================================