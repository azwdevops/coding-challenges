import collections

import random
from typing import List, Optional, Counter

# =================== START OF LEETCODE 15 3SUM ====================================
# two pointer algorithm
class ThreeSumSolution:
  def threeSum(self, nums: List[int]) -> List[List[int]]:
    result = []
    nums.sort()
    length = len(nums)
    for i in range(length - 2):
      if i > 0 and nums[i] == nums[i - 1]:
        continue
      left = i + 1
      right = length - 1

      while left < right:
        total = nums[i] + nums[left] + nums[right]
        if total < 0:
          # to seek to move towards 0 we have to move left by 1 since numbers are sorted in ascending order
          left += 1
        elif total > 0:
          right -= 1
        else:
          result.append([nums[i], nums[left], nums[right]])
          while left < right and nums[left] == nums[left + 1]:
            left +=1
          while left < right and nums[right] == nums[right - 1]:
            right -= 1
          left += 1
          right -= 1
    return result

# =================== END OF LEETCODE 15 3SUM ====================================

# =================== START OF LEETCODE 3 LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS ====================================
# sliding window
class LongestSubStringWithoutRepeatingCharacterSolution:
  def lengthOfLongestSubstring(self, s: str) -> int:
    if len(s) == 0:
      return 0
    map= {}
    max_length = start = 0
    for i in range(len(s)):
      if s[i]in map and start <= map[s[i]]:
        start = map[s[i]] + 1
      else:
        max_length = max(max_length, i - start + 1)
      map[s[i]] = i
    return max_length

# =================== END OF LEETCODE 3 LONGEST SUBSTRING WITHOUT REPEATING CHARACTERS ====================================

# =================== START OF LEETCODE 560. Subarray Sum Equals K ====================================
# sliding window
class SubarraySumEqualKSolution:
  def subarraySum(self, nums: List[int], k: int) -> int:
    sumdict = {0:1}
    n = len(nums)
    count = 0
    s = 0
    for num in nums:
      s += num
      if s- k in sumdict:
        count += sumdict[s-k]
      if s in sumdict:
        sumdict[s] += 1
      else:
        sumdict[s] = 1
    return count


# =================== END OF LEETCODE 560. Subarray Sum Equals K ====================================


# =================== START OF LEETCODE 8. String to Integer (atoi) ====================================
class StringToIntegerSolution:
  def myAtoi(self, s: str) -> int:
    stripped = s.strip()
    if not stripped:
      return 0
    negative = False
    out = 0
    if stripped[0] == '-':
      negative = True
    elif stripped[0] == '+':
      negative = False
    elif not stripped[0].isnumeric():
      return 0
    else:
      out = ord(stripped[0]) - ord('0')
    for i in range(1, len(stripped)):
      if stripped[i].isnumeric():
        out = out * 10 + (ord(stripped[i]) - ord('0'))
        if not negative and out >= 2147483647:
          return 2147483647
        if negative and out >= 2147483648:
          return -2147483648
      else:
        break

    if not negative:
      return out
    else:
      return -out
    
# new_solution = StringToIntegerSolution()
# print(new_solution.myAtoi('-042'))
# print(new_solution.myAtoi('-1337c0d3'))
# print(new_solution.myAtoi('1337c0d3'))

# =================== END OF LEETCODE 8. String to Integer (atoi) ====================================


# =================== START OF LEETCODE 13 roman to integer ====================================
class RomanToIntegerSolution:
  def romanToInt(self, s: str) -> int:
    my_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    previous = 0
    for i in range(len(s)):
      current = my_dict[s[i]]
      if current > previous:
        total += current - 2 * previous
      else:
        total += current
      previous = current

    return total
  
# new_solution = RomanToIntegerSolution()
# print(new_solution.romanToInt('III'))
# print(new_solution.romanToInt('LVIII'))
# print(new_solution.romanToInt('MCMXCIV'))

# =================== END OF LEETCODE 13 roman to integer ====================================


# =================== START OF LEETCODE 937. Reorder Data in Log Files ====================================
class ReorderDataLogSolution:
  def reorderLogFiles(self, logs: List[str]) -> List[str]:
    letter_logs, digit_logs = [], []
    for log in logs:
      if (log.split()[1]).isdigit():
        digit_logs.append(log)
      else:
        letter_logs.append(log.split())

    letter_logs.sort(key = lambda x :x[0])
    letter_logs.sort(key = lambda x :x[1:])

    for i in range(len(letter_logs)):
      letter_logs[i] = ' '.join(letter_logs[i])

    return letter_logs + digit_logs
  
# new_solution = ReorderDataLogSolution()
# print(new_solution.reorderLogFiles(["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]))


# =================== END OF LEETCODE 937. Reorder Data in Log Files ====================================

# =================== START OF LEETCODE 53. Maximum Subarray ====================================
class MaximumSubarraySolution:
  def maxSubArray(self, nums: List[int]) -> int:
    total_sum = max_sum = nums[0]
    for item in nums[1:]:
      total_sum = max(item, total_sum + item)
      max_sum = max(max_sum, total_sum)

    return max_sum

# =================== END OF LEETCODE 53. Maximum Subarray ====================================

# =================== START OF LEETCODE 14. Longest Common Prefix ====================================

class LongestCommonPrefixSolution:
  def longestCommonPrefix(self, strs: List[str]) -> str:
    if len(strs) == 0:
      return ""
    min_length = len(strs[0])
    for i in range(len(strs)):
      min_length = min(min_length, len(strs[i]))

    common_prefix = ''
    i = 0

    while i < min_length:
      char = strs[0][i]
      for j in range(1, len(strs)):
        if strs[j][i] != char:
          return common_prefix
      common_prefix += char
      i +=1
    return common_prefix

# new_solution = LongestCommonPrefixSolution()
# print(new_solution.longestCommonPrefix(["flower","flow","flight"]))
# =================== END OF LEETCODE 14. Longest Common Prefix ====================================

# =========================== START OF QUESTION 16. 3Sum Closest ==================================
class ThreeSumClosestSolution:
  def threeSumClosest(self, nums:List[int], target: int) -> int:
    nums.sort()
    result = sum(nums[:3])
    for i in range(len(nums) - 2):
      start = i + 1
      end = len(nums) - 1
      while start < end:
        current_sum = nums[i] + nums[start] + nums[end]
        if abs(current_sum - target) < abs(result - target):
          result = current_sum
        if current_sum < target:
          start += 1
        else:
          end -= 1

    return result
  
# =========================== END OF QUESTION 16. 3Sum Closest ==================================


# =========================== START OF QUESTION 572. Subtree of Another Tree ==================================

class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

class IsSubTreeSolution:
  def isSameTree(self, root:Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    if root is None and subRoot is None:
      return True
    if subRoot is None:
      return True
    if root is None and subRoot is not None:
      return False
    return self.isSame(root, subRoot) or self.isSame(root.left, subRoot) or self.isSame(root.right, subRoot)


  def isSame(self, root, subRoot):
    if root is None and subRoot is None:
      return True
    if root is None or subRoot is None:
      return False
    return root.val == subRoot.val and self.isSame(root.left, subRoot.left) and self.isSame(root.right, subRoot.right)

# =========================== END OF QUESTION 572. Subtree of Another Tree ==================================

# =================== START OF LEETCODE 54. Spiral Matrix ====================================
class SpiralMatrixSolution:
  def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    if not matrix:
      return []
    row_begin = 0
    row_end = len(matrix)
    column_begin = 0
    column_end = len(matrix[0])
    result = []

    while row_end > row_begin and column_end > column_begin:
      for i in range(column_begin, column_end):
        result.append(matrix[row_begin][i])
      for j in range(row_begin + 1, row_end - 1):
        result.append(matrix[j][column_end - 1])
      
      if row_end != row_begin + 1:
        for i in range(column_end - 1, column_begin - 1, -1):
          result.append(matrix[row_end-1][i])
      if column_begin != column_end - 1:
        for j in range(row_end - 2, row_begin, -1):
          result.append(matrix[j][column_begin])
      row_begin += 1
      row_end -= 1
      column_begin += 1
      column_end -= 1
    return result

# new_solution = SpiralMatrixSolution()
# print(new_solution.spiralOrder([[1,2,3],[4,5,6],[7,8,9]]))
# print(new_solution.spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))

# =================== END OF LEETCODE 54. Spiral Matrix ======================================


# =================== START OF LEETCODE 202. Happy Number ======================================
class HappyNumberSolution:
  def isHappy(self, n: int) -> bool:
    def squareSum(num):
      result = 0
      while num > 0:
        remainder = num % 10
        result += remainder * remainder
        num = num // 10
      return result
    seen = set()

    while squareSum(n) not in seen:
      sum1 = squareSum(n)
      if sum1 == 1:
        return True
      else:
        seen.add(sum1)
        n = sum1
    return False


# =================== END OF LEETCODE 202. Happy Number ======================================


# =================== START OF LEETCODE 101. Symmetric Tree ======================================
class SymmetricTreeSolution:
  def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    if root is None:
      return True
    return self.isMirror(root.left, root.right)
  
  def isMirror(self, left, right):
    if left and right:
      return left.val == right.val and self.isMirror(left.left, right.right) and self.isMirror(left.right, right.left)
    return False
    


# =================== END OF LEETCODE 101. Symmetric Tree ========================================


# =================== START OF LEETCODE 5. Longest Palindromic Substring ========================================
class LongestPalidromicSubstringSolution:
  def longestPalidrome(self, s: str) -> str:
    result = ''
    for i in range(len(s)):
      odd = self.helper(s, i, i)
      even = self.helper(s, i, i+1)

      result = max(odd, even, result, key=len)

    return result

  def helper(self, s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
      left -= 1
      right -= 1
    return s[left + 1:right]



# =================== END OF LEETCODE 5. Longest Palindromic Substring ========================================


# =================== START OF LEETCODE 56. Merge Intervals ========================================
class MergeIntervalSolution:
  def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x:x[0])
    i = 1
    while i < len(intervals):
      if intervals[i][0] <= intervals[i-1][1]:
        intervals[i-1][0] = min(intervals[i-1][0], intervals[i][0])
        intervals[i-1][1] = max(intervals[i-1][1], intervals[i][1])

        intervals.pop(i)
      else:
        i = i + 1
    return intervals 


# =================== END OF LEETCODE 56. Merge Intervals ==========================================


# =================== START OF LEETCODE 49. Group Anagrams ========================================

class GroupAnagramSolution:
  def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    new_dict = {}
    for word in strs:
      sorted_word = ''.join(sorted(word))
      if sorted_word not in new_dict:
        new_dict[sorted_word] = [word]
      else:
        new_dict[sorted_word].append(word)
    result = []
    for item in new_dict.values():
      result.append(item)
    return result



# =================== END OF LEETCODE 49. Group Anagrams ========================================



# =================== START OF LEETCODE 1143. Longest Common Subsequence ========================================
class LongestCommonSubsequenceSolution:
  def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    n, m = len(text1), len(text2)
    dp = [[0] * (m+1) for z in range(n+1)]
    for i in range(n):
      for j in range(m):
        if text1[i] == text2[j]:
          dp[i+1][j+1] = dp[i][j] + 1
        else:
          dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]

# =================== END OF LEETCODE 1143. Longest Common Subsequence ========================================


# =================== END OF LEETCODE 146. LRU Cache ========================================

class DLL:
  def __init__(self, key, val):
    self.key = key
    self.val = val
    self.next = None
    self.prev = None

class LRUCache:

    def __init__(self, capacity: int):
      self.head = DLL(-1,-1)
      self.tail = self.head
      self.hash = {}
      self.capacity = capacity
      self.length = 0
        

    def get(self, key: int) -> int:
      if key not in self.hash:
        return -1
      node = self.hash[key]
      val = node.val
      while node.next:
        node.prev = node.next
        node.next.prev = node.prev
        self.tail.next = node
        node.prev = self.tail
        node.next = None
        self.tail = node
      return val

    def put(self, key: int, value: int) -> None:
      if key in self.hash:
        node = self.hash[key]
        node.val = value
        while node.next:
          node.prev = node.next
          node.next.prev = node.prev
          self.tail.next = node
          node.prev = self.tail
          node.next = None
          self.tail = node
      else:
        node = DLL(key, value)
        self.hash[hash] = node
        self.tail.next = node
        node.prev = self.tail
        self.tail = node
        self.length += 1
        if self.length > self.capacity:
          remove = self.head.next
          self.head.next = self.head.next.next
          self.head.next.prev = self.head
          del self.hash[remove.key]
          self.length -= 1

# =================== END OF LEETCODE 146. LRU Cache ========================================


# =================== START OF LEETCODE 1. Two Sum ========================================
class TwoSumSolution:
  def twoSum(self, nums: List[int], target: int) -> List[int]:
    if len(nums) <= 0:
      return False
    new_dict = {}
    for i in range(len(nums)):
      if nums[i] in new_dict:
        return [i, new_dict[nums[i]]]
      else:
        new_dict[target - nums[i]] = i


# =================== END OF LEETCODE 1. Two Sum ==========================================


# =========================== START OF QUESTION 438. Find All Anagrams in a String ==================================
class StringAnagramSolution:
  def findAnagrams(self, s: str, p: str) -> List[int]:
    s_length, p_length = len(s), len(p)
    if s_length < p_length:
      return []
    p_count = Counter(p)
    s_count = Counter()

    result = []
    for i in range(s_length):
      s_count[s[i]] += 1
      if i >= p_length:
        if s_count[s[i - p_length]] == 1:
          del s_count[s[i - p_length]]
        else:
          s_count[s[i - p_length]] -= 1
      if p_count == s_count:
        result.append(i - p_length + 1)
    return result
    

# =========================== END OF QUESTION 438. Find All Anagrams in a String ====================================


# =========================== START OF QUESTION 528. Random Pick with Weight ====================================
class RandomPickWithWeightSolution:
  def __init__(self, w: List[int]):
    self.weights = w
    prefix_sum = 0
    for weight in w:
      prefix_sum += weight
      self.prefix_sum.append(prefix_sum)
    self.total_sum = prefix_sum

  def pickIndex(self) -> int:
    random_num = self.total_sum * random.random()
    low, high = 0, len(self.prefix_sum)
    while low < high:
      mid = low + (high - low) // 2
      if random_num > self.prefix_sum[mid]:
        low = mid + 1
      else:
        high = mid
    return low

# =========================== END OF QUESTION 528. Random Pick with Weight ====================================


# =========================== START OF QUESTION 543. Diameter of Binary Tree ====================================
class DiameterBinaryTreeSolution:
  def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
    if root is None:
      return 0
    left_height = self.height(root.left)
    right_height = self.height(root.right)

    left_diameter = self.diameterOfBinaryTree(root.left)
    right_diameter = self.diameterOfBinaryTree(root.right)

    return max(left_height + right_height, max(left_diameter, right_diameter))
  

  def height(self, root):
    if root is None:
      return 0
    else:
      return 1 + max(self.height(root.left), self.height(root.right))


# =========================== END OF QUESTION 543. Diameter of Binary Tree ====================================


# =========================== START OF QUESTION 200. Number of Islands ====================================
class NumberIslandSolution:
  def numIslands(self, grid: List[List[str]]) -> int:
    if not grid:
      return 0
    count = 0
    for i in range(len(grid)):
      for j in range(len(grid[0])):
        if grid[i][j] == '1':
          self.dfs(grid, i, j)
          count += 1

    return count

  def dfs(self, grid, i, j):
    if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':
      return

    grid[i][j] = '0'
    self.dfs(grid, i + 1, j)
    self.dfs(grid, i - 1, j)
    self.dfs(grid, i, j + 1)
    self.dfs(grid, i, j - 1)

# =========================== END OF QUESTION 200. Number of Islands ====================================


# =========================== START OF QUESTION 490. The Maze ====================================
class TheMazeSolution:
  def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
    m,n,visited = len(maze), len(maze[0]), set()

    def dfs(x, y):
      if (x, y) not in visited:
        visited.add((x,y))
      else:
        return False
      if [x, y] == destination:
        return True
      for i, j in (0,-1), (0,1), (-1, 0), (1,0):
        new_X, new_Y = x, y
        while 0 <= new_X + i < m and 0 <= new_Y + j < n and maze[new_X + i][new_Y + j] != 1:
          new_X += i
          new_Y += j
        if dfs(new_X, new_Y):
          return True
      return False
    return(dfs(*start))  


# =========================== END OF QUESTION 490. The Maze ====================================

# =========================== START OF QUESTION 567. Permutation in String ====================================
class PermutationInStringSolution:
  def checkInclusion(self, s1: str, s2: str) -> bool:
    if len(s1) > len(s2):
      return False
    s1_length =len(s1)
    s1_counter = Counter(s1)
    window_counter = Counter()

    for index, value in enumerate(s2):
      window_counter[value] += 1
      if index >= s1_length:
        element_from_left = s2[index - s1_length]

        if window_counter[element_from_left] == 1:
          del window_counter[element_from_left]
        else: 
          window_counter[element_from_left] -= 1
      if window_counter == s1_counter:
        return True
      
    return False

# =========================== END OF QUESTION 567. Permutation in String ====================================


# =========================== START OF QUESTION 24. Swap Nodes in Pairs ====================================
class ListNode:
  def __init__(self, x):
    self.val = x
    self.next = None

class SwapNodePairSolution:
  def swapPairs(self, head: ListNode) -> ListNode:
    d1 = d = ListNode(0)
    d.next = head

    while d.next and d.next.next:
      p = d.next
      q = d.next.next
      d.next, p.next, q.next = q, q.next, p
      d = p
    return d1.next


# =========================== END OF QUESTION 24. Swap Nodes in Pairs ====================================


# =========================== START OF QUESTION 208. Implement Trie (Prefix Tree) ====================================
class TreeNode:
  def __init__(self, val):
    self.val = val
    self.children = {}
    self.endhere = False

class TrieSolution:
  def __init__(self):
    self.root = TreeNode(None)

  def insert(self, word):
    parent = self.root
    for i, char in enumerate(word):
      if char not in parent.children:
        parent.children[char] = TreeNode(char)
      parent = parent.children[char]
      if i == len(word) - 1:
        parent.endhere = True

  def search(self, word):
    parent = self.root
    for char in word:
      if char not in parent.children:
        return False
      parent = parent.children[char]
    return parent.endhere
  
  def startsWith(self, prefix):
    parent= self.root
    for char in prefix:
      if char not in parent.children:
        return False
      parent = parent.children[char]
    return True

# =========================== END OF QUESTION 208. Implement Trie (Prefix Tree) ====================================

# =========================== START OF QUESTION 918. Maximum Sum Circular Subarray ====================================
class MaximumSumCircularSubarraySolution:
  def maxSubarraySumCircular(self, A: List[int]) -> int:
    k = self.Kadane(A)
    cumulative_sum = 0
    for i in range(len(A)):
      cumulative_sum += A[i]
      A[i] = -A[i]
    cumulative_sum = cumulative_sum + self.Kadane(A)

    if cumulative_sum > k and cumulative_sum != 0:
      return cumulative_sum
    else:
      return k

  def Kadane(self, nums):
    current_sum, max_sum = nums[0], nums[0]
    for n in nums[1:]:
      current_sum = max(n, current_sum + n)
      max_sum = max(current_sum, max_sum)
    return max_sum

# =========================== END OF QUESTION 918. Maximum Sum Circular Subarray ====================================


# =================== START OF LEETCODE 322. Coin Change ========================================

class CoinChangeSolution:
  def coinChange(self, coins: List[int], amount: int) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
      for x in range(coin, amount + 1):
        dp[x] += dp[x - coin]
    return dp[amount]
  
# =================== END OF LEETCODE 322. Coin Change ========================================


# =================== START OF LEETCODE 349. Intersection of Two Arrays ========================================
class IntersectionOfTwoArraySolution:
  def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    i = 0
    j = 0
    result =[]
    nums1.sort()
    nums2.sort()
    while i < len(nums1) and j < len(nums2):
      if nums1[i] == nums2[j]:
        result.append(nums1[i])
        i += 1
        j += 1
      elif nums1[i] < nums2[j]:
        i += 1
      elif nums1[i] > nums2[j]:
        j += 1
    return result 

# =================== END OF LEETCODE 349. Intersection of Two Arrays ========================================


# =================== START OF LEETCODE 1277. Count Square Submatrices with All Ones ========================================
class CountSquareSubmatriceSolution:
  def countSquares(self, matrix: List[List[int]]) -> int:
    n = len(matrix)
    m = len(matrix[0])
    answer_matrix = [[0] * (m + 1) for _ in range(n + 1)]
    count = 0
    for row in range(1, n + 1):
      for col in range(1, m + 1):
        if matrix[row - 1][col - 1] == 1:
          answer_matrix[row][col] = 1 + min(answer_matrix[row][col - 1], answer_matrix[row-1][col], answer_matrix[row - 1][col - 1])
          count += answer_matrix[row][col]
    return count
# =================== END OF LEETCODE 1277. Count Square Submatrices with All Ones ========================================


# =================== START OF LEETCODE 23. Merge k Sorted Lists ========================================
class MergeSortedListSolution:
  def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

    # solution 1 Divide and Conquer
    if not lists:
      return
    if len(lists) == 1:
      return lists[0]
    mid = len(lists) // 2
    l = self.mergeKLists(lists[:mid])
    r = self.mergeKLists(lists[mid:])
    return self.merge(l, r)
  

    # solution 2 Sorting and Merging
    # self.node = []
    # dummy = head = ListNode(0)
    # for l in lists:
    #   while l:
    #     self.node.append(l.val)
    #     l = l.next
    # for x in sorted(self.node):
    #   dummy.next = ListNode(x)
    #   dummy = dummy.next
    # return head.next
  
  def merge(self, l1, l2):
    current = dummy = ListNode(0)
    while l1 and l2:
      if l1.val <= l2.val:
        current.next == ListNode(l1.val)
        current = current.next
        l1 = l1.next
      else:
        current.next = ListNode(l2.val)
        current = current.next
        l2 = l2.next
    if l1:
      current.next = l1
    else:
      current.next = l2
    return dummy


# =================== END OF LEETCODE 23. Merge k Sorted Lists ========================================


# =================== START OF LEETCODE 33. Search in Rotated Sorted Array ========================================
# algorithm used Binary Search
class SearchRotatedSortedArray:
  def search(self, nums:List[int], target: int) -> int:
    if not nums:
      return -1
    low, high = 0, len(nums) - 1
    while low <= high:
      mid = (low + high) // 2
      if target == nums[mid]:
        return mid
      if nums[low] <= nums[mid]:
        if nums[low] <= target <= nums[mid]:
          high = mid - 1
        else:
          low = mid + 1
      else:
        if nums[mid] <= target <= nums[high]:
          low = mid + 1
        else:
          high = mid - 1
    return -1 

# =================== END OF LEETCODE 33. Search in Rotated Sorted Array ========================================


# =================== START OF LEETCODE 48. Rotate Image ========================================
class RotateImageSolution:
  def rotate(self, matrix: List[List[int]]) -> None:
    n = len(matrix[0])
    for row in range(n):
      for col in range(row, n):
        matrix[col][row], matrix[row][col] = matrix[row][col], matrix[col][row]
    
    for i in range(n):
      matrix[i].reverse()

# =================== END OF LEETCODE 48. Rotate Image ========================================


# =================== START OF LEETCODE 2. Add Two Numbers ========================================
# algorithm Linked List Traversal and Arithmetic Operations.
class AddTwoNumberSolution:
  def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    resultList = current = ListNode(0)
    carry = 0
    while l1 or l2 or carry:
      if l1:
        carry += l1.val
        l1 = l1.next
      if l2:
        carry += l2.val
        l2 = l2.next
      current.next = ListNode(carry % 10)
      current = current.next
      carry = carry // 10
    return resultList.next
# =================== END OF LEETCODE 2. Add Two Numbers ========================================


# =================== START OF LEETCODE 525. Contiguous Array ========================================
# algorithm Hash Map (Dictionary) with Prefix Sum
class ContiguousArraySolution:
  def findMaxLength(self, nums: List[int]) -> int:
    my_dict = {}
    subarr, count = 0, 0
    for i in range(len(nums)):
      if nums[i] == 1:
        count += 1
      else:
        count += -1
      if count == 0:
        subarr = i + 1
      if count in my_dict:
        subarr = max(subarr, i - my_dict[count])
      else:
        my_dict[count] = i
    return subarr
# =================== END OF LEETCODE 525. Contiguous Array ========================================


# =================== START OF LEETCODE 678. Valid Parenthesis String ========================================
#  algorithm Greedy Algorithm with Two-Pass Traversal
class ValidParenthesisStringSolution:
  def checkValidString(self, s: str) -> bool:
    if len(s) == 0 or s == '*':
      return True
    if len(s) == 1:
      return False
    leftBalance = 0
    for i in s:
      if i == ')':
        leftBalance -= 1
      else:
        leftBalance += 1
      
      if leftBalance < 0:
        return False
    if leftBalance == 0:
      return True
    rightBalance = 0
    for i in reversed(s):
      if i == '(':
        rightBalance -= 1
      else:
        rightBalance += 1
      if rightBalance < 0:
        return False
    return True
  
    # alternative solution using stack
    stack, star_stack = [], []
    for index, char in enumerate(s):
      if char == '(':
        stack.append(index)
      elif char == '*':
        star_stack.append(index)
      elif char == ')':
        if len(stack) > 0:
          stack.pop()
        elif len(star_stack) > 0:
          star_stack.pop()
        else:
          return False
    while stack and star_stack:
      if stack[-1] < star_stack[-1]:
        stack.pop()
        star_stack.pop()
      else:
        break
    return len(stack) == 0
# =================== END OF LEETCODE 678. Valid Parenthesis String ========================================


# =================== START OF LEETCODE 234. Palindrome Linked List ========================================
# algorithm Two-Pointer Technique with Stack
class PalindromeLinkedListSolution:
  def isPalidrome(self, head: Optional[ListNode]) -> bool:
    if head is None:
      return True
    slow, fast = head, head
    stack = []
    while fast and fast.next:
      stack.append(slow.val)
      slow = slow.next
      fast= fast.next.next
    if fast:
      slow = slow.next
    while (slow and len(stack)):
      if stack.pop() != slow.val:
        return False
      slow = slow.next
    return True
# =================== END OF LEETCODE 234. Palindrome Linked List ========================================


# =================== START OF LEETCODE 124. Binary Tree Maximum Path Sum ========================================
# ref video https://www.youtube.com/watch?v=_wUz0XKQ5JM&list=PLYAlGR1wWgUUyYZ3wX2GdnhiL-QVhAXfR&index=40&ab_channel=thecodingworld
# algorithm
class BinaryTreeMaximumPathSumSolution:
  def maxPathSum(self, root: Optional[TreeNode]) -> int:
    self.maximum = float('-inf')
    def dfs(root):
      if root is None:
        return 0
      left_max = max(0, dfs(root.left))
      right_max = max(0, dfs(root.right))
      self.maximum = max(self.maximum, left_max + right_max + root.val)
      return max(left_max, right_max) + root.val
    dfs(root)
    return self.maximum

# =================== END OF LEETCODE 124. Binary Tree Maximum Path Sum ========================================

# =================== START OF LEETCODE 19. Remove Nth Node From End of List ========================================
class RemoveNthNodeFromEndofListSolution:
  def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    fast = slow = head
    for i in range(n):
      fast = fast.next
    if fast is None:
      return head.next
    while fast.next:
      fast = fast.next
      slow = slow.next
    slow.next = slow.next.next
    return head

# =================== END OF LEETCODE 19. Remove Nth Node From End of List ========================================


# =================== START OF LEETCODE 238. Product of Array Except Self ========================================
class ProductofArrayExceptSelfSolution:
  def productExceptSelf(self, nums: List[int]) -> List[int]:

    # product from left to right
    left = [1] * len(nums)
    for i in range(1, len(nums)):
      left[i] = left[i - 1] * nums[i - 1]

    # product from right to left
    right = [1] * len(nums)
    for i in range(len(nums) - 2, -1, -1):
      right[i] = right[i + 1] * nums[i + 1]

    result = [1] * len(nums)
    for i in range(len(nums)):
      result[i] = left[i] * right[i]
    
    return result



# =================== END OF LEETCODE 238. Product of Array Except Self ========================================


# =================== START OF LEETCODE 206. Reverse Linked List ========================================
class ReverseLinkedListSolution:
  def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    current = head
    next = None
    while current:
      next = current.next
      current.next = prev
      prev = current
      current = next

      head = prev

    return head


# =================== END OF LEETCODE 206. Reverse Linked List ========================================


# =================== START OF LEETCODE 278. First Bad Version ========================================
class FirstBadVersionSolution:
  def firstBadVersion(self, n: int) -> int:
    start = 1
    end = n
    while start < end:
      mid = (start + end) // 2
      # isBadVersion is from api in leetcode 
      if isBadVersion(mid):
        end = mid
      else:
        start = mid + 1
    return start

# =================== END OF LEETCODE 278. First Bad Version ========================================


# =================== START OF LEETCODE 387. First Unique Character in a String ========================================
class FirstUniqueCharacterInaString:
  def firstUniqChar(self, s: str) -> int:
    d = {}
    for i in range(len(s)):
      if s[i] not in d:
        d[s[i]] = 1
      else:
        d[s[i]] += 1
    for i in range(len(s)):
      if d[s[i]] == 1:
        return i
    return -1

# =================== END OF LEETCODE 387. First Unique Character in a String ========================================


# =================== START OF LEETCODE 207. Course Schedule ========================================
class CourseScheduleSolution:
  def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    in_degree = collections.defaultdict(set)
    out_degree = collections.defaultdict(set)

    for x, y in prerequisites:
      out_degree[y].add(x)
      in_degree[x].add(y)

    connection_removed = 0
    in_degree_zero = []

    for i in range(numCourses):
      if not in_degree[i]:
        in_degree_zero.append(i)
        connection_removed += 1
    while in_degree_zero:
      node = in_degree_zero.pop()
      for x in out_degree[node]:
        in_degree[x].remove(node)
        if not in_degree[x]:
          in_degree_zero.append(x)
          connection_removed += 1

    return connection_removed == numCourses
# =================== END OF LEETCODE 207. Course Schedule ========================================

# =================== START OF LEETCODE 1029. Two City Scheduling ========================================
class TwoCitySchedulingSolution:
  def twoCitySchedCost(self, costs: List[List[int]]) -> int:
    sorted_cost = sorted(costs, key = lambda x:x[0] - x[1])
    result = 0
    for i in range(len(costs)):
      if i < len(costs) / 2:
        result += sorted_cost[i][0]
      else:
        result += sorted_cost[i][1]
    return result
# =================== END OF LEETCODE 1029. Two City Scheduling ========================================



# =================== START OF LEETCODE 103. Binary Tree Zigzag Level Order Traversal ========================================
class BinaryTreeZigzagLevelOrderTraversalSolution:
  def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if root is None:
      return None
    result = []
    level = 0

    self.zigzag_helper(root, level, result)

    return result
  
  def zigzag_helper(self, root, level, result):
    if root is None:
      return None
    if len(result) < level + 1:
      result.append([])
    if level % 2 == 1:
      result[level].append(root.val)
    else:
      result[level].insert(0, root.val)
    self.zigzag_helper(root.right, level + 1, result)
    self.zigzag_helper(root.left, level + 1, result)

# =================== END OF LEETCODE 103. Binary Tree Zigzag Level Order Traversal ========================================


# =================== END OF LEETCODE 121. Best Time to Buy and Sell Stock ========================================
class BestTimeToBuyAndSellStockSolution:
  def maxProfit(self, prices: List[int]) -> int:
    if not prices:
      return 0
    answer = 0
    minimum = prices[0]
    for i in range(1, len(prices)):
      if prices[i] < minimum:
        minimum = prices[i]
      else:
        answer = max(answer, prices[i] - minimum)
    return answer

# =================== END OF LEETCODE 121. Best Time to Buy and Sell Stock ========================================

# =================== START OF LEETCODE 1429. First Unique Number ========================================
class FirstUniqueNumberSolution:
  def __init__(self, nums: List[int]):
    self.q = []
    self.dict = {}
    for i in nums:
      self.add(i)
  
  def showFirstUnique(self) -> int:
    while len(self.q) > 0 and self.dict[self.q[0]] > 1:
      self.q.pop(0)
    if len(self.q) == 0:
      return -1
    else:
      return self.q[0]

  def add(self, value: int) -> None:
    if value in self.dict:
      self.dict[value] += 1
    else:
      self.dict[value] = 1
      self.q.append(value)
# =================== END OF LEETCODE 1429. First Unique Number ========================================


# =================== START OF LEETCODE 1008. Construct Binary Search Tree from Preorder Traversal ========================================
class ConstructBinarySearchTreefromPreorderTraversalSolution:
  def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
    in_order = sorted(preorder)
    return self.bstFromPreorderAndInorder(preorder, in_order)
  
  def bstFromPreorderAndInorder(self, preorder, in_order):
    length_pre = len(preorder)
    length_in = len(in_order)

    if length_pre != length_in or preorder == None or in_order == None or length_pre == 0:
      return None
    root = TreeNode(preorder[0])
    root_index = in_order.index(root.val)

    root.left = self.bstFromPreorderAndInorder(preorder[1:root_index + 1], in_order[:root_index])

    root.right = self.bstFromPreorderAndInorder(preorder[root_index + 1:], in_order[root_index + 1:])

    return root
# =================== END OF LEETCODE 1008. Construct Binary Search Tree from Preorder Traversal ========================================



# leetcode 402
class Solution402:
  def removeKdigits(self, num: str, k: int) -> str:
    stack = []
    number_to_remove = k
    for char in num:
      while k and stack and stack[-1] > char:
        stack.pop()
        number_to_remove -= 1
      stack.append(char)
    answer = ''.join(stack[0:len(num) - k]).lstrip('0')
    if len(answer):
      return answer
    else:
      return '0'

# solution = Solution402()
# print(solution.removeKdigits('1432219', 3))
# print(solution.removeKdigits('10200', 1))
# print(solution.removeKdigits('10', 2))
# print(solution.removeKdigits('1173', 2))


# leetcode 993
class Solution993:
  def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
    x_info = []
    y_info = []
    depth = 0
    parent = None
    if root is None:
      return False
    self.dfs(root, x, y, depth, parent, x_info, y_info)
    return x_info[0][0] == y_info[0][0] and x_info[0][1] != y_info[0][1]
  
  def dfs(self, root, x, y, depth, parent, x_info, y_info):
    if root is None:
      return None
    if root.val == x:
      x_info.append((depth, parent))
    if root.val == y:
      y_info.append((depth, parent))

    self.dfs(root.left, x, y, depth + 1, root, x_info, y_info)
    self.dfs(root.right, x, y, depth + 1, root, x_info, y_info)


# leetcode 