from typing import List
from collections import defaultdict, Counter
from itertools import combinations, permutations

# leetcode 1365
class Solution1365:
  def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
    # create a frequency dict
    freq = defaultdict(int)
    for num in nums:
      freq[num] += 1
    # sort the unique elements of nums
    sorted_nums = sorted(freq.keys())
    # create a cumulative count dictionary
    cumulative_count = {}
    total = 0
    for num in sorted_nums:
      cumulative_count[num] = total
      total += freq[num]

    # construct the result array using the cumulative count dict
    result = []
    for num in nums:
      result.append(cumulative_count[num])
    
    return result
  

# solution = Solution1365()
# print(solution.smallerNumbersThanCurrent([8,1,2,2,3]))


# leetcode 1656
class OrderedStream1656:
  def __init__(self, n: int):
    # initialize the stream with None values and set the pointer to 0
    self.stream = [None] * n
    self.pointer = 0

  def insert(self, idKey: int, value: str) -> List[str]:
    # place the value in the correct position in the stream
    self.stream[idKey - 1] = value

    # initialize the chunk list to collect values in the correct order
    chunk = []
    # collect the values in the chunk starting from the current pointer
    while self.pointer < len(self.stream) and self.stream[self.pointer] is not None:
      chunk.append(self.stream[self.pointer])
      self.pointer += 1

    # return the chunk of values
    return chunk
  

# leetcode 2913
class Solution2913:
  def sumCounts(self, nums: List[int]) -> int:
    total = 0
    n = len(nums)
    left = 0
    while left < n:
      freq = set()
      distinct_count = 0
      for right in range(left, n):
        if not nums[right] in freq:
          freq.add(nums[right])
          distinct_count += 1
        total += distinct_count ** 2
      left += 1
    return total
  
# solution = Solution2913()
# print(solution.sumCounts([1,2,1]))


# leetcode 3184
class Solution3184:
  def countCompleteDayPairs(self, hours: List[int]) -> int:
    # initialize the dict to store the remainder counts
    remainder_count = defaultdict(int)
    # initialize the counter for complete day pairs
    complete_day_pairs = 0
    # iterate through each hour in the array
    for hour in hours:
      # calculate the remainder when divided by 24
      remainder = hour % 24
      # calculate the complement remainder that forms a multiple of 24
      complement = (24 - remainder) % 24
      # if the complement remainder exists in the dict, add its count to the counter
      if complement in remainder_count:
        complete_day_pairs += remainder_count[complement]
      # update the dictionary with the current remainder
      remainder_count[remainder] += 1

    return complete_day_pairs
  

# solution = Solution3184()
# print(solution.countCompleteDayPairs([12,12,30,24,24]))


# leetcode 2670
class Solution2670:
  def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
    n = len(nums)
    prefix_distinct = [0] * n
    suffix_distinct = [0] * n
    prefix_set = set()
    suffix_set = set()
    # calculate prefix distinct counts
    for i in range(n):
      prefix_set.add(nums[i])
      prefix_distinct[i] = len(prefix_set)

    # calculate suffix distinct counts
    for i in range(n - 1, -1, -1):
      suffix_distinct[i] = len(suffix_set)
      suffix_set.add(nums[i])

    # calculate distinct difference array
    result = [0] * n
    for i in range(n):
      result[i] = prefix_distinct[i] - suffix_distinct[i]

    return result
  
# solution = Solution2670()
# print(solution.distinctDifferenceArray([1,2,3,4,5]))
# print(solution.distinctDifferenceArray([1, 2, 1, 4, 2]))

# my_list = list('jdkjkafjahfkafa')
# my_list.sort()
# print(my_list.index('a'))

# leetcode 1370
class Solution1370:
  def sortString(self, s: str) -> str:
    # count the occurences of each character
    char_count = Counter(s)
    # get a sorted list of unique characters
    unique_chars = sorted(char_count.keys())
    new_str = ''
    while char_count:
      # pick smallest to largest
      for char in unique_chars:
        if char in char_count:
          new_str += char
          char_count[char] -= 1
          if char_count[char] == 0:
            del char_count[char]
      # pick the largest to smallest
      for char in reversed(unique_chars):
        if char in char_count:
          new_str += char
          char_count[char] -= 1
          if char_count[char] == 0:
            del char_count[char]
    return new_str

# solution = Solution1370()
# print(solution.sortString('aaaabbbbcccc'))


# leetcode 2399
class Solution2399:
  def checkDistances(self, s: str, distance: List[int]) -> bool:
    # initialize an array to store the first occurrence of each letter
    first_occurrence = [-1] * 26
    for index, char in enumerate(s):
      alpha_index = ord(char) - ord('a')
      if first_occurrence[alpha_index] == -1:
        # store the first occurrence of the character
        first_occurrence[alpha_index] = index
      else:
        # calculate the distance between the two occurrences
        actual_distance = index - first_occurrence[alpha_index] - 1
        if actual_distance != distance[alpha_index]:
          return False
        
    return True
  
# solution = Solution2399()
# print(solution.checkDistances("abaccb", [1,3,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))


# leetcode 2062
class Solution2062:
  def countVowelSubstrings(self, word: str) -> int:
    vowels = set('aeiou')
    count = 0
    n = len(word)
    for i in range(n):
      if word[i] in vowels:
        seen = set()
        for j in range(i, n):
          if word[j] in vowels:
            seen.add(word[j])
            if len(seen) == 5:
              count += 1
          else:
            break


# leetcode 888
class Solution888:
  def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
    # calculate total candies that Alice and Bob have
    sumA = sum(aliceSizes)
    sumB = sum(bobSizes)

    # calculate the difference delta
    delta = (sumA - sumB) // 2
    # create a set of Bob's candies for quick lookup
    setB = set(bobSizes)

    # find the pair (a, b)
    for a in aliceSizes:
      b = a - delta
      if b in setB:
        return [a, b]
      
# leetcode 1995
class Solution1995:
  def countQuadruplets(self, nums: List[int]) -> int:
    n = len(nums)
    count = 0
    sums_map = defaultdict(list)
    # iterate over possible values for d
    for d in range(n - 1, 2, -1):
      # update sums_map with pairs (nums[a], nums[b]) such that a < b < d
      for b in range(d - 1):
        for a in range(b):
          sums_map[nums[a] + nums[b]].append((a, b))

      # check if there's any pair (nums[a], nums[b], nums[c]) such that a < b < c and nums[a] + nums[b] + nums[c] == nums[d]
      for c in range(d - 1):
        for (a, b) in sums_map[nums[d] - nums[c]]:
          if a < b and b < c:
            count += 1

    return count


# leetcode 1763
class Solution1763:
  def longestNiceSubstring(self, s: str) -> str:
    def isNice(sub):
      char_set = set(sub)
      for c in char_set:
        if s.swapcase() not in char_set:
          return False
      return True
    
    n = len(s)
    if n < 2:
      return ''
    # check the whole string first
    if isNice(s):
      return s
    
    # try to divide the string at each character that is not nice
    for i in range(n):
      if s[i].swapcase() not in s:
        left = self.longestNiceSubstring(s[:i])
        right = self.longestNiceSubstring(s[i+1:])
        if len(left) >= len(right):
          return left
        else:
          return right
        
    return ''
  
  # using a stack
  def longestNiceSubstringUsingStack(self, s: str) -> str:
    def isNice(sub):
      char_set = set(sub)
      for char in char_set:
        if char.swapcase() not in char_set:
          return False
      return True
    
    stack = [(0, len(s))]
    longest_nice = ''

    while stack:
      start, end = stack.pop()
      if end - start < len(longest_nice):
        continue
      substring = s[start:end]
      if isNice(substring):
        if len(substring) > len(longest_nice):
          longest_nice = substring
      else:
        for i in range(start, end):
          if s[i].swapcase() not in substring:
            stack.append((start, i))
            stack.append(i + 1, end)
            break

    return longest_nice
  
# leetcode 2094
class Solution2094:
  def findEvenNumbers(self, digits: List[int]) -> List[int]:
    unique_integers = set()
    # generate all combinations of 3 digits
    for comb in combinations(digits, 3):
      for perm in permutations(comb):
        # form the integer from the permutation
        num = int(''.join(map(str, perm)))

        # check if the number is valid (no leading zero and even)
        if perm[0] != 0 and num % 2 == 0:
          unique_integers.add(num)

    # return the sorted list of unique integers
    return sorted(unique_integers)
  
# solution = Solution2094()
# print(solution.findEvenNumbers([2,1,3,0]))
# print(solution.findEvenNumbers([2,2,8,8,2]))


# leetcode 2549
class Solution2549:
  def distinctIntegers(self, n: int) -> int:
    def findDivisors(x):
      divisors = set()
      for i in range(1, int(x**0.5) + 1):
        if x % i == 0:
          divisors.add(i)
          divisors.add(x // i)
      return divisors
    
    n_minus_1 = n - 1
    divisors_of_n_minus_1 = findDivisors(n_minus_1)

    return len(divisors_of_n_minus_1) + 1

# solution = Solution2549()
# print(solution.distinctIntegers(5))
# print(solution.distinctIntegers(3))


# leetcode 953
class Solution953:
  def isAlienSorted(self, words: List[str], order: str) -> bool:
    # create a mapping of each character to its position in the order
    char_position = {char: index for index, char in enumerate(order)}
    def compare_words(word1, word2):
      # compare characters of the two words on the alien alphabet order
      for c1,c2 in zip(word1, word2):
        if char_position[c1] < char_position[c2]:
          return True
        elif char_position[c1] > char_position[c2]:
          return False
      # if we reach here, it means the words are equal up to the length of the shorter word
      return len(word1) <= len(word2)
    
    # iterate through each pair of consecutive words and compare them
    for i in range(len(words) - 1):
      if not compare_words(words[i], words[i + 1]):
        return False
      
    return True
  
# solution = Solution953()
# print(solution.isAlienSorted(["world", "word","row"], 'worldabcefghijkmnpqstuvxyz'))
# print(solution.isAlienSorted(["fxasxpc","dfbdrifhp","nwzgs","cmwqriv","ebulyfyve","miracx","sxckdwzv","dtijzluhts","wwbmnge","qmjwymmyox"], 'zkgwaverfimqxbnctdplsjyohu'))

# leetcode 1275
class Solution1275:
  def tictactoe(self, moves: List[List[int]]) -> str:
    # initialize a 3x3 grid
    grid = [['' for _ in range(3)] for _ in range(3)]
    # function to check if the current player has won
    def check_winner(player):
      # check rows and columns
      for row in range(3):
        if all(grid[row][col] == player for col in range(3)):
          return True
      for col in range(3):
        if all(grid[row][col] == player for row in range(3)):
          return True
      # check diagonals
      if grid[0][0] == grid[1][1] == grid[2][2] == player or grid[0][2] == grid[1][1] == grid[2][0] == player:
        return True
    
    # simulate the moves
    for index, [row, col] in enumerate(moves):
      player = 'X' if index % 2 == 0 else '0'
      grid[row][col] = player
      # check if the current player has won
      if check_winner(player):
        return 'A' if player == 'X' else 'B'
      
    # check if the game is a draw or pending
    if len(moves) == 9:
      return 'Draw'
    else:
      return 'Pending'
    