from typing import List, Optional
from collections import defaultdict, Counter


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# leetcode 724
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        left_sum = 0
        right_sum = sum(nums)

        for i in range(len(nums)):
            right_sum -= nums[i]
            if left_sum == right_sum:
                return i
            left_sum += nums[i]
        return -1


# leetcode 1768
class Solution1768:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        i = j = 0
        merged = ""
        len_w1 = len(word1)
        len_w2 = len(word2)
        while i < len_w1 and j < len_w2:
            merged += word1[i] + word2[j]
            i += 1
            j += 1
        if i < len_w1:
            merged += word1[i:]
        else:
            merged += word2[j:]
        return merged


# leetcode 230
class Solution230:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        # using recursive approach
        self.k = k
        self.result = None

        def inOrder(root):
            if root is None or self.result is not None:
                return
            inOrder(root.left)
            self.k -= 1
            if self.k == 0:
                self.result = root.val
                return
            inOrder(root.right)

        inOrder(root)

        return self.result

        # using iterative approach
        stack = []
        result = None
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right


# leetcode 872
class Solution872:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        leaf1 = []
        leaf2 = []

        def getLeaf(root, result):
            if root.left is None and root.right is None:
                result.append(root.val)
                return
            if root.left:
                getLeaf(root.left, result)
            if root.right:
                getLeaf(root.right, result)
            return

        getLeaf(root1, leaf1)
        getLeaf(root2, leaf2)

        return leaf1 == leaf2


# leetcode 377
class Solution377:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(target + 1):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]
        return dp[-1]


# leetcode 216
class Solution216:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        result = []

        def comb(current, k, n, start):
            if k == 0 and n == 0:
                result.append(current)
                return
            if k <= 0:
                return
            for i in range(start, min(10, n + 1)):
                comb(current + [i], k - 1, n - i, i + 1)
            return

        comb([], k, n, 1)
        return result


# leetcode 40
class Solution40:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        result = []

        def comb(current, start, target):
            if target == 0:
                result.append(current)
                return
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                if candidates[i] > target:
                    break
                comb(current + candidates[i], i + 1, target - candidates[i])
            return

        comb([], 0, target)
        return result


# leetcode 39
class Solution39:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        result = []

        def comb(current, start, target):
            if target == 0:
                result.append(current)
                return
            for i in range(start, len(candidates)):
                if candidates[i] > target:
                    break
                comb(current + [candidates[i]], i, target - candidates[i])

            return

        comb([], 0, target)
        return result

        # dynamic programming approach
        result = [[] for _ in range(target + 1)]
        result[0] = [[]]
        for cand in candidates:
            for i in range(cand, target + 1):
                for comb in result[i - cand]:
                    result[i].append(comb + [cand])
        return result[target]


# leetcode 283
class Solution283:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        insert = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[insert], nums[i] = nums[i], nums[insert]
                insert += 1


# leetcode 700
class Solution700:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # recursive approach
        if root is None:
            return None
        if root.val == val:
            return root
        if val < root.val:
            return self.searchBST(root.val, val)
        else:
            return self.searchBST(root.right, val)

        # iterative approach
        current = root
        while current and current.val != val:
            if val < current.val:
                current = current.left
            else:
                current = current.right
        return current


# leetcode 394
class Solution394:
    def decodeString(self, s: str) -> str:
        stack = []
        for char in s:
            if char != "]":
                stack.append(char)
            else:
                current = ""
                while stack and stack[-1] != "[":
                    current += stack.pop()
                stack.pop()  # remove the [ bracket
                num = ""
                while stack and stack[-1].isdigit():
                    num += stack.pop()
                stack.append(current * int(num))
        return "".join(stack)


# leetcode 417
class Solution417:

    def check(self, row, col, heights, ocean):
        ocean[row][col] = True
        my_list = [[row + 1, col], [row - 1, col], [row, col - 1], [row, col + 1]]
        for r, c in my_list:
            if 0 <= r < len(heights) and 0 <= c < len(heights[0]) and not ocean[r][c] and heights[r][c] >= heights[row][col]:
                self.check(r, c, heights, ocean)
        return

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        result = []
        rows, cols = len(heights), len(heights[0])
        pacific = [[False for i in range(cols)] for j in range(rows)]
        atlantic = [[False for i in range(cols)] for j in range(rows)]

        for i in range(rows):
            self.check(i, 0, heights, pacific)
            self.check(i, cols - 1, heights, atlantic)

        for i in range(cols):
            self.check(0, i, heights, pacific)
            self.check(rows - 1, i, heights, atlantic)

        for r in range(rows):
            for c in range(cols):
                if atlantic[r][c] and pacific[r][c]:
                    result.append([r, c])
        return result


# leetcode 125
class Solution125:
    def isPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1
        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True


# leetcode 242
class Solution242:
    def isAnagram(self, s: str, t: str) -> bool:
        # solution 1
        return Counter(s) == Counter(t)

        # solution 2
        s_counts = Counter(s)
        for char in t:
            if char not in s_counts:
                return False
            s_counts[char] -= 1
            if s_counts[char] == 0:
                del s_counts[char]
        return len(s_counts) == 0


# leetcode 49
class Solution49:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        my_dict = defaultdict(list)
        for word in strs:
            my_list = [0] * 26
            for char in word:
                my_list[ord(char) - ord("a")] += 1
            my_tuple = tuple(my_list)  # since dict keys must be immutable, we change the list(mutable) to a tuple(immutable)
            my_dict[my_tuple].append(word)
        return my_dict.values()


# leetcode 3
class Solution3:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start = 0
        end = 0
        max_len = 0
        my_dict = {}
        while end < len(s):
            if s[end] in my_dict and my_dict[s[end]] >= start:
                start = my_dict[s[end]] + 1
            max_len = max(max_len, end - start + 1)
            my_dict[s[end]] = end
            end += 1
        return max_len


# leetcode 159
class Solution159:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        start = 0
        end = 0
        max_len = 0
        my_dict = {}
        while end < len(s):
            my_dict[s[end]] = end
            if len(my_dict) > 2:
                min_index = min(my_dict.values())
                start = min_index + 1
                del my_dict[s[min_index]]
            max_len = max(max_len, end - start + 1)
            end += 1
        return max_len


# leetcode 5
class Solution5:

    def isPalidrome(self, string, left, right):
        while left > 0 and right < len(string) - 1 and string[left - 1] == string[right + 1]:
            left -= 1
            right += 1
        return left, right, right - left + 1

    def longestPalindrome(self, s: str) -> str:
        max_len = 0
        left = 0
        right = 0
        for index in range(len(s) - 1):
            # middle is a char
            l1, r1, max_len1 = self.isPalidrome(s, index, index)
            if max_len1 > max_len:
                max_len = max_len1
                left, right = l1, r1
            # if middle between two chars
            if s[index] == s[index + 1]:
                l2, r2, max_len2 = self.isPalidrome(s, index, index + 1)
                if max_len2 > max_len:
                    max_len = max_len2
                    left, right = l2, r2

        return s[left : right + 1]


# leetcode 20
class Solution20:
    def isValid(self, s: str) -> bool:
        my_dict = {"(": ")", "{": "}", "[": "]"}
        stack = []
        for char in s:
            if char in my_dict:
                stack.append(char)
            else:
                if stack == [] or my_dict[stack.pop()] != char:
                    return False
        return stack == []


# leetcode 647
class Solution647:

    def palidromes(self, s, left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count

    def countSubstrings(self, s: str) -> int:
        counts = 0
        for i in range(len(s)):
            counts += self.palidromes(s, i, i)
            counts += self.palidromes(s, i, i + 1)

        return counts


# leetcode 76
class Solution76:
    def minWindow(self, s: str, t: str) -> str:
        my_dict = defaultdict(int)
        for char in t:
            my_dict[char] += 1
        formed, total = 0, len(my_dict)
        left = right = 0
        len_answer = float("inf")
        sub_left, sub_right = 0, 0
        while right < len(s):
            char = s[right]
            if char in my_dict:
                my_dict -= 1
                if my_dict[char] == 0:
                    formed += 1
            while left <= right and formed == total:
                current_len = right - left + 1
                if current_len < len_answer:
                    len_answer = current_len
                    sub_left, sub_right = left, right + 1
                char = s[left]
                if char in my_dict:
                    if my_dict[char] == 0:
                        formed -= 1
                    my_dict[char] += 1
                left += 1

            r += 1
        return "" if len_answer == float("inf") else s[sub_left:sub_right]


# leetcode 2375
class Solution2375:
    def smallestNumber(self, pattern: str) -> str:
        result = []
        stack = []
        n = len(pattern)

        for i in range(n + 1):
            stack.append(str(i + 1))

            if i == n or pattern[i] == "I":
                while stack:
                    result.append(stack.pop())
        return "".join(result)


# solution = Solution2375()
# print(solution.smallestNumber("IIIDIDDD"))
# print(solution.smallestNumber("DDD"))


# leetcode 1727
class Solution1727:
    def largestSubmatrix(self, matrix: List[List[int]]) -> int:
        m = len(matrix)
        n = len(matrix[0])

        # step 1 compute the heights array
        heights = [[0] * n for _ in range(m)]
        for j in range(n):
            for i in range(m):
                if matrix[i][j] == 1:
                    heights[i][j] = heights[i - 1][j] + 1 if i > 0 else 1

        # step 2 sort each row in descending order
        max_area = 0
        for i in range(m):
            heights[i].sort(reverse=True)

            # step 3 calculate the maximum area
            for j in range(n):
                max_area = max(max_area, heights[i][j] * (j + 1))

        return max_area


# leetcode 921
class Solution921:
    def minAddToMakeValid(self, s: str) -> int:
        opening_brackets = 0
        closing_brackets = 0
        for char in s:
            if char == "(":
                closing_brackets += 1
            else:
                closing_brackets -= 1
                if closing_brackets < 0:
                    opening_brackets += 1
        return opening_brackets + closing_brackets


# leetcode 2294
class Solution2294:
    def partitionArray(self, nums: List[int], k: int) -> int:
        nums.sort()
        count = 0
        i = 0
        while i < len(nums):
            count += 1
            start = nums[i]
            while i < len(nums) and nums[i] - start <= k:
                i += 1
        return count


# leetcode 1529
class Solution1529:
    def minFlips(self, target: str) -> int:
        operations = 0
        current_bit = "0"
        for char in target:
            if char != current_bit:
                operations += 1
                current_bit = char
        return operations


# leetcode 2486
class Solution2486:
    def appendCharacters(self, s: str, t: str) -> int:
        left, right = 0, 0
        while left < len(s) and right < len(t):
            if s[left] == t[right]:
                right += 1
            left += 1

        return len(t) - right


# leetcode 1963
class Solution1963:
    def minSwaps(self, s: str) -> int:
        balance = 0
        min_balance = 0
        for char in s:
            if char == "[":
                balance += 1
            else:
                balance -= 1

        if min_balance < 0:
            return (-min_balance + 1) // 2
        return 0  # return 0 otherwise meaning string is balanced


# solution = Solution1963()
# print(solution.minSwaps("][]["))
# print(solution.minSwaps("]]][[["))


# leetcode 103
class Solution103:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        s1 = [root]
        s2 = []
        level = []
        result = []
        while s1 or s2:
            while s1:
                root = s1.pop()
                level.append(root.val)
                if root.left:
                    s2.append(root.left)
                if root.right:
                    s2.append(root.right)
            result.append(level)
            level = []
            while s2:
                root = s2.pop()
                level.append(root.val)
                if root.right:
                    s1.append(root.right)
                if root.left:
                    s1.append(root.left)
            if level != []:
                result.append(level)
                level = []
        return result


# leetcode 102
class Solution102:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        queue = [root]
        next_queue = []
        level = []
        result = []
        while queue != []:
            for root in queue:
                level.append(root.val)
                if root.left:
                    next_queue.append(root.left)
                if root.right:
                    next_queue.append(root.right)
            result.append(level)
            level = []
            queue = next_queue
            next_queue = []
        return result


# leetcode 144
class Solution144:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # solution 1 recursion
        if root is None:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

        # solution 2 iteratively
        if root is None:
            return []
        stack = [root]
        result = []
        while stack:
            root = stack.pop()
            result.append(root.val)
            if root.right:
                stack.append(root.right)
            if root.left:
                stack.append(root.left)
        return result


# leetcode 94
class Solution94:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # solution 1 using recursion
        if root is None:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

        # solution 2 using iterative solution
        stack = []
        result = []
        while root or stack != []:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            result.append(root.val)
            root = root.right
        return result


# leetcode 100
class Solution100:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p is None and q is None:
            return True
        if (p and q and p.val != q.val) or (p is None or q is None):
            return False
        return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)


# leetcode 572
class Solution572:
    def sameTree(self, p, q):
        if p is None and q is None:
            return True
        if p is None or q is None or p.val != q.val:
            return False
        return self.sameTree(p.left, q.left) and self.sameTree(p.right, q.right)

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if subRoot is None:
            return True
        if root is None:
            return False
        if self.sameTree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
