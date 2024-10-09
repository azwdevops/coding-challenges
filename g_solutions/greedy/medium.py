from typing import List
from collections import defaultdict, Counter
import urllib.request


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# leetcode 1689
class Solution16989:
    def minPartitions(self, n: str) -> int:
        # initialize max_digit to 0
        max_digit = 0
        # iterate through each character in the string n
        for digit in n:
            # convert character to an integer
            digit_value = int(digit)
            # update the max_digit if the current digit is greater
            max_digit = max(max_digit, digit_value)

        # return the maximum digit found, which is the answer
        return max_digit


# leetcode 807
class Solution807:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        n = len(grid)
        max_row = [0] * n
        max_col = [0] * n

        # compute the max values for each row and each column
        for i in range(n):
            for j in range(n):
                max_row[i] = max(max_row[i], grid[i][j])
                max_col[j] = max(max_col[j], grid[i][j])
        # Calculate the total max sum possible
        max_sum = 0
        for i in range(n):
            for j in range(n):
                max_sum += min(max_col[j], max_row[i]) - grid[i][j]

        return max_sum


# leetcode 1382
class Solution1382:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        # using recursion
        def inorderTraversal(node):
            if not node:
                return []
            return inorderTraversal(node.left) + [node.val] + inorderTraversal(node.right)

        def buildBalancedBST(nodes, start, end):
            if start > end:
                return None
            mid = (start + end) // 2
            root = TreeNode(nodes[mid])
            root.left = buildBalancedBST(nodes, start, mid - 1)
            root.right = buildBalancedBST(nodes, mid + 1, end)

            return root

        # step 1 get sorted node values via inorder traversal
        nodes = inorderTraversal(root)

        # step 2 build and return the balanced BST
        return buildBalancedBST(nodes, 0, len(nodes) - 1)

        # using iterative approach (stack)
        # step 1 perform iterative inorder traversal to get the sorted node values
        stack = []
        node = root
        nodes = []

        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            nodes.append(node.val)
            node = node.right

        # step 2 build a balanced BST using an iterative method
        if not nodes:
            return None

        stack = [(0, len(nodes) - 1)]
        root = None
        nodes_map = {}

        while stack:
            start, end = stack.pop()
            if start > end:
                continue
            mid = (start + end) // 2
            new_node = TreeNode(nodes[mid])

            if not root:
                root = new_node
            nodes_maps[mid] = new_node

            if start <= mid - 1:
                stack.append((start, mid - 1))
                nodes_map[mid].left = TreeNode(0)  # placeholder

            if mid + 1 <= end:
                stack.append((mid + 1, end))
                nodes_map[mid].right = TreeNode(0)  # placeholder

        return root


# leetcode 1561
class Solution1561:
    def maxCoins(self, piles: List[int]) -> int:
        # step 1 sort the piles
        piles.sort()

        # step 2 calculate the maximum coins you can collect
        maxCoins = 0
        n = len(piles) // 3
        for i in range(len(piles) - 2, n - 1, -2):
            maxCoins += piles[i]
        return maxCoins


# leetcode 1605
class Solution1605:
    def restoreMatrix(self, rowSum: List[int], colSum: List[int]) -> List[List[int]]:
        # step 1: initialize an empty matrix with zeroes
        matrix = [[0] * len(colSum) for _ in range(len(rowSum))]

        # step 2: fill the matrix using the greedy approach
        for i in range(len(rowSum)):
            for j in range(len(colSum)):
                # fill the current cell with the minimum of rowSum[i] and colSum[j]
                value = min(rowSum[i], colSum[j])
                matrix[i][j] = value

                # subtract the filled value from rowSum[i] and colSum[j]
                rowSum[i] -= value
                colSum[j] -= value

        return matrix


# leetcode 1877
class Solution1877:
    def minPairSum(self, nums: List[int]) -> int:
        # step 1: sort the nums array
        nums.sort()
        # step 2 initialize the maximum pair sum
        maxPairSum = 0
        n = len(nums)

        # step 3 pair up the smallest with the largest and so on
        for i in range(n // 2):
            # calculate the current pair sum
            currentPairSum = nums[i] + nums[n - 1 - i]
            # update the maxPairSum with the maximum of the currentPairSum and maxPairSum
            maxPairSum = max(maxPairSum, currentPairSum)

        # step 4 return the minimized pair sum
        return maxPairSum


# leetcode 3016
class Solution3016:
    def minimumPushes(self, word: str) -> int:
        keypad = {str(x): 0 for x in range(2, 10)}
        key_position = {}
        current_min = 2
        minimum_pushes = 0
        word_chars = defaultdict(int)
        for char in word:
            word_chars[char] += 1
        word_chars_items = list(word_chars.items())
        word_chars_items.sort(key=lambda x: -x[1])

        for key, value in word_chars_items:
            if key in key_position:
                minimum_pushes += key_position[key] * value
            else:
                new_position = keypad[str(current_min)] + 1
                key_position[key] = new_position
                minimum_pushes += new_position * value
                keypad[str(current_min)] += 1
                if current_min == 9:
                    current_min = 2
                else:
                    current_min += 1

        return minimum_pushes


# solution = Solution3016()
# print(solution.minimumPushes("aabbccddeeffgghhiiiiii"))


# leetcode 861
class Solution861:
    def matrixScore(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        # step 1 toggle columns to maximize leading 1s

        for j in range(n):
            count_ones = sum(grid[i][j] for i in range(m))
            if count_ones <= m / 2:
                # toggle the column
                for i in range(m):
                    grid[i][j] = 1 - grid[i][j]

        # step 2 calculate the score
        score = 0
        for row in grid:
            # convert the row to its decimal value and add to score
            score += int("".join(map(str, row)), 2)

        return score


# leetcode 763
class Solution763:
    def partitionLabels(self, s: str) -> List[int]:
        # step 1 record the last occurence of each character
        last_occurence = {char: i for i, char in enumerate(s)}
        # step 2 initialize variables
        partitions = []
        start = 0
        end = 0
        # step 3 traverse the string and partition
        for i in range(len(s)):
            # extend the end of the current partition
            end = max(end, last_occurence[s[i]])

            # if we reach the end of the current partition
            if i == end:
                partitions.append(i - start + 1)
                start = i + 1

        return partitions


# leetcode 2966
class Solution2966:
    def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(0, len(nums), 3):
            triplet = nums[i : i + 3]
            if len(triplet) == 3 and max(triplet) - min(triplet) <= k:
                result.append(triplet)
            else:
                return []
        return result


# leetcode 1850
class Solution1850:
    def getMinSwaps(self, num: str, k: int) -> int:
        def nextPermutation(arr):
            i = len(arr) - 2
            while i >= 0 and arr[i] >= arr[i + 1]:
                i -= 1
            if i == -1:
                arr.reverse()
                return

            j = len(arr) - 1
            while arr[j] <= arr[i]:
                j -= 1

            arr[i], arr[j] = arr[j], arr[i]
            arr[i + 1 :] = reversed(arr[i + 1 :])

        def findKthPermutation(num, k):
            num_arr = list(num)
            for _ in range(k):
                nextPermutation(num_arr)
            return "".join(num_arr)

        def minSwapsToTransform(original, target):
            original = list(original)
            swaps = 0

            for i in range(len(original)):
                if original[i] != target[i]:
                    j = i

                    while original[j] != target[i]:
                        j += 1

                    while j > i:
                        original[j], original[j - 1] = original[j - 1], original[j]
                        swaps += 1
                        j -= 1

            return swaps

        kth_permutation = findKthPermutation(num, k)
        return minSwapsToTransform(num, kth_permutation)


# solution = Solution1850()
# print(solution.getMinSwaps("5489355142", 4))


# leetcode 969
class Solution969:
    def pancakeSort(self, arr: List[int]) -> List[int]:
        result = []
        n = len(arr)
        for size in range(n, 1, -1):
            # find the index of the maximum number within the first size elements
            max_index = arr.index(max(arr[:size]))
            # if the maximum number is not at its correct position, we need to flip
            if max_index != size - 1:
                # if the maximum number is not at the first position, flips it to the front
                if max_index != 0:
                    result.append(max_index + 1)
                    arr[: max_index + 1] = reversed(arr[: max_index + 1])

                # now flip it to its correct position
                result.append(size)
                arr[:size] = reversed(arr[:size])

        return result


# leetcode 1433
class Solution1433:
    def checkIfCanBreak(self, s1: str, s2: str) -> bool:
        # sort both strings
        sorted_s1 = sorted(s1)
        sorted_s2 = sorted(s2)
        # check if sorted_s1 can break sorted_s2
        s1_breaks_s2 = True
        s2_breaks_s1 = True
        for i in range(len(s1)):
            if sorted_s1[i] < sorted_s2[i]:
                s1_breaks_s2 = False
            if sorted_s2[i] < sorted_s1[i]:
                s2_breaks_s1 = False
        return s1_breaks_s2 or s2_breaks_s1


# solution = Solution1433()
# print(solution.checkIfCanBreak("leetcodee", "interview"))
# print(solution.checkIfCanBreak("abc", "xya"))
# print(solution.checkIfCanBreak("abe", "acd"))


# leetcode 2285
class Solution2285:
    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        # step 1: initialize a list to count connections for each city
        connection_counts = [0] * n
        # step 2: count the connections
        for road in roads:
            connection_counts[road[0]] += 1
            connection_counts[road[1]] += 1

        # step 3: sort cities by the number of connections (descending order)
        sorted_cities = sorted(range(n), key=lambda x: connection_counts[x], reverse=True)

        # step assign values to cities based on their sorted order
        city_values = [0] * n

        for i in range(n):
            city_values[sorted_cities[i]] = n - i

        # step 5 calculate the total importance
        total_importance = 0
        for road in roads:
            total_importance += city_values[road[0]] + city_values[road[1]]
        return total_importance


# leetcode 2895
class Solution2895:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # step 1: sort the tasks in descending order
        tasks.sort(reverse=True)
        # step 2 sort the processors by availability time
        processorTime.sort()

        # step 3 assign tasks to processors and calculate the completion time
        completion_times = []
        num_processors = len(processorTime)

        for i in range(num_processors):
            core_completion_time = processorTime[i]
            for j in range(4):
                core_completion_time += tasks[i * 4 + j]
            completion_times.append(core_completion_time)

        # step 4: return the maximum completion time across all processors
        return max(completion_times)


# leetcode 714
class Solution714:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)

        # initial values: holding a stock on day 0 vs not holding
        hold = -prices[0]  # profit if we buy the stock on day 0
        cash = 0  # profit if we don't buy the stock on day 0

        for i in range(1, n):
            # update hold and cash for day i
            hold = max(hold, cash - prices[i])
            cash = max(cash, hold + prices[i] - fee)

        return cash


# solution = Solution714()
# print(solution.maxProfit([1, 3, 2, 8, 4, 9], 2))


# leetcode 1338
class Solution1338:
    def minSetSize(self, arr: List[int]) -> int:
        # step 1 count the frequency of each element
        frequency = Counter(arr)
        # step 2 sort the frequencies in descending order
        sorted_frequencies = sorted(frequency.values(), reverse=True)

        # step 3 initialize variables
        num_removed = 0
        num_sets = 0
        half_size = len(arr) // 2

        # step 4: iterate through the sorted frequencies
        for freq in sorted_frequencies:
            num_removed += freq
            num_sets += 1
            if num_removed >= half_size:
                return num_sets


# leetcode 2358
class Solution2358:
    def maximumGroups(self, grades: List[int]) -> int:
        n = len(grades)
        k = 1
        while k * (k + 1) // 2 <= n:
            k += 1
        return k - 1


# solution = Solution2358()
# print(solution.maximumGroups([8, 8]))
# print(solution.maximumGroups([10, 6, 12, 7, 3, 5]))


# leetcode 1130
class Solution1130:
    def mctFromLeafValues(self, arr: List[int]) -> int:
        stack = []
        result = 0
        for value in arr:
            while stack and stack[-1] <= value:
                mid = stack.pop()
                if stack:
                    result += mid * min(stack[-1], value)
                else:
                    result += mid * value
            stack.append(value)

        # handle remaining elements in stack
        while len(stack) > 1:
            result += stack.pop() * stack[-1]

        return result


# leetcode 2279
class Solution2279:
    def maximumBags(self, capacity: List[int], rocks: List[int], additionalRocks: int) -> int:
        neededRocks = [capacity[i] - rocks[i] for i in range(len(capacity))]
        neededRocks.sort()

        fullBags = 0
        for rocksNeeded in neededRocks:
            if rocksNeeded <= additionalRocks:
                additionalRocks -= rocksNeeded
                fullBags += 1
            else:
                break

        return fullBags


# leetcode 122
class Solution122:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        max_profit = 0
        for i in range(1, n):
            profit = prices[i] - prices[i - 1]
            if profit > 0:
                max_profit += profit

        return max_profit


# leetcode 1753
class Solution1753:
    def maximumScore(self, a: int, b: int, c: int) -> int:
        stones = [a, b, c]
        score = 0
        # sort stones in descending order
        stones.sort(reverse=True)

        # while there are at least two non-empty piles
        while stones[1] > 0:
            # take one stone from the two largest piles
            stones[0] -= 1
            stones[1] -= 1

            # increment score
            score += 1

            # sort the piles again
            stones.sort(reverse=True)

        return score


# leetcode 1029
class Solution1029:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        # sort the differences in cost between city A and city B
        costs.sort(key=lambda x: x[0] - x[1])
        total_cost = 0
        n = len(costs) // 2

        # send the first n people to city A and the rest to city B
        for i in range(n):
            total_cost += costs[i][0]  # city A cost
            total_cost += costs[i + n][1]  # city B costs

        return total_cost


# leetcode 1663
class Solution1663:
    def getSmallestString(self, n: int, k: int) -> str:
        # step 1: initialize the string with all a's
        result = ["a"] * n
        # step 2: calculate the remaining value needed to reach k
        remaining_value = k - n
        # step 3: start from the end of the string and adjust characters
        index = n - 1
        while remaining_value > 0:
            # max value we can add at this position is 25 ('z' - 'a')
            add_value = min(25, remaining_value)
            result[index] = chr(ord("a") + add_value)

            # update the remaining value and move to the next character
            remaining_value -= add_value
            index -= 1

        # step 4 return the resulting string
        return "".join(result)


# leetcode 1899
class Solution1899:
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        # intialize flags for each element in target
        found_x = found_y = found_z = False
        x, y, z = target

        for a, b, c in triplets:
            # consider only valid triplets
            if a <= x and b <= y and c <= z:
                if a == x:
                    found_x = True
                if b == y:
                    found_y = True
                if c == z:
                    found_z = True
            # if all conditions are met, we can achieve the target
            if found_x and found_y and found_z:
                return True

        return False


# leetcode 2971
class Solution2971:
    def largestPerimeter(self, nums: List[int]) -> int:
        # step 1 sort the array in non-decreasing order
        nums.sort()
        # step 2 start from the end of the array and check for the triangle condition
        for i in range(len(nums) - 1, 1, -1):
            # Check if the sum of the smallest n-1 sides is greater than the largest side
            if sum(nums[:i]) > nums[i]:
                # If valid, return the perimeter
                return sum(nums[: i + 1])

        # step 4: if no valid triangle found, return -1
        return -1


# solution = Solution2971()
# print(solution.largestPerimeter([1, 12, 1, 2, 5, 50, 3]))


# leetcode 1846
class Solution1846:
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        # step 1 sort the array
        arr.sort()
        # step 2 set the first element to 1
        arr[0] = 1
        # step 3 adjust each element to be at most 1 more than the previous element
        for i in range(1, len(arr)):
            if arr[i] > arr[i - 1] + 1:
                arr[i] = arr[i - 1] + 1
        # step 4 the last element will be the maximum possible value
        return arr[-1]


# leetcode 1414
class Solution1414:
    def findMinFibonacciNumbers(self, k: int) -> int:
        # generate all fibonacci numbers less than or equal to k
        fibs = [1, 1]
        while fibs[-1] <= k:
            fibs.append(fibs[-1] + fibs[-2])
        # initialize the count of fibonacci numbers used

        count = 0
        i = len(fibs) - 2  # start from the largest fibonacci number <= k
        # subtract the largest possible fibonacci number from k
        while k > 0:
            if fibs[i] <= k:
                k -= fibs[i]
                count += 1
            i -= 1

        return count


# leetcode 3192
class Solution3192:
    def minOperations(self, nums: List[int]) -> int:
        operations = 0
        flipped = False  # initially we haven't flipped

        for num in nums:
            # if the current number doesn't match the expected value based on the flipped state
            if (num == 0 and not flipped) or (num == 1 and flipped):
                operations += 1
                flipped = not flipped  # toggle the flipped state

        return operations


# solution = Solution3192()
# print(solution.minOperations([0, 1, 1, 0, 1]))
# print(solution.minOperations([1,0,0,0]))


# leetcode 1247
class Solution1247:
    def minimumSwap(self, s1: str, s2: str) -> int:
        xy_mismatch = 0
        yx_mismatch = 0

        # count mismatches
        for i in range(len(s1)):
            if s1[i] == "x" and s2[i] == "y":
                xy_mismatch += 1
            elif s1[i] == "y" and s2[i] == "x":
                yx_mismatch += 1

        # check if the total mismatches is odd
        if (xy_mismatch + yx_mismatch) % 2 != 0:
            return -1
        # minimum swaps
        return (xy_mismatch // 2) + (yx_mismatch // 2) + (xy_mismatch % 2) * 2


# solution = Solution1247()
# print(solution.minimumSwap("xx", "yy"))
# print(solution.minimumSwap("xy", "yx"))
# print(solution.minimumSwap("xx", "xy"))


# leetcode 2517
class Solution2517:
    def maximumTastiness(self, price: List[int], k: int) -> int:
        price.sort()

        def canPickWithMinDifference(d):
            count = 1  # start with the first candy
            last_price = price[0]
            for i in range(1, len(price)):
                if price[i] - last_price >= d:
                    count += 1
                    last_price = price[i]
                if count >= k:
                    return True
            return False

        low, high = 0, price[-1] - price[0]

        while low <= high:
            mid = (low + high) // 2
            if canPickWithMinDifference(mid):
                low = mid + 1  # try for a larger minimum difference
            else:
                high = mid - 1  # try for a smaller minimum difference

        return high  #  there will be the maximum tastiness


solution = Solution2517()
print(solution.maximumTastiness([13, 5, 1, 8, 21, 2], 3))
