from typing import List


class SolutionMaximumNonAdjacentSum:
    def maximumNonAdjacentSum(self, nums):
        n = len(nums)
        memo = {}

        def recursiveCall(currentIndex):
            if currentIndex in memo:
                return memo[currentIndex]
            if currentIndex == 0:
                return nums[0]
            if currentIndex < 0:
                return 0
            picked = nums[currentIndex] + recursiveCall(currentIndex - 2)
            notPicked = recursiveCall(currentIndex - 1)

            currentResult = max(picked, notPicked)
            memo[currentIndex] = currentResult

            return currentResult

        return recursiveCall(n - 1)


# solution = SolutionMaximumNonAdjacentSum()
# print(solution.maximumNonAdjacentSum([1, 2, 3, 1, 3, 5, 8, 1, 9]))


def maximumNonAdjacentSumTabulation(nums):
    n = len(nums)
    dp = [-1] * n
    dp[0] = nums[0]
    for i in range(1, n):
        pick = nums[i]
        if i > 1:
            pick += dp[i - 2]
        nonPick = dp[i - 1]
        dp[i] = max(pick, nonPick)

    return dp[n - 1]


# print(maximumNonAdjacentSumTabulation([1, 2, 3, 1, 3, 5, 8, 1, 9]))


def maximumNonAdjacentSumTabulationSpaceOptimized(nums):
    prev = 0
    prev2 = 0
    n = len(nums)

    for i in range(n):
        pick = nums[i]
        if i > 1:
            pick += prev2
        nonPick = prev

        current = max(pick, nonPick)
        prev2 = prev
        prev = current

    return prev


# print(maximumNonAdjacentSumTabulationSpaceOptimized([1, 2, 3, 1, 3, 5, 8, 1, 9]))


def houseRobber2(valueInHouse):
    # Write your function here.
    if len(valueInHouse) == 1:
        return valueInHouse[0]

    def helper(arr):
        n = len(arr)
        prev = 0
        prev2 = 0

        for i in range(n):
            pick = arr[i]
            if i > 1:
                pick += prev2
            nonPick = prev

            prev2 = prev
            prev = max(pick, nonPick)

        return prev

    return max(helper(valueInHouse[:-1]), helper(valueInHouse[1:]))


def ninjaTrainingMemoization(n: int, points: List[List[int]]) -> int:
    memo = {}

    def recursiveCall(day, prevTask):
        if (day, prevTask) in memo:
            return memo[(day, prevTask)]
        if day == 0:
            max_value = 0
            for task in range(3):
                if task != prevTask:
                    max_value = max(max_value, points[0][task])
            memo[(day, prevTask)] = max_value
            return max_value

        max_value = 0

        for task in range(3):
            if task != prevTask:
                currentPoints = points[day][task] + recursiveCall(day - 1, task)
                max_value = max(max_value, currentPoints)
        memo[(day, prevTask)] = max_value

        return max_value

    return recursiveCall(n - 1, 3)


# print(ninjaTrainingMemoization(3, [[1, 2, 5], [3, 1, 1], [3, 3, 3]]))


def ninjaTrainingTabulation(n: int, points: List[List[int]]) -> int:
    dp = [[0 for _ in range(4)] for _ in range(n)]
    # initialize DP table for day0 with base cases
    dp[0][0] = max(points[0][1], points[0][2])
    dp[0][1] = max(points[0][0], points[0][2])
    dp[0][2] = max(points[0][0], points[0][1])
    dp[0][3] = max(points[0][0], points[0][1], points[0][2])

    # loop through the days starting from the second day
    for day in range(1, n):
        for last in range(4):
            dp[day][last] = 0  # initialize the maximum points for the current day and last activity
            for task in range(3):
                if task != last:
                    # calculate the total points for the current day's activity and the previous day's maximum points
                    activity = points[day][task] + dp[day - 1][task]
                    dp[day][last] = max(dp[day][last], activity)

    return dp[n - 1][3]


# print(ninjaTrainingTabulation(3, [[1, 2, 5], [3, 1, 1], [3, 3, 3]]))
# print(ninjaTrainingTabulation(3, [[10, 40, 70], [20, 50, 80], [30, 60, 90]]))


def ninjaTrainingTabulationSpaceOptimized(n: int, points: List[List[int]]) -> int:
    # initialize a list prev to store the maximum points for each possible last activity on the previous day
    prev = [0] * 4
    # initialize prev with maximum points for the first day's activities
    prev[0] = max(points[0][1], points[0][2])
    prev[1] = max(points[0][0], points[0][2])
    prev[2] = max(points[0][0], points[0][1])
    prev[3] = max(points[0][0], points[0][1], points[0][2])

    # loop through the days starting from the second day
    for day in range(1, n):
        # initialize a temporary list temp to store the maximum points for each possible last activity on the current day
        temp = [0] * 4
        for last in range(4):
            # initialize temp for the current last activity
            temp[last] = 0
            for task in range(3):
                if task != last:
                    # calculate the total points for the current day's activities and the previous day's maximum points
                    activity = points[day][task] + prev[task]
                    # update temp with the maximum points for the current last activity
                    temp[last] = max(temp[last], activity)

        # update prev with temp for the next iteration
        prev = temp

    # return the maximum points achievable after the last day with any activity
    return prev[3]


# print(ninjaTrainingTabulationSpaceOptimized(3, [[1, 2, 5], [3, 1, 1], [3, 3, 3]]))
# print(ninjaTrainingTabulationSpaceOptimized(3, [[10, 40, 70], [20, 50, 80], [30, 60, 90]]))


def uniquePaths(m, n):
    # Write your code here.
    dp = [[0 for _ in range(m)] for _ in range(n)]
    dp[0][0] = 1
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            ways = 0
            if i > 0:
                ways += dp[i - 1][j]
            if j > 0:
                ways += dp[i][j - 1]
            dp[i][j] = ways
    return dp[n - 1][m - 1]


# print(uniquePaths(3, 2))


def uniquePathsSpaceOptimized(m, n):
    prev = [0] * m
    for i in range(n):
        temp = [0] * m
        for j in range(m):
            if i == 0 and j == 0:
                temp[0] = 1
                continue
            up = 0
            left = 0
            if i > 0:
                up = prev[i - 1]
            if j > 0:
                left = temp[j - 1]
            temp[j] = left + up

        prev = temp
    return prev[-1]


# print(uniquePathsSpaceOptimized(3, 2))


def mazeObstacles(n, m, mat):
    # Write your code here.
    prev = [0] * m

    for i in range(n):
        temp = [0] * m
        for j in range(m):
            if i == 0 and j == 0:
                temp[0] = 1
                continue
            if mat[i][j] == -1:
                continue
            up = 0
            left = 0
            if i > 0 and mat[i - 1][j] == 0:
                up = prev[j]
            if j > 0 and mat[i][j - 1] == 0:
                left = temp[j - 1]
            temp[j] = up + left
        prev = temp
    return prev[-1] % ((10**9) + 7)


# note iteration is starting from bottom to up since bottom is variable but starting is fixed for rows
class SolutionCherryPickup:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        dp = [[[0] * cols for _ in range(cols)] for _ in range(rows)]

        for j1 in range(cols):
            for j2 in range(cols):
                if j1 == j2:
                    dp[rows - 1][j1][j2] = grid[rows - 1][j1]
                else:
                    dp[rows - 1][j1][j2] = grid[rows - 1][j1] + grid[rows - 1][j2]
        for i in range(rows - 2, -1, -1):
            for j1 in range(cols):
                for j2 in range(cols):
                    max_cherries = float("-inf")
                    for delta_j1 in range(-1, 2):
                        for delta_j2 in range(-1, 2):
                            current_cherries = 0
                            if j1 == j2:
                                current_cherries = grid[i][j1]
                            else:
                                current_cherries = grid[i][j1] + grid[i][j2]
                            if j1 + delta_j1 < 0 or j1 + delta_j1 >= cols or j2 + delta_j2 < 0 or j2 + delta_j2 >= cols:
                                current_cherries += float("-inf")
                            else:
                                current_cherries += dp[i + 1][j1 + delta_j1][j2 + delta_j2]

                            max_cherries = max(max_cherries, current_cherries)
                    dp[i][j1][j2] = max_cherries

        return dp[0][0][cols - 1]


# https://www.naukri.com/code360/problems/partition-equal-subset-sum_892980?source=youtube&campaign=striver_dp_videos&leftPanelTabValue=SUBMISSION
def canPartition(arr, n):
    # Write your code here.
    total_sum = sum(arr)
    if total_sum % 2 != 0:
        return False

    k = total_sum // 2
    dp = [[False] * (k + 1) for _ in range(n)]

    for i in range(n):
        dp[i][0] = True

    for index in range(1, n):
        for target in range(1, k + 1):
            notTake = dp[index - 1][target]
            take = False
            if arr[index] <= target:
                take = dp[index - 1][target - arr[index]]

            dp[index][target] = take or notTake

    return dp[n - 1][k]


def countPartitions(d, arr):
    mod = int(1e9 + 7)
    n = len(arr)
    totalSum = sum(arr)
    if totalSum - d < 0:
        return 0

    if (totalSum - d) % 2 != 0:
        return 0

    s2 = (totalSum - d) // 2

    memo = [[-1 for _ in range(s2 + 1)] for _ in range(n)]

    def countPartitionsUtil(index, target, arr, memo):
        if index == 0:
            if target == 0 and arr[0] == 0:
                return 2
            if target == 0 or target == arr[0]:
                return 1
            return 0

        if memo[index][target] != -1:
            return memo[index][target]

        notTaken = countPartitionsUtil(index - 1, target, arr, memo)

        taken = 0
        if arr[index] <= target:
            taken = countPartitionsUtil(index - 1, target - arr[index], arr, memo)

        memo[index][target] = (notTaken + taken) % mod

        return memo[index][target]
