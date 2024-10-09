from typing import List


# frog jump using memoization
def frogJumpMemoized(n, heights):
    memo = {}

    def recursiveCall(index, heights, memo):
        if index == 0:
            return 0

        if index in memo:
            return memo[index]

        left = recursiveCall(index - 1, heights, memo) + abs(heights[index] - heights[index - 1])
        right = float("inf")
        if index > 1:
            right = recursiveCall(index - 2, heights, memo) + abs(heights[index] - heights[index - 2])

        memo[index] = min(left, right)

        return memo[index]

    return recursiveCall(n - 1, heights, memo)


# frog jump using tabulation
def frogJumpTabulation(n, heights):
    dp = [0] * n
    for i in range(1, n):
        single_step = dp[i - 1] + abs(heights[i] - heights[i - 1])
        double_steps = float("inf")

        if i > 1:
            double_steps = dp[i - 2] + abs(heights[i] - heights[i - 2])

        dp[i] = min(single_step, double_steps)

    return dp[n - 1]


# frog jump using tabulation space optimized
def frogJumpTabulationOptimized(n, heights):
    prev = 0
    prev2 = 0

    for i in range(1, n):
        single_step = prev + abs(heights[i] - heights[i - 1])

        double_step = float("inf")

        if i > 1:
            double_step = prev2 + abs(heights[i] - heights[i - 2])

        current = min(single_step, double_step)

        prev2 = prev
        prev = current

    return prev


# frog jump with k jumps tabulation
def frogKJumpTabulation(n, k, heights):
    dp = [0] * n

    for i in range(1, n):
        min_jump = float("inf")
        for j in range(1, k + 1):
            if i >= j:
                j_step = dp[i - j] + abs(heights[i] - heights[i - j])
                min_jump = min(min_jump, j_step)
        dp[i] = min_jump

    return dp[n - 1]


# maximum sum in array picking non-adjacent elements, memoized
def maximumNonAdjacentSumMemoized(nums):
    n = len(nums)

    memo = {}

    def recursiveCall(index, nums):
        if index in memo:
            return memo[index]
        if index == 0:
            return nums[0]

        if index < 0:
            return 0

        pick = recursiveCall(index - 2, nums) + nums[index]
        notPick = recursiveCall(index - 1, nums) + 0

        memo[index] = max(pick, notPick)

        return memo[index]

    return recursiveCall(n - 1, nums)


# maximum sum in array picking non-adjacent elements using tabulation
def maximumNonAdjacentSumTabulation(arr):
    n = len(arr)
    # code here
    dp = [0] * n
    dp[0] = arr[0]

    for i in range(1, n):
        pick = arr[i]
        if i > 1:
            pick += dp[i - 2]
        notPick = dp[i - 1]

        dp[i] = max(pick, notPick)

    return dp[n - 1]


# maximum sum in array picking non-adjacent elements using tabulation optimized
def maximumNonAdjacentSumTabulationOptimized(arr):
    n = len(arr)
    prev = arr[0]
    prev2 = 0

    for i in range(1, n):
        pick = arr[i] + prev2
        notPick = prev

        current = max(pick, notPick)

        prev2 = prev
        prev = current

    return prev


# house robber when houses are in a circle tabulation optimized
def houseRobberTabulationOptimized(houses):
    if len(houses) == 1:
        return houses[0]

    def helper(nums, length):
        prev2 = 0
        prev = nums[0]

        for i in range(1, length):
            rob = nums[i] + prev2
            notRob = prev

            current = max(rob, notRob)

            prev2 = prev
            prev = current

        return prev

    n = len(houses)

    return max(helper(houses[: n - 1], n - 1), helper(houses[1:], n - 1))


# ninja training memoized solution
def ninjaTrainingMemoized(n, points):
    memo = {}

    def recursiveCall(day, last, points, memo):
        if (day, last) in memo:
            return memo[(day, last)]

        if day == 0:
            max_value = 0
            for task in range(3):
                if task != last:
                    max_value = max(max_value, points[0][task])

            memo[(day, last)] = max_value

            return memo[(day, last)]

        max_value = 0
        for task in range(3):
            if task != last:
                point = points[day][task] + recursiveCall(day - 1, task, points, memo)
                max_value = max(max_value, point)

        memo[(day, last)] = max_value

        return memo[(day, last)]

    return recursiveCall(n - 1, 3, points, memo)


# ninja training tabulation solution
def ninjaTrainingTabulation(n, points):
    dp = [[0] * 4 for _ in range(n)]

    dp[0][0] = max(points[0][1], points[0][2])
    dp[0][1] = max(points[0][0], points[0][2])
    dp[0][2] = max(points[0][0], points[0][1])
    dp[0][3] = max(points[0][0], points[0][1], points[0][2])

    for day in range(n):
        for last in range(4):
            max_value = 0
            for task in range(3):
                if task != last:
                    point = points[day][task] + dp[day - 1][task]
                    max_value = max(max_value, point)

            dp[day][last] = max_value

    return dp[n - 1][3]


# ninja training tabulation space optimized
def ninjaTrainingTabulationOptimized(n, points):
    prev = [0] * 4

    prev[0] = max(points[0][1], points[0][2])
    prev[1] = max(points[0][1], points[0][2])
    prev[2] = max(points[0][1], points[0][2])
    prev[3] = max(points[0][1], points[0][2])

    for day in range(n):
        temp = [0] * 4
        for last in range(4):
            temp[last] = 0

            for task in range(3):
                if task != last:
                    temp[last] = max(temp[last], points[day][task] + prev[task])

        prev = temp

    return prev[3]


# unique paths memoized
def uniquePathsMemoized(n, m):
    memo = [[-1] * m for _ in range(n)]

    def recursiveCall(i, j, memo):
        if memo[i][j] != -1:
            return memo[i][j]

        if i == 0 and j == 0:
            return 1
        if i < 0 or j < 0:
            return 0
        up = recursiveCall(i - 1, j, memo)
        left = recursiveCall(i - 1, j, memo)
        memo[i][j] = up + left

        return memo[i][j]

    return recursiveCall(n - 1, m - 1, memo)


# unique paths tabulation
def uniquePathsTabulation(n, m):
    dp = [[0] * m for _ in range(n)]
    dp[0][0] = 1

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            up = 0
            left = 0

            if i > 0:
                up = dp[i - 1][j]
            if j > 0:
                left = dp[i][j - 1]

            dp[i][j] = up + left

    return dp[n - 1][m - 1]


# unique paths tabulation space optimized
def uniquePathsTabulationOptimized(n, m):
    prev = [0] * m

    for i in range(n):
        current = [0] * m
        for j in range(m):
            if i == 0 and j == 0:
                current[0] = 1
            else:
                up = prev[j]
                left = 0
                if j > 1:
                    left = current[j - 1]
                current[j] = up + left

        prev = current

    return prev[-1]


# unique paths with obstacles
def mazeObstaclesMemoized(n, m, mat):
    mod = int(1e9 + 7)
    memo = [[-1] * m for _ in range(n)]

    def recursiveCall(i, j, mat, memo):
        if memo[i][j] != -1:
            return memo[i][j]

        if i >= 0 and j >= 0 and mat[i][j] == -1:
            return 0
        if i == 0 and j == 0:
            return 1
        if i < 0 or j < 0:
            return 0

        up = 0
        left = 0

        if j > 0:
            left = recursiveCall(i, j - 1, mat, memo)

        if i > 0:
            up = recursiveCall(i - 1, j, mat, memo)

        memo[i][j] = (up + left) % mod

        return memo[i][j]

    return recursiveCall(n - 1, m - 1, mat, memo)


# unique paths with obstacles tabulation
def mazeObstaclesTabulation(n, m, mat):
    mod = int(1e9 + 7)
    dp = [[0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if i >= 0 and j >= 0 and mat[i][j] == -1:
                dp[i][j] = 0
            if i == 0 and j == 0:
                dp[0][0] = 1
            else:
                up = 0
                left = 0
                if i > 0:
                    up = dp[i - 1][j]
                if j > 0:
                    left = dp[i][j - 1]

                dp[i][j] = (left + up) % mod

    return dp[n - 1][m - 1]


# unique paths with obstacles tabulation space optimized
def mazeObstaclesTabulationOptimized(n, m, mat):
    mod = int(1e9 + 7)
    prev = [0] * m

    for i in range(n):
        current = [0] * m
        for j in range(m):
            if mat[i][j] == -1:
                current[j] = 0
            if i == 0 and j == 0:
                current[0] = 1
            else:
                up = prev[j]
                left = 0
                if j > 0:
                    left = current[j - 1]
                current[j] = (up + left) % mod

        prev = current

    return prev[-1]


# min path sum memoized
def minPathSumMemoized(grid):
    n = len(grid)
    m = len(grid[0])

    memo = [[-1] * m for _ in range(n)]

    def recursiveCall(i, j):
        if i == 0 and j == 0:
            return grid[0][0]

        if i < 0 or j < 0:
            return float("inf")

        up = grid[i][j] + recursiveCall(i - 1, j)
        left = grid[i][j] + recursiveCall(i, j - 1)

        memo[i][j] = min(up, left)

        return memo[i][j]

    return recursiveCall(n - 1, m - 1)


# min path sum tabulation
def minPathSumTabulation(grid):
    n = len(grid)
    m = len(grid[0])

    dp = [[0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                dp[0][0] = grid[0][0]
            else:
                if i > 0:
                    up = dp[i - 1][j] + grid[i][j]
                else:
                    up = float("inf")

                if j > 0:
                    left = dp[i][j - 1] + grid[i][j]
                else:
                    left = float("inf")

                dp[i][j] = min(left, up)

    return dp[n - 1][m - 1]


# min path sum tabulation space optimized
def minPathSumTabulationOptimized(grid):
    n = len(grid)
    m = len(grid[0])

    prev = [0] * m

    for i in range(n):
        current = [0] * m
        for j in range(m):
            if i == 0 and j == 0:
                current[0] = grid[0][0]
            else:
                up = prev[j] + grid[i][j]
                if j > 0:
                    left = current[j - 1] + grid[i][j]
                else:
                    left = float("inf")

                current[j] = min(up, left)

        prev = current

    return prev[-1]


# min path sum for a triangle memoized
def minimumPathSumMemoized(triangle, n):
    memo = [[-1] * n for _ in range(n)]

    def recursiveCall(i, j):
        if i == n - 1:
            return triangle[n - 1][j]

        if memo[i][j] != -1:
            return memo[i][j]

        down = triangle[i][j] + recursiveCall(i + 1, j)
        diagonal = triangle[i][j] + recursiveCall(i + 1, j + 1)

        memo[i][j] = min(down, diagonal)

        return memo[i][j]

    return recursiveCall(0, 0)


# min path for a triangle tabulation
def minimumPathSumTabulation(triangle, n):
    dp = [[0] * n for _ in range(n)]

    for j in range(n):
        dp[n - 1][j] = triangle[n - 1][j]

    for i in range(n - 2, -1, -1):
        for j in range(i, -1, -1):
            if i == n - 1:
                dp[n - 1][j] = triangle[n - 1][j]
            else:
                down = triangle[i][j] + dp[i + 1][j]
                diagonal = triangle[i][j] + dp[i + 1][j + 1]

                dp[i][j] = min(down, diagonal)

    return dp[0][0]


def minimumPathSumTabulationOptimized(triangle, n):
    front = [0] * n
    current = [0] * n

    for j in range(n):
        front[j] = triangle[n - 1][j]

    for i in range(n - 2, -1, -1):
        for j in range(i, -1, -1):
            down = triangle[i][j] + front[j]
            diagonal = triangle[i][j] + front[j + 1]

            current[j] = min(down, diagonal)

        front = current

    return front[0]


# get max path sum memoized
def getMaxPathSumMemoized(matrix):
    n = len(matrix)
    m = len(matrix[0])

    memo = [[-1] * m for _ in range(n)]

    def recursiveCall(i, j):
        if i < 0 or j >= m:
            return float("-inf")

        if i == 0:
            return matrix[0][j]

        if memo[i][j] != -1:
            return memo[i][j]

        up = matrix[i][j] + recursiveCall(i - 1, j)
        left_diagonal = matrix[i][j] + recursiveCall(i - 1, j - 1)
        right_diagonal = matrix[i][j] + recursiveCall(i - 1, j + 1)

        memo[i][j] = max(up, left_diagonal, right_diagonal)

        return memo[i][j]

    max_value = float("-inf")
    for j in range(m):
        max_value = max(max_value, recursiveCall(n - 1, j))

    return max_value


# get max sum tabulation
def getMaxPathSumTabulation(matrix):
    n = len(matrix)
    m = len(matrix[0])

    dp = [[0] * m for _ in range(n)]

    for j in range(m):
        dp[0][j] = matrix[0][j]

    for i in range(1, n):
        for j in range(m):
            up = dp[i - 1][j] + matrix[i][j]

            left_diagonal = float("-inf")
            right_diagonal = float("-inf")

            if j > 0:
                left_diagonal = matrix[i][j] + dp[i - 1][j - 1]
            if j < m:
                right_diagonal = matrix[i][j] + dp[i - 1][j + 1]

            dp[i][j] = max(up, left_diagonal, right_diagonal)

    return max(dp[n - 1])


# get max path tabulation optimized
def getMaxPathSumTabulationOptimized(matrix):
    n = len(matrix)
    m = len(matrix[0])

    prev = [matrix[0][j] for j in range(m)]

    current = [0] * m

    for i in range(1, n):
        for j in range(m):
            up = matrix[i][j] + prev[j]
            left_diagonal = float("-inf")
            right_diagonal = float("-inf")

            if j > 0:
                left_diagonal = matrix[i][j] + prev[j - 1]
            if j < m - 1:
                right_diagonal = matrix[i][j] + prev[j + 1]

            current[j] = max(up, left_diagonal, right_diagonal)

        prev = current

    return max(prev)


# max chocolates using memoization
def maximumChocolatesMemoized(r, c, grid):
    memo = [[[-1] * c for _ in range(c)] for _ in range(c)]

    def recursiveCall(i, j1, j2):
        if j1 < 0 or j2 < 0 or j1 >= c or j2 >= c:
            return float("-inf")
        if i == r - 1:
            if j1 == j2:
                return grid[i][j1]
            else:
                return grid[i][j1] + grid[i][j2]
        if memo[i][j1][j2] != -1:
            return memo[i][j1][j2]

        # explore all paths of alice and bob simultenously
        max_value = float("-inf")
        for dj1 in range(-1, 2):
            for dj2 in range(-1, 2):
                value = 0
                if j1 == j2:
                    value = grid[i][j1]
                else:
                    value = grid[i][j1] + grid[i][j2]

                value += recursiveCall(i + 1, j1 + dj1, j2 + dj2)

                max_value = max(max_value, value)

        memo[i][j1][j2] = max_value

        return memo[i][j1][j2]

    return recursiveCall(0, 0, c - 1)


# maximum chocolates using tabulation
def maximumChocolatesTabulation(r, c, grid):
    dp = [[[0] * c for _ in range(c)] for _ in range(r)]

    for j1 in range(c):
        for j2 in range(c):
            if j1 == j2:
                dp[r - 1][j1][j2] = grid[r - 1][j1]
            else:
                dp[r - 1][j1][j2] = grid[r - 1][j1] + grid[r - 1][j2]

    for i in range(r - 2, -1, -1):
        for j1 in range(c):
            for j2 in range(c):
                max_value = float("-inf")
                for dj1 in range(-1, 2):
                    for dj2 in range(-1, 2):

                        value = float("-inf")

                        if j1 + dj1 >= 0 and j1 + dj1 < c and j2 + dj2 >= 0 and j2 + dj2 < c:
                            if j1 == j2:
                                value += grid[i][j1]
                            else:
                                value = grid[i][j1] + grid[i][j2]

                            value += dp[i + 1][j1 + dj1][j2 + dj2]

                        max_value = max(max_value, value)

                dp[i][j1][j2] = max_value

    return dp[0][0][c - 1]


# max chocolates using tabulation space optimized
def maximumChocolatesTabulationOptimized(r, c, grid):
    prev = [[0] * c for _ in range(c)]
    current = [[0] * c for _ in range(c)]

    for j1 in range(c):
        for j2 in range(c):
            if j1 == j2:
                prev[j1][j2] = grid[r - 1][j2]
            else:
                prev[j1][j2] = grid[r - 1][j1] + grid[r - 1][j2]

    for i in range(r - 2, -1, -1):
        for j1 in range(c):
            for j2 in range(c):
                max_value = float("-inf")

                for dj1 in range(-1, 2):
                    for dj2 in range(-1, 2):
                        value = 0
                        if j1 + dj1 < 0 or j1 + dj1 >= c or j2 + dj2 < 0 or j2 + dj2 >= c:
                            value = float("-inf")
                        else:
                            if j1 == j2:
                                value = grid[i][j1] + prev[j1 + dj1][j2 + dj2]
                            else:
                                value = grid[i][j1] + grid[i][j2] + prev[j1 + dj1][j2 + dj2]

                        max_value = max(max_value, value)

                current[j1][j2] = max_value

        prev = current[::]

    return prev[0][c - 1]


# subset sum equal to K memoized
def subsetSumToKMemoized(n, k, arr):
    memo = [[-1] * (k + 1) for _ in range(n)]

    def recursiveCall(index, target, arr):
        if target == 0:
            return True
        if index == 0:
            return arr[0] == target

        if memo[index][target] != -1:
            return memo[index][target]

        notTake = recursiveCall(index - 1, target, arr)
        take = False

        if arr[index] <= target:
            take = recursiveCall(index - 1, target - arr[index], arr)

        memo[index][target] = take or notTake

        return memo[index][target]

    return recursiveCall(n - 1, k)


def subsetSumToKTabulation(n, k, arr):
    dp = [[False] * (k + 1) for _ in range(n)]

    for i in range(n):
        dp[i][0] = True

    dp[0][arr[0]] = True

    for index in range(n):
        for target in range(1, k + 1):
            notTake = dp[index - 1][target]
            take = False
            if arr[index] <= target:
                take = dp[index - 1][target - arr[index]]

            dp[index][target] = take or notTake

    return dp[n - 1][k]


def subsetSumToKTabulationOptimized(n, k, arr):
    prev = [0] * (k + 1)
    current = [0] * (k + 1)

    prev[0] = current[0] = True

    prev[arr[0]] = True

    for index in range(n):
        for target in range(k + 1):
            notTake = prev[target]

            take = False

            if arr[index] <= target:
                take = prev[target - arr[index]]

            current[target] = take or notTake

        prev = current

    return prev[k]


# if we can partition the array into two subsets with equal sum
def canPartition(arr, n):
    total_sum = sum(arr)

    if total_sum % 2 == 1:
        return False

    target = total_sum // 2

    def subsetSumToTarget(subset, target):
        prev = [False] * (target + 1)
        current = [False] * (target + 1)

        current[0] = True
        current[arr[0]] = True

        for i in range(n):
            for j in range(target):
                notTake = prev[j]

                take = False
                if subset[i] <= j:
                    take = prev[j - subset[i]]
                current[j] = take or notTake

            prev = current[:]

        return prev[-1]

    return subsetSumToTarget(arr, target)


# minimum subset difference
def minSubsetSumDifferenceTabulation(arr, n):
    total_sum = sum(arr)

    k = total_sum

    dp = [[0] * (k + 1) for _ in range(n)]
    for i in range(n):
        dp[i][0] = True

    if arr[0] <= k:
        dp[0][arr[0]] = True

    for index in range(1, n):
        for target in range(1, k + 1):
            notTake = dp[index - 1][target]
            take = False

            if arr[index] <= target:
                take = dp[index - 1][target - arr[index]]

            dp[index][target] = take or notTake

    min_diff = float("inf")

    for s1 in range(total_sum // 2):
        if dp[n - 1][s1]:
            # s2 = total_sum - s1
            # difference between s1 and s2 is thus s1 - s2
            # which is s1 - (total_sum - s1) which becomes 2s1 - total_sum

            min_diff = min(min_diff, abs((2 * s1) - total_sum))

    return min_diff


# find number of ways of achieving k from the array by summing various elements
def findWaysMemoized(arr: List[int], k: int) -> int:
    n = len(arr)
    memo = [[-1] * (k + 1) for _ in range(n)]

    def recursiveCall(index, target):
        if target == 0:
            return 1

        if index == 0:
            return 1 if arr[0] == target else 0

        if memo[index][target] != -1:
            return memo[index][target]

        notTake = recursiveCall(index - 1, target)
        take = 0

        if arr[index] <= target:
            take = recursiveCall(index - 1, target - arr[index])

        memo[index][target] = take + notTake

        return memo[index][target]

    return recursiveCall(n - 1, k)


# find number of ways tabulation
def findWaysTabulation(arr: List[int], k: int) -> int:
    n = len(arr)
    dp = [[0] * (k + 1) for _ in range(n)]

    for i in range(n):
        dp[i][0] = 1

    if arr[0] <= k:
        dp[0][arr[0]] = 1

    # to count for presence of 0 in the subsets, we use
    if arr[0] == 0:
        dp[0][0] = 2

    for index in range(1, n):
        for target in range(1, k + 1):
            notTake = dp[index - 1][target]
            take = 0
            if arr[index] <= target:
                take = dp[index - 1][target - arr[index]]

            dp[index][target] = notTake + take

    return dp[n - 1][k]


# find number of ways tabulation optimized
def findWaysTabulationOptimized(arr: List[int], k: int) -> int:
    n = len(arr)

    prev = [0] * (k + 1)
    current = [0] * (k + 1)

    prev[0] = 1
    current[0] = 1

    if arr[0] <= k:
        prev[arr[0]] = 1

    if arr[0] == 0:
        current[0] = 2

    for index in range(1, n):
        for target in range(k + 1):
            notTake = prev[target]

            take = 0
            if arr[index] <= target:
                take = prev[target - arr[index]]

            current[target] = take + notTake

        prev = current[:]

    return prev[k]


# count partitions given difference using memoization
def countPartitionsMemoized(n, d, arr):
    # total_sum = S1 + S2
    # S1 - S2 = D
    # S1 = total_sum - S2
    # S1 - S2 = D can thus be rewritten as (total_sum - S2) - S2 = D
    # total_sum - 2S2 = D
    # 2S2 = total_sum - D
    # S2 = (total_sum - D) / 2
    # thus the question can be restated as number of subsets whose sum is S2, given by the formula S2 = (total_sum - D) / 2
    # we know both total_sum and D, thus we can compute S2

    total_sum = sum(arr)
    if (total_sum - d) < 0 or (total_sum - d) % 2 != 0:
        return 0

    S2 = (total_sum - d) // 2

    MOD = (10**9) + 7

    # we use the findWaysMemoized function in the previous problems, scroll above
    return findWaysMemoized(arr, S2) % MOD


# count partitions given difference using tabulation
def countPartitionsTabulation(n, d, arr):
    total_sum = sum(arr)
    k = (total_sum - d) / 2

    dp = [[0] * (k + 1) for _ in range(n)]

    for i in range(n):
        dp[i][0] = 1

    # to count for presence of 0 in the subsets, we use
    if arr[0] == 0:
        dp[0][0] = 2

    if arr[0] != 0 and arr[0] <= k:
        dp[0][arr[0]] = 1

    for index in range(1, n):
        for target in range(1, k + 1):
            notTake = dp[index - 1][target]
            take = 0
            if arr[index] <= target:
                take = dp[index - 1][target - arr[index]]

            dp[index][target] = notTake + take

    return dp[n - 1][k]


# count partitions given difference using tabulation space optimized
def countPartitionsTabulationOptimized(n, d, arr):
    total_sum = sum(arr)
    if (total_sum - d) < 0 or (total_sum - d) % 2 != 0:
        return 0

    S2 = (total_sum - d) // 2

    prev = [0] * (S2 + 1)
    current = [0] * (S2 + 1)

    prev[0] = 1
    current[0] = 1

    if arr[0] <= S2:
        prev[arr[0]] = 1

    if arr[0] == 0:
        prev[0] = 2

    MOD = (10**9) + 7

    for index in range(1, n):
        for target in range(S2 + 1):
            notTake = prev[target]
            take = 0
            if arr[index] <= target:
                take = prev[target - arr[index]]

            current[target] = (take + notTake) % MOD

        prev = current[::]

    return prev[S2] % MOD


# knapsack problem using memoization
def knapsackMemoized(weights, values, n, maxWeight):
    memo = [[-1] * maxWeight for _ in range(n)]

    def recursiveCall(index, maxWeight, weights, values):
        if index == 0:
            if weights[0] <= maxWeight:
                return values[index]
            else:
                return 0

        if memo[index][maxWeight] != -1:
            return memo[index][maxWeight]

        notTake = recursiveCall(index - 1, maxWeight, weights, values)
        take = float("-inf")

        if weights[index] <= maxWeight:
            take = values[index] + recursiveCall(index - 1, maxWeight - weights[index], weights, values)

        memo[index][maxWeight] = max(take, notTake)

        return memo[index][maxWeight]

    return recursiveCall(n - 1, maxWeight, weights, values)


# knapsack problem using tabulation
def knapsackTabulation(weights, values, n, maxWeight):
    dp = [[0] * (maxWeight + 1) for _ in range(n)]

    for w in range(maxWeight + 1):
        dp[0][w] = values[0]

    for index in range(1, n):
        for w in range(maxWeight + 1):
            notTake = dp[index - 1][w]
            take = float("-inf")

            if weights[index] <= w:
                take = values[index] + dp[index - 1][w - weights[index]]

            dp[index][w] = max(take, notTake)

    return dp[n - 1][maxWeight]


# knapsack problem using tabulation optimized
def knapsackTabulationOptimized(weights, values, n, maxWeight):
    prev = [weights[i] for i in range(maxWeight + 1)]
    current = [0] * (maxWeight + 1)

    for index in range(n):
        for w in range(maxWeight + 1):
            notTake = prev[w]

            take = float("-inf")

            if weights[index] <= w:
                take = values[index] + prev[w - weights[index]]

            current[w] = max(take, notTake)

        prev = current

    return prev[-1]


# knapsack problem using tabulation optimized to use only a single row
def knapsackTabulationOptimizedToOneRow(weights, values, n, maxWeight):
    prev = [weights[i] for i in range(maxWeight + 1)]

    for index in range(n):
        for w in range(maxWeight, -1, -1):
            notTake = prev[w]

            take = float("-inf")

            if weights[index] <= w:
                take = values[index] + prev[w - weights[index]]

            prev[w] = max(take, notTake)

    return prev[-1]


# find the minimum number of elements to form a given target using memoization
def minimumElementsMemoized(nums, target):

    n = len(nums)
    memo = [[-1] * (target + 1) for _ in range(n)]

    def recursiveCall(index, current_target):
        if index == 0:
            if current_target % nums[0] == 0:
                return current_target // nums[0]
            else:
                return float("inf")
        if memo[index][current_target] != -1:
            return memo[index][current_target]

        notTake = 0 + recursiveCall(index - 1, current_target, nums)
        take = float("inf")

        if nums[index] <= current_target:
            take = 1 + recursiveCall(index, current_target - nums[index], nums)

        memo[index][current_target] = min(take, notTake)

        return memo[index][current_target]

    result = recursiveCall(n - 1, target)

    if result >= float("inf"):
        return -1

    return result


# find the minimum number of elements to form a given target using tabulation
