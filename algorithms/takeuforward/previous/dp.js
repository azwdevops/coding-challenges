// frog jump using memoization
class SolutionFrogJump {
  solve(index, heights, dp) {
    if (index === 0) return 0;
    if (dp[index] !== -1) return dp[index];

    let jumpTwo = Infinity;
    let jumpOne = this.solve(index - 1, heights, dp) + Math.abs(heights[index] - heights[index - 1]);
    if (index > 1) {
      jumpTwo = this.solve(index - 2, heights, dp) + Math.abs(heights[index] - heights[index - 2]);
    }
    dp[index] = Math.min(jumpOne, jumpTwo);
    return dp[index];
  }

  frogJump(n, heights) {
    const dp = new Array(n).fill(-1);
    return this.solve(n - 1, heights, dp);
  }
}

// const solution = new SolutionFrogJump();
// console.log(solution.frogJump(4, [10, 20, 30, 10]));

// frog jump using tabulation
class SolutionFrogJumpTabulation {
  frogJump(n, heights) {
    const dp = new Array(n).fill(-1);
    dp[0] = 0;
    for (let i = 1; i < n; i++) {
      let jumpTwo = Infinity;
      let jumpOne = dp[i - 1] + Math.abs(heights[i] - heights[i - 1]);
      if (i > 1) {
        jumpTwo = dp[i - 2] + Math.abs(heights[i] - heights[i - 2]);
      }
      dp[i] = Math.min(jumpOne, jumpTwo);
    }
    return dp[n - 1];
  }
}

// const solution = new SolutionFrogJumpTabulation();
// console.log(solution.frogJump(4, [10, 20, 30, 10]));

class SolutionMaximumNonAdjacentSum {
  recursiveCall(currentIndex, nums, memo) {
    if (currentIndex in memo) {
      return memo[currentIndex];
    }
    if (currentIndex === 0) {
      return nums[currentIndex];
    }
    if (currentIndex < 0) {
      return 0;
    }
    const pick = nums[currentIndex] + this.recursiveCall(currentIndex - 2, nums, memo);
    const notPick = 0 + this.recursiveCall(currentIndex - 1, nums, memo);

    const currentResult = Math.max(pick, notPick);
    memo[currentIndex] = currentResult;

    return currentResult;
  }
  maximumNonAdjacentSum(nums) {
    const memo = {};
    return this.recursiveCall(nums.length - 1, nums, memo);
  }
}

// const solution = new SolutionMaximumNonAdjacentSum();
// console.log(solution.maximumNonAdjacentSum([1, 2, 3, 1, 3, 5, 8, 1, 9]));

const maximumNonAdjacentSumTabulation = function (nums) {
  const n = nums.length;
  const dp = new Array(n).fill(-1);
  dp[0] = nums[0];
  for (let i = 1; i < n; i++) {
    let pick = nums[i];
    if (i > 1) {
      pick += dp[i - 2];
    }
    const nonPick = dp[i - 1];
    dp[i] = Math.max(pick, nonPick);
  }
  return dp[n - 1];
};

// console.log(maximumNonAdjacentSumTabulation([1, 2, 3, 1, 3, 5, 8, 1, 9]));

const maximumNonAdjacentSumTabulationSpaceOptimized = function (nums) {
  let prev = 0;
  let prev2 = 0;
  const n = nums.length;
  for (let i = 0; i < n; i++) {
    let pick = nums[i];
    if (i > 1) {
      pick += prev2;
    }
    const nonPick = prev;

    const current = Math.max(pick, nonPick);
    prev2 = prev;
    prev = current;
  }
  return prev;
};

// console.log(maximumNonAdjacentSumTabulationSpaceOptimized([1, 2, 3, 1, 3, 5, 8, 1, 9]));

class SolutionNinjaTraining {
  recursiveCall(day, prevTask, points) {
    if (day === 0) {
      let max_value = 0;
      for (let task = 0; task < 3; task++) {
        if (task !== prevTask) {
          max_value = Math.max(max_value, points[0][task]);
        }
      }
      return max_value;
    }
    let max_value = 0;
    for (let task = 0; task < 3; task++) {
      if (task !== prevTask) {
        const currentPoints = points[day][task] + this.recursiveCall(day - 1, task, points);
        max_value = Math.max(max_value, currentPoints);
      }
    }
    return max_value;
  }
  ninjaTraining(n, points) {
    return this.recursiveCall(n - 1, 3, points);
  }
}

// const solution = new SolutionNinjaTraining();
// console.log(
//   solution.ninjaTraining(3, [
//     [1, 2, 5],
//     [3, 1, 1],
//     [3, 3, 3],
//   ])
// );

class SolutionUniquePaths {
  recursiveCall(i, j) {
    if (i === 0 && j === 0) {
      return 1;
    }
    if (i < 0 || j < 0) {
      return 0;
    }
    const up = this.recursiveCall(i - 1, j);
    const left = this.recursiveCall(i, j - 1);
    return up + left;
  }
  uniquePaths(m, n) {
    return this.recursiveCall(n - 1, m - 1);
  }
}

// const solution = new SolutionUniquePaths();
// console.log(solution.uniquePaths(3, 2));

class SolutionUniquePaths2 {
  recursiveCall(i, j, mat, dp) {
    if (i < 0 || j < 0 || mat[i][j] === -1) {
      return 0;
    }
    if (i === 0 && j === 0) {
      return 1;
    }
    if (dp[i][j] !== -1) {
      return dp[i][j];
    }
    let up = this.recursiveCall(i - 1, j, mat, dp);
    let left = this.recursiveCall(i, j - 1, mat, dp);
    dp[i][j] = up + left;

    return dp[i][j];
  }
  mazeObstacles(n, m, mat) {
    const dp = Array.from({ length: n }, () => new Array(m).fill(-1));
    return this.recursiveCall(n - 1, m - 1, mat, dp);
  }
}

// const solution = new SolutionUniquePaths2();
// console.log(
//   solution.mazeObstacles(3, 3, [
//     [0, 0, 0],
//     [0, -1, 0],
//     [0, 0, 0],
//   ])
// );

const minSumPath = function (grid) {
  const rows = grid.length;
  const cols = grid[0].length;
  const memo = Array.from({ length: rows }, () => Array(cols).fill(-1));

  const recursiveCall = function (i, j) {
    if (i === 0 && j === 0) {
      return grid[0][0];
    }
    if (i < 0 || j < 0) {
      return Infinity;
    }
    if (memo[i][j] !== -1) {
      return memo[i][j];
    }
    const up = grid[i][j] + recursiveCall(i - 1, j);
    const left = grid[i][j] + recursiveCall(i, j - 1);
    memo[i][j] = Math.min(up, left);
    return memo[i][j];
  };

  return recursiveCall(rows - 1, cols - 1);
};

// console.log(
//   minSumPath([
//     [5, 9, 6],
//     [11, 5, 2],
//   ])
// );

const minSumPathTriangle = function (triangle, n) {
  const recursiveCall = function (i, j) {
    if (i === n - 1) {
      return triangle[n - 1][j];
    }
    const down = triangle[i][j] + recursiveCall(i + 1, j);
    const diagonal = triangle[i][j] + recursiveCall(i + 1, j + 1);

    return Math.min(down, diagonal);
  };

  return recursiveCall(0, 0);
};

// console.log(minSumPathTriangle([[1], [2, 3], [3, 6, 7], [8, 9, 6, 10]], 4));

const minSumPathTriangleTabulation = function (triangle, n) {
  const dp = Array.from({ length: n }, () => Array(n).fill(0));
  for (let j = 0; j < n; j++) {
    dp[n - 1][j] = triangle[n - 1][j];
  }
  for (let i = n - 2; i >= 0; i--) {
    for (let j = 0; j <= i; j++) {
      const down = triangle[i][j] + dp[i + 1][j];
      const diagonal = triangle[i][j] + dp[i + 1][j + 1];
      dp[i][j] = Math.min(down, diagonal);
    }
  }
  return dp[0][0];
};

// console.log(minSumPathTriangleTabulation([[1], [2, 3], [3, 6, 7], [8, 9, 6, 10]], 4));

const getMaxPathSum = function (matrix) {
  const n = matrix.length;
  const m = matrix[0].length;

  const recursiveCall = function (i, j) {
    if (j < 0 || j >= m) {
      return -Infinity;
    }
    if (i === 0) {
      return matrix[0][j];
    }
    const down = matrix[i][j] + recursiveCall(i - 1, j);
    const leftDiagonal = matrix[i][j] + recursiveCall(i - 1, j - 1);
    const rightDiagonal = matrix[i][j] + recursiveCall(i - 1, j + 1);

    return Math.max(down, leftDiagonal, rightDiagonal);
  };

  let max_value = -Infinity;

  for (let j = 0; j < m; j++) {
    max_value = Math.max(max_value, recursiveCall(n - 1, j));
  }
  return max_value;
};

// console.log(
//   getMaxPathSum([
//     [1, 2, 10, 4],
//     [100, 3, 2, 1],
//     [1, 1, 20, 2],
//     [1, 2, 2, 1],
//   ])
// );

const maximumChocolatesMemoized = function (r, c, grid) {
  const recursiveCall = function (i, j1, j2, r, c, grid) {
    if (j1 < 0 || j2 < 0 || j1 >= c || j2 >= c) {
      return -Infinity;
    }
    if (i === r - 1) {
      if (j1 === j2) {
        return grid[i][j1];
      } else {
        return grid[i][j1] + grid[i][j2];
      }
    }
    // explore all paths of alice and bob simultenously
    let maxi = -Infinity;
    for (let dj1 = -1; dj1 <= 1; dj1++) {
      for (let dj2 = -1; dj2 <= 1; dj2++) {
        let value = 0;
        if (j1 === j2) {
          value = grid[i][j1];
        } else {
          value = grid[i][j1] + grid[i][j2];
        }
        value += recursiveCall(i + 1, j1 + dj1, j2 + dj2, r, c, grid);
        maxi = Math.max(maxi, value);
      }
    }
    return maxi;
  };

  return recursiveCall(0, 0, c - 1, r, c, grid);
};

const maximumChocolatesTabulation = function (r, c, grid) {
  const dp = Array.from({ length: r }, () => Array.from({ length: c }, () => Array(c).fill(0)));
  for (let j1 = 0; j1 < c; j1++) {
    for (let j2 = 0; j2 < c; j2++) {
      if (j1 === j2) {
        dp[n - 1][j1][j2] = grid[n - 1][j1];
      } else {
        dp[n - 1][j1][j2] = grid[n - 1][j1] + grid[n - 1][j2];
      }
    }
  }

  for (let i = r - 1; i >= 0; i--) {
    for (let j1 = 0; j1 < c; j1++) {
      for (let j2 = 0; j2 < c; j2++) {
        let max_value = -Infinity;
        for (let delta_j1 of [-1, 0, 1]) {
          for (let delta_j2 of [-1, 0, 1]) {
            let value = 0;
            if (j1 === j2) {
              value = grid[i][j1];
            } else {
              value = grid[i][j1] + grid[i][j2];
            }
            if (j1 + delta_j1 >= 0 && j1 + delta_j1 < c && j2 + delta_j2 >= 0 && j2 + delta_j2 < c) {
              value += dp[i + 1][j1 + delta_j1][j2 + delta_j2];
            } else {
              value += -Infinity;
            }
            max_value = Math.max(max_value, value);
          }
        }
        dp[i][j1][j2] = max_value;
      }
    }
  }
  return dp[0][0][c - 1];
};

// maximumChocolatesTabulation(4, 3, [[]]);

const subsetSumToKMemoized = function (n, k, arr) {
  const memo = Array.from({ length: n }, () => Array(k + 1).fill(-1));
  const recursiveCall = function (index, target, arr, memo) {
    if (target === 0) {
      return true;
    }
    if (index === 0) {
      return arr[0] === target;
    }
    if (memo[index][target] !== -1) {
      return memo[index][target];
    }
    const notTake = recursiveCall(index - 1, target, arr, memo);
    const take = false;
    if (a[index] <= target) {
      take = recursiveCall(index - 1, target - arr[index], arr, memo);
    }
    const result = take || notTake;
    memo[index][target] = result;
    return result;
  };

  return recursiveCall(n - 1, k, arr, memo);
};

const subsetSumToKTabulation = function (n, k, arr) {
  const dp = Array.from({ length: n }, () => Array(k + 1).fill(false));
  for (let i = 0; i < n; i++) {
    dp[i][0] = true;
  }

  if (arr[0] <= k) {
    dp[0][arr[0]] = true;
  }

  for (let index = 1; index < n; index++) {
    for (let target = 1; target <= k; target++) {
      const notTake = dp[index - 1][target];
      const take = false;
      if (arr[index] <= target) {
        take = dp[index - 1][target - arr[index]];
      }
      dp[index][target] = take || notTake;
    }
  }
  return dp[n - 1][k];
};

const minSubsetSumDifference = function (arr, n) {
  const subsetSumUtil = function (index, target, arr, dp) {
    if (target === 0) {
      return (dp[index][target] = true);
    }
    if (index === 0) {
      return (dp[index][target] = arr[0] === target);
    }
    if (dp[index][target] !== -1) {
      return dp[index][target];
    }
    const notTaken = subsetSumUtil(index - 1, target, arr, dp);

    let taken = false;
    if (arr[index] <= target) {
      taken = subsetSumUtil(index - 1, target - arr[index], arr, dp);
    }

    return (dp[index][target] = notTaken || taken);
  };
  let totalSum = 0;
  for (let i = 0; i < n; i++) {
    totalSum += arr[i];
  }

  const dp = new Array(n);
  for (let i = 0; i < n; i++) {
    dp[i] = new Array(totalSum + 1).fill(-1);
  }
  for (let i = 0; i <= totalSum; i++) {
    const dummy = subsetSumUtil(n - 1, i, arr, dp);
  }

  let min_value = Infinity;
  for (let i = 0; i <= totalSum; i++) {
    if (dp[n - 1][i] === true) {
      const diff = Math.abs(i - (totalSum - i));
      min_value = Math.min(min_value, diff);
    }
  }
  return min_value;
};

const findWays = function (num, k) {
  const findWaysUtil = function (index, target, arr, dp) {
    if (target === 0) {
      return 1;
    }
    if (index === 0) {
      return arr[0] === target ? 1 : 0;
    }

    if (dp[index][target] !== -1) {
      return dp[index][target];
    }
    const notTaken = findWaysUtil(index - 1, target, arr, dp);

    let taken = 0;
    if (arr[index] <= target) {
      taken = findWaysUtil(index - 1, target - arr[index], arr, dp);
    }

    return (dp[index][target] = notTaken + taken);
  };

  const n = num.length;
  const dp = new Array(n);
  for (let i = 0; i < n; i++) {
    dp[i] = new Array(k + 1).fill(-1);
  }

  return findWaysUtil(n - 1, k, num, dp);
};
