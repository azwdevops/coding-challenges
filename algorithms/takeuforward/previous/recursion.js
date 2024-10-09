// function to print all the subsequences of an array

const subsequenceRecursion = function (index, arr, arr_length, currentSubsequence, result) {
  if (index === arr_length) {
    result.push(currentSubsequence.slice());
    return;
  }
  // take or pick the particular index into the subsequence
  currentSubsequence.push(arr[index]);
  subsequenceRecursion(index + 1, arr, arr_length, currentSubsequence, result);
  currentSubsequence.pop();

  // not pick or not take condition, this element is not added to your subsequence
  subsequenceRecursion(index + 1, arr, arr_length, currentSubsequence, result);
};
// const result = [];
// subsequenceRecursion(0, [3, 1, 2], 3, [], result);
// console.log(result);

// function to print subsequences that total to a given sum
const subsequenceRecursionSum = function (index, result, currentSubsequence, currentSum, neededSum, arr, n = arr.length) {
  if (index === n) {
    if (currentSum === neededSum) {
      result.push(currentSubsequence.slice());
    }
    return;
  }

  // pick condition, i.e means include the value at arr[index]
  currentSubsequence.push(arr[index]);
  currentSum += arr[index];

  subsequenceRecursionSum(index + 1, result, currentSubsequence, currentSum, neededSum, arr);

  // not pick condition i.e means do not include the value at arr[index]

  currentSum -= arr[index];
  currentSubsequence.pop();
  subsequenceRecursionSum(index + 1, result, currentSubsequence, currentSum, neededSum, arr);
};

// const result = [];
// subsequenceRecursionSum(0, result, [], 0, 2, [1, 2, 1]);
// console.log(result);

// function to check if any subsequences can total to a given sum
const subsequenceRecursionSumPossible = function (index, currentSum, neededSum, arr, n = arr.length) {
  if (index === n) {
    if (currentSum === neededSum) {
      return true;
    }
    return false;
  }

  // pick condition, i.e means include the value at arr[index]
  currentSum += arr[index];

  if (subsequenceRecursionSumPossible(index + 1, currentSum, neededSum, arr)) {
    return true;
  }

  // not pick condition i.e means do not include the value at arr[index]

  currentSum -= arr[index];
  if (subsequenceRecursionSumPossible(index + 1, currentSum, neededSum, arr)) {
    return true;
  }
  return false;
};

// console.log(subsequenceRecursionSumPossible(0, 0, 2, [1, 2, 1]));

// function to count all subsequences that total to a given sum
const subsequenceRecursionSumCount = function (index, currentSum, neededSum, arr, n = arr.length) {
  if (currentSum > neededSum) {
    return 0;
  }
  if (index === n) {
    if (currentSum === neededSum) {
      return 1;
    }
    return 0;
  }

  // pick condition, i.e means include the value at arr[index]
  currentSum += arr[index];

  const valuePicked = subsequenceRecursionSumCount(index + 1, currentSum, neededSum, arr);

  // not pick condition i.e means do not include the value at arr[index]

  currentSum -= arr[index];
  const valueNotPicked = subsequenceRecursionSumCount(index + 1, currentSum, neededSum, arr);
  return valuePicked + valueNotPicked;
};

// console.log(subsequenceRecursionSumCount(0, 0, 2, [1, 2, 1]));

class SolutionMergeSort {
  mergeSortHelper(arr, low, high) {
    if (low === high) {
      return;
    }
    const mid = Math.floor((low + high) / 2);
    this.mergeSortHelper(arr, low, mid);
    this.mergeSortHelper(arr, mid + 1, high);
    this.merge(arr, low, mid, high);
  }

  merge(arr, low, mid, high) {
    const temp = [];
    const left = low;
    const right = mid + 1;

    while (left <= mid && right <= high) {
      if (arr[left] <= arr[right]) {
        temp.push(arr[left]);
        left++;
      } else {
        temp.push(arr[right]);
        right++;
      }
    }
    while (left <= mid) {
      temp.push(arr[left]);
      left++;
    }
    while (right <= high) {
      temp.push(arr[right]);
      right++;
    }
    for (let i = low; i <= high; i++) {
      arr[i] = temp[i - low];
    }
  }
  mergeSort(arr, n = arr.length) {
    this.mergeSortHelper(arr, 0, n - 1);
  }
}

class SolutionQuickSort {
  getPivotIndex(arr, low, high) {
    const pivot = arr[low];
    let i = low;
    let j = high;
    while (i < j) {
      while (i < high && arr[i] <= pivot) {
        i++;
      }
      while (j > low && arr[j] > pivot) {
        j--;
      }
      if (i < j) {
        // swap the values
        const temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
    const temp1 = arr[low];
    arr[low] = arr[j];
    arr[j] = temp1;
    return j;
  }
  quickSortHelper(arr, low, high) {
    if (low < high) {
      const pivotIndex = this.getPivotIndex(arr, low, high);
      this.quickSortHelper(arr, low, pivotIndex - 1);
      this.quickSortHelper(arr, pivotIndex + 1, high);
    }
  }
  quickSort(arr) {
    this.quickSortHelper(arr, 0, arr.length - 1);
    return arr;
  }
}

class SolutionTargetSumCombinations {
  findCombinations(index, arr, target, result, currentSubsequence) {
    if (index === arr.length) {
      if (target === 0) {
        result.push(currentSubsequence.slice());
      }
      return;
    }
    if (arr[index] <= target) {
      currentSubsequence.push(arr[index]);
      this.findCombinations(index, arr, target - arr[index], result, currentSubsequence);
      currentSubsequence.pop();
    }
    this.findCombinations(index + 1, arr, target, result, currentSubsequence);
  }

  combinationSum(candidates, target) {
    const result = [];
    this.findCombinations(0, candidates, target, result, []);
    return result;
  }
}

// const solution = new SolutionTargetSumCombinations();
// console.log(solution.combinationSum([2, 3, 6, 7], 7));

class SolutionSubsetSum {
  helper(index, sum, arr, n, result) {
    if (index === n) {
      result.push(sum);
      return;
    }
    // pick the element
    this.helper(index + 1, sum + arr[index], arr, n, result);
    // do not pick the element
    this.helper(index + 1, sum, arr, n, result);
  }
  subsetSums(arr, n) {
    const result = [];
    this.helper(0, 0, arr, n, result);
    result.sort((a, b) => a - b);
    return result;
  }
}

// const solution = new SolutionSubsetSum();
// console.log(solution.subsetSums([1, 2, 1], 3));

// get subsets without duplicates when given an array nums that contains duplicate values
class SolutionSubsetsWithDuplicates {
  findSubsets(index, nums, currentSubset, result) {
    result.push(currentSubset.slice());
    for (let i = index; i < nums.length; i++) {
      if (i != index && nums[i] == nums[i - 1]) {
        continue;
      }
      currentSubset.push(nums[i]);
      this.findSubsets(i + 1, nums, currentSubset, result);
      currentSubset.pop();
    }
  }

  subsetsWithDuplicates(nums) {
    nums.sort((a, b) => a - b);
    const result = [];
    this.findSubsets(0, nums, [], result);

    return result;
  }
}

// const solution = new SolutionSubsetsWithDuplicates();
// console.log(solution.subsetsWithDuplicates([1, 2, 2]));

// get all the possible permutations using visited array
class SolutionPermutations {
  recursionPermute(nums, currentPermutation, result, visited) {
    if (currentPermutation?.length === nums.length) {
      result.push(currentPermutation.slice());
      return;
    }
    for (let i = 0; i < nums.length; i++) {
      if (!visited[i]) {
        visited[i] = true;
        currentPermutation.push(nums[i]);
        this.recursionPermute(nums, currentPermutation, result, visited);
        currentPermutation.pop();
        visited[i] = false;
      }
    }
  }

  permute(nums) {
    const result = [];
    const visited = Array.from({ length: nums.length }).fill(false);
    this.recursionPermute(nums, [], result, visited);
    return result;
  }
}

// const solution = new SolutionPermutations();
// console.log(solution.permute([1, 2, 3]));

// get all the possible permutations using swapping method
class SolutionPermutations2 {
  recursionPermute(index, nums, result) {
    if (index === nums.length) {
      const currentPermutation = nums.slice();
      result.push(currentPermutation);
      return;
    }
    for (let i = index; i < nums.length; i++) {
      this.swap(i, index, nums);
      this.recursionPermute(index + 1, nums, result);
      this.swap(i, index, nums);
    }
  }
  swap(i, j, nums) {
    const temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
  }

  permute(nums) {
    const result = [];
    this.recursionPermute(0, nums, result);
    return result;
  }
}

// const solution = new SolutionPermutations2();
// console.log(solution.permute([1, 2, 3]));

class SolutionSolveNQueens {
  isSafe(row, col, board, n) {
    // check the upper diagonal
    const oldRow = row;
    const oldCol = col;
    while (row >= 0 && col >= 0) {
      if (board[row][col] === "Q") {
        return false;
      }
      row--;
      col--;
    }
    // check the left side
    row = oldRow;
    col = oldCol;
    while (col >= 0) {
      if (board[row][col] === "Q") {
        return false;
      }
      col--;
    }
    // check lower diagonal
    row = oldRow;
    col = oldCol;
    while (row < n && col >= 0) {
      if (board[row][col] === "Q") {
        return false;
      }
      row++;
      col--;
    }
    return true;
  }

  solve(col, board, result, n) {
    // if all queens are placed, add the solution
    if (col === n) {
      // board representation as a string
      const currentBoardForm = [];
      for (let row of board) {
        currentBoardForm.push(`${row.join("")}`);
      }
      result.push(currentBoardForm);
    }
    // try placing a queen in every row in the current column
    for (let row = 0; row < n; row++) {
      if (this.isSafe(row, col, board, n)) {
        board[row][col] = "Q"; // place the queen
        this.solve(col + 1, board, result, n); // recurse for the next column
        board[row][col] = "."; // backtrack
      }
    }
  }
  solveNQueens(n) {
    const result = [];
    const board = Array.from({ length: n }, () => Array(n).fill("."));
    this.solve(0, board, result, n);

    return result;
  }
}

// const solution = new SolutionSolveNQueens();
// console.log(solution.solveNQueens(4));

class SolutionGraphColoring {
  isSafe(node, colors, graph, V, nodeColor) {
    for (let i = 0; i < V; i++) {
      if (i !== node && graph[i][node] === 1 && colors[i] === nodeColor) {
        return false;
      }
    }
    return true;
  }
  solveColors(node, colors, k, V, graph) {
    if (node === V) {
      return true;
    }
    for (let colorValue = 1; colorValue <= k; colorValue++) {
      if (this.isSafe(node, colors, graph, V, colorValue)) {
        colors[node] = colorValue;
        if (this.solveColors(node + 1, colors, k, V, graph)) {
          return true;
        }
        colors[node] = 0;
      }
    }
    return false;
  }
  graphColoring(graph, k, V) {
    const colors = Array(V).fill(0);
    return this.solveColors(0, colors, k, V, graph);
  }
}

// const solution = new SolutionGraphColoring();
// console.log(
//   solution.graphColoring(
//     [
//       [0, 1, 1, 1],
//       [1, 0, 1, 0],
//       [1, 1, 0, 1],
//       [1, 0, 1, 0],
//     ],
//     3,
//     4
//   )
// );
// console.log(
//   solution.graphColoring(
//     [
//       [0, 1, 1],
//       [1, 0, 1],
//       [1, 1, 0],
//     ],
//     2,
//     3
//   )
// );

class SolutionPalidromePartition {
  partition(s) {
    const result = [];
    const path = [];
    this.partitionHelper(0, s, path, result);
    return result;
  }

  partitionHelper(startIndex, s, path, result) {
    if (startIndex === s.length) {
      result.push(path.slice());
      return;
    }
    for (let i = startIndex; i < s.length; i++) {
      if (this.isPalidrome(s, startIndex, i)) {
        path.push(s.slice(startIndex, i + 1));
        this.partitionHelper(i + 1, s, path, result);
        path.pop();
      }
    }
  }
  isPalidrome(s, start, end) {
    while (start <= end) {
      if (s.charAt(start++) != s.charAt(end--)) {
        return false;
      }
    }
    return true;
  }
}

// const solution = new SolutionPalidromePartition();
// console.log(solution.partition("aab"));

class SolutionFindPath {
  findPathHelper(i, j, grid, n, result, move, visited) {
    visited[i][j] = 1;
    if (i === n - 1 && j === n - 1) {
      result.push(move.slice());
      return;
    }
    // downward move
    if (i + 1 < n && !visited[i + 1][j] && grid[i + 1][j] === 1) {
      visited[i + 1][j] = 1;
      this.findPathHelper(i + 1, j, grid, n, result, move + "D", visited);
      visited[i + 1][j] = 0;
    }
    // left move
    if (j - 1 >= 0 && !visited[i][j - 1] && grid[i][j - 1] === 1) {
      visited[i][j - 1] = 1;
      this.findPathHelper(i, j - 1, grid, n, result, move + "L", visited);
      visited[i][j - 1] = 0;
    }

    // right move
    if (j + 1 < n && !visited[i][j + 1] && grid[i][j + 1] === 1) {
      visited[i][j + 1] = 1;
      this.findPathHelper(i, j + 1, grid, n, result, move + "R", visited);
      visited[i][j + 1] = 0;
    }
    // upward move
    if (i - 1 >= 0 && !visited[i - 1][j] && grid[i - 1][j] === 1) {
      visited[i - 1][j] = 1;
      this.findPathHelper(i - 1, j, grid, n, result, move + "U", visited);
      visited[i - 1][j] = 0;
    }
  }
  findPath(grid) {
    const n = grid.length;
    const result = [];
    const visited = Array.from({ length: n }, () => Array(n).fill(0));
    if (grid[0][0] === 0) {
      return [];
    }
    this.findPathHelper(0, 0, grid, n, result, "", visited);
    return result;
  }
}

// const solution = new SolutionFindPath();
// console.log(
//   solution.findPath([
//     [1, 0, 0, 0],
//     [1, 1, 0, 1],
//     [1, 1, 0, 0],
//     [0, 1, 1, 1],
//   ])
// );

class SolutionKthPermutation {
  getPermutation(n, k) {
    let fact = 1; // factorial
    const numbers = [];
    // calculate the (n-1)! and populate numbers array with [1,2,....n-1]
    for (let i = 1; i < n; i++) {
      fact = fact * i;
      numbers.push(i);
    }
    numbers.push(n);
    // push n into the numbers array
    let result = "";
    k = k - 1; // make k zero-based index

    // generate the k-th permutations
    while (true) {
      result += numbers[Math.floor(k / fact)].toString(); // pick element at k / fact index
      numbers.splice(Math.floor(k / fact), 1); // remove the used number

      if (numbers.length === 0) {
        break;
      }
      k = k % fact; // update k
      fact = fact / numbers.length; // update factorial for remaining numbers
    }
    return result;
  }
}

// const solution = new SolutionKthPermutation();
// console.log(solution.getPermutation(3, 3));
// console.log(solution.getPermutation(4, 9));

class SolutionNumberOfInversions {
  merge(arr, low, mid, high) {
    let temp = [];
    let left = low;
    let right = mid + 1;
    let count = 0;

    while (left <= mid && right <= high) {
      if (arr[left] <= arr[right]) {
        temp.push(arr[left]);
        left++;
      } else {
        temp.push(arr[right]);
        count += mid - left + 1;
        right++;
      }
    }
    while (left <= mid) {
      temp.push(arr[left]);
      left++;
    }
    while (right <= high) {
      temp.push(arr[right]);
      right++;
    }
    for (let i = low; i <= high; i++) {
      arr[i] = temp[i - low];
    }
    return count;
  }
  mergeSort(arr, low, high) {
    let count = 0;
    if (low >= high) {
      return count;
    }
    let mid = Math.floor((low + high) / 2);
    count += this.mergeSort(arr, low, mid);
    count += this.mergeSort(arr, mid + 1, high);
    count += this.merge(arr, low, mid, high);
    return count;
  }
  numberOfInversions(arr) {
    const n = arr.length;
    return this.mergeSort(arr, 0, n - 1);
  }
}

// const solution = new SolutionNumberOfInversions();
// console.log(solution.numberOfInversions([5, 3, 2, 1, 4]));
