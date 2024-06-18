// ================== START OF SOLUTION FOR QUESTION 1 ========================

// given an m by n 2d grid map of 1's (land) and 0's (water), return the number of islands. An island is surrounded by water and is formed by connecting adjacent
// lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water

const getAdjacentNeighbors = function (i, j, grid, visited) {
  const adjacentNeighbors = [];
  if (i > 0 && !visited[i - 1][j]) {
    adjacentNeighbors.push([i - 1, j]);
  }
  if (i < grid.length - 1 && !visited[i + 1][j]) {
    adjacentNeighbors.push([i + 1, j]);
  }
  if (j > 0 && !visited[i][j - 1]) {
    adjacentNeighbors.push([i, j - 1]);
  }
  if (j < grid[0].length - 1 && !visited[i][j + 1]) {
    adjacentNeighbors.push([i, j + 1]);
  }

  return adjacentNeighbors;
};

const depthFirstSearch = function (i, j, grid, visited) {
  const stack = [[i, j]];
  let islandSize = 0;
  while (stack.length) {
    let currentNode = stack.pop();
    let [i, j] = currentNode;
    // check if visited at i and j
    if (visited[i][j]) {
      continue;
    }
    visited[i][j] = true;
    // check if cell is part of an island
    if (grid[i][j] === "0") {
      continue;
    }
    islandSize++;

    let adjacentNeighbors = getAdjacentNeighbors(i, j, grid, visited);

    stack.push(...adjacentNeighbors);
  }

  return islandSize > 0 ? true : false;
};

const numIslands = function (grid) {
  const visited = grid.map((row) => row.map((cell) => false));
  let islandCount = 0;
  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[i].length; j++) {
      if (depthFirstSearch(i, j, grid, visited)) {
        islandCount++;
      }
    }
  }

  return islandCount;
};

// console.log(
//   numIslands([
//     ["1", "1", "1", "1", "0"],
//     ["1", "1", "0", "1", "0"],
//     ["1", "1", "0", "0", "0"],
//     ["0", "0", "0", "0", "0"],
//   ])
// );
// console.log(
//   numIslands([
//     ["1", "1", "0", "0", "0"],
//     ["1", "1", "0", "0", "0"],
//     ["0", "0", "1", "0", "0"],
//     ["0", "0", "0", "1", "1"],
//   ])
// );

// ================== END OF SOLUTION FOR QUESTION 1 ========================

// ================== START OF SOLUTION FOR QUESTION 2 ========================

// Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

// An input string is valid if:

// Open brackets must be closed by the same type of brackets.
// Open brackets must be closed in the correct order.
// Every close bracket has a corresponding open bracket of the same type.

const isValidBrackets = function (s) {
  const openingBrackets = ["(", "{", "["];
  const closingBrackets = [")", "}", "]"];
  const openingBracketStack = [];
  let isValidBracketString = true;

  for (const [index, bracket] of Object.entries(s)) {
    if ((Number(index) === s.length - 1 && openingBrackets.includes(bracket)) || s.length === 1) {
      isValidBracketString = false;
      break;
    } else if (openingBrackets.includes(bracket)) {
      openingBracketStack.push(bracket);
    } else if (closingBrackets.includes(bracket) && openingBracketStack.at(-1) === openingBrackets.at(closingBrackets.indexOf(bracket))) {
      openingBracketStack.pop();
    } else {
      isValidBracketString = false;
      break;
    }
  }
  return isValidBracketString && openingBracketStack.length === 0 ? true : false;
};

// console.log(isValidBrackets("(("));
// isValidBrackets("((");

// alternative solution
const isValidBrackets2 = function (s) {
  const stack = [];
  const parens = "() {} []";
  let i = 0;
  while (i < s.length) {
    stack.push(s[i]);
    i++;
    let open = stack[stack.length - 2];
    let close = stack[stack.length - 1];

    let potentialParens = open + close;
    if (parens.includes(potentialParens)) {
      stack.pop();
      stack.pop();
    }
  }
  return stack.length === 0;
};

// console.log(isValidBrackets2("(("));

// ================== END OF SOLUTION FOR QUESTION 2 ==========================

// ================== START OF SOLUTION FOR QUESTION 3 ========================
// There are n cities. Some of them are connected, while some are not. If city a is connected directly with city b, and city b is connected directly with city c, then city a is connected indirectly with city c.

// A province is a group of directly or indirectly connected cities and no other cities outside of the group.

// You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and the jth city are directly connected, and isConnected[i][j] = 0 otherwise.

// Return the total number of provinces.

const getEdges = function (index, edge) {
  const edges = [];

  for (let i = 0; i < edge.length; i++) {
    if (index === i) {
      continue;
    }
    if (edge[i] === 0) {
      continue;
    }
    edges.push(i);
  }
  return edges;
};

const buildAdjacencyList = function (edges, n = edges.length) {
  const adjacencyList = Array.from({ length: n }, () => []);
  for (let i = 0; i < edges.length; i++) {
    adjacencyList[i].push(...getEdges(i, edges[i]));
  }
  return adjacencyList;
};

const depthFirstProvinceSearch = function (node, adjacencyList, visited) {
  visited[node] = true;
  for (let neighbor of adjacencyList[node]) {
    if (!visited[neighbor]) {
      visited[neighbor] = true;
      depthFirstProvinceSearch(neighbor, adjacencyList, visited);
    }
  }
};

const numProvinces = function (isConnected) {
  const adjacencyList = buildAdjacencyList(isConnected);
  const visited = {};
  let provinceCount = 0;
  for (let vertex = 0; vertex < adjacencyList.length; vertex++) {
    if (!visited[vertex]) {
      provinceCount++;
      depthFirstProvinceSearch(vertex, adjacencyList, visited);
    }
  }
  return provinceCount;
};

// console.log(
//   numProvinces([
//     [1, 1, 0],
//     [1, 1, 0],
//     [0, 0, 1],
//   ])
// );
// console.log(
//   numProvinces([
//     [1, 0, 0],
//     [0, 1, 0],
//     [0, 0, 1],
//   ])
// );

// ================== END OF SOLUTION FOR QUESTION 3 ==========================

// ================== START OF SOLUTION FOR 207. Course Schedule ==========================
// There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

// For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
// Return true if you can finish all courses. Otherwise, return false.

const buildCanFinishAdjacencyList = function (n, edges) {
  const adjacencyList = Array.from({ length: n }, () => []);
  for (let edge of edges) {
    let [source, destination] = edge;
    adjacencyList[source].push(destination);
  }
  return adjacencyList;
};

const hasCycleDFS = function (node, adjacencyList, visited, arrival, departure) {
  arrival[node]++;
  visited[node] = true;

  for (let neighbor of adjacencyList[node]) {
    if (!visited[neighbor]) {
      visited[neighbor] = true;
      if (hasCycleDFS(neighbor, adjacencyList, visited, arrival, departure)) {
        return true;
      }
    } else {
      if (departure[neighbor] === 0) {
        return true;
      }
    }
  }

  departure[node]++;
  return false;
};

const canFinish = function (numCourses, prerequisites) {
  const adjacencyList = buildCanFinishAdjacencyList(numCourses, prerequisites);
  const visited = {};
  const arrival = Array.from({ length: numCourses }, () => 0);
  const departure = Array.from({ length: numCourses }, () => 0);

  for (let vertex = 0; vertex < adjacencyList.length; vertex++) {
    if (!visited[vertex]) {
      if (hasCycleDFS(vertex, adjacencyList, visited, arrival, departure)) {
        return false;
      }
    }
  }

  return true;
};
// console.log(canFinish(2, [[1, 0]]));
// console.log(
//   canFinish(5, [
//     [1, 4],
//     [2, 4],
//     [3, 1],
//     [3, 2],
//   ])
// );
// console.log(
//   canFinish(2, [
//     [1, 0],
//     [0, 1],
//   ])
// );

// ================== END OF SOLUTION FOR 207. Course Schedule ==========================

// ================== START OF SOLUTION FOR 210. Course Schedule II ==========================
// There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi]
// indicates that you must take course bi first if you want to take course ai. For example, the pair [0, 1], indicates that to take course 0 you have to first take
// course 1. Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them.
// If it is impossible to finish all courses, return an empty array.

// ================== END OF SOLUTION FOR 210. Course Schedule II ==========================

// ================== START OF SOLUTION FOR 5. Longest Palindromic Substring ==========================
// Given a string s, return the longest palindromic substring in s.

class LongestPalindromicSubstringSolution {
  longestPalidromicString(string) {
    let result = "";
    for (let i = 0; i < string.length; i++) {
      // if odd string
      let odd = this.helper(string, i, i);
      let even = this.helper(string, i, i + 1);

      result = [odd, even, result].reduce((maxValue, currentValue) => (currentValue.length > maxValue.length ? currentValue : maxValue));
    }
    return result;
  }
  helper(string, left, right) {
    while (left >= 0 && right < string.length && string[left] === string[right]) {
      left--;
      right++;
    }
    return string.slice(left + 1, right);
  }
}

// const newSolution = new LongestPalindromicSubstringSolution();
// console.log(newSolution.longestPalidromicString("babad"));

// ================== END OF SOLUTION FOR 5. Longest Palindromic Substring ==========================

// ================== START OF SOLUTION FOR 56. Merge Intervals ==========================
// Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that
// cover all the intervals in the input.
const merge = function (intervals) {
  intervals.sort((a, b) => a[0] - b[0]);
  let i = 1;
  while (i < intervals.length) {
    if (intervals[i][0] <= intervals[i - 1][1]) {
      intervals[i - 1][0] = Math.min(intervals[i - 1][0], intervals[i][0]);
      intervals[i - 1][1] = Math.max(intervals[i - 1][1], intervals[i][1]);
      intervals.splice(i, 1);
    } else {
      i++;
    }
  }
  return intervals;
};

// console.log(
//   merge([
//     [1, 3],
//     [2, 6],
//     [8, 10],
//     [15, 18],
//   ])
// );

// ================== END OF SOLUTION FOR 56. Merge Intervals ==========================

// ================== START OF SOLUTION FOR 49. Group Anagrams ==========================
// Given an array of strings strs, group the anagrams together. You can return the answer in any order. An Anagram is a word or phrase formed by rearranging the
// letters of a different word or phrase, typically using all the original letters exactly once.
const groupAnagrams = function (strs) {
  const anagramsDict = {};
  for (const item of strs) {
    let sortedItem = item.split("").sort().join("");
    if (anagramsDict[sortedItem] !== undefined) {
      anagramsDict[sortedItem].push(item);
    } else {
      anagramsDict[sortedItem] = [item];
    }
  }
  return Object.values(anagramsDict);
};

// console.log(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]));

// ================== END OF SOLUTION FOR 49. Group Anagrams ==========================

// ================== START OF SOLUTION FOR 1143. Longest Common Subsequence ==========================
// Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.
// A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative
// order of the remaining characters.
// For example, "ace" is a subsequence of "abcde". A common subsequence of two strings is a subsequence that is common to both strings.
const longestCommonSubsequence = function (text1, text2) {
  if (text1.length === 0 || text2.length === 0) {
    return 0;
  }
  if (text1 === text2) {
    return text1.length;
  }
  let short_text;
  let long_text = [];
  let common_subsequence = "";
  text1.length > text2.length ? ((short_text = text2), (long_text = text1.split(""))) : ((short_text = text1), (long_text = text2.split("")));
  let currentValidIndex = 0;
  for (let i = 0; i < short_text.length; i++) {
    const index = long_text.indexOf(short_text[i]);
    if (index !== -1 && index > currentValidIndex) {
      common_subsequence += short_text[i];
      long_text.splice(index, 1);
      currentValidIndex = index;
    }
  }
  return common_subsequence.length;
};

// console.log(longestCommonSubsequence("abcde", "ace"));
// console.log(longestCommonSubsequence("ezupkr", "ubmrapg"));

// ================== END OF SOLUTION FOR 1143. Longest Common Subsequence ==========================

// ================== START OF SOLUTION FOR 1. Two Sum ==========================

const twoSum = function (nums, target) {
  const myDict = {};

  for (let i = 0; i < nums.length; i++) {
    let currentDifference = target - nums[i];
    if (myDict[currentDifference] !== undefined) {
      return [myDict[currentDifference], i];
    } else {
      myDict[nums[i]] = i;
    }
  }
};

// console.log(twoSum([2, 7, 11, 15], 9));
// console.log(twoSum([3, 2, 4], 6));

// ================== END OF SOLUTION FOR 1. Two Sum ==========================

// ================== START OF SOLUTION FOR 438. Find All Anagrams in a String ==========================
const findAnagrams = function (s, p) {};

// console.log(findAnagrams("cbaebabacd", "abc"));

// ================== END OF SOLUTION FOR 438. Find All Anagrams in a String ==========================

// ================== START OF SOLUTION FOR 528. Random Pick with Weight ==========================
const RandomPickSolution = function (w) {
  this.prefix_sum = 0;
  this.prefix_sum_array = [];
  for (const weight of w) {
    this.prefix_sum += weight;
    this.prefix_sum_array.push(this.prefix_sum);
  }
  this.total_sum = this.prefix_sum;
};

RandomPickSolution.prototype.pickIndex = function () {
  const random_num = this.total_sum * Math.random();
  let [low, high] = [0, this.prefix_sum];
  while (low < high) {
    const mid = low + Math.floor((high - low) / 2);
    if (random_num > this.prefix_sum_array[mid]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
};

// const new_solution = new RandomPickSolution([[[1]], []]);
// const new_solution = new RandomPickSolution([[[1, 3]], [], [], [], [], []]);
// console.log(new_solution.pickIndex());

// ================== END OF SOLUTION FOR 528. Random Pick with Weight ==========================

const coinChange = function (coins, amount) {
  const amountsArray = new Array(amount + 1).fill(amount + 1);
  amountsArray[0] = 0;
  for (const coin of coins) {
    for (let i = coin; i <= amount; i++) {
      amountsArray[i] = Math.min(amountsArray[i], amountsArray[i - coin] + 1);
    }
  }
  if (amountsArray[amount] == amount + 1) {
    return -1;
  } else {
    return amountsArray[amount];
  }
};

// console.log(coinChange([1, 2, 5], 11));
// console.log(coinChange([2], 3));
// console.log(coinChange([186, 419, 83, 408], 6249));

const maxProduct = function (nums) {
  const numsLength = nums.length;
  if (numsLength === 0) {
    return 0;
  }
  if (numsLength === 1) {
    return nums[0];
  }
  let maxValue = nums[0];
  let runningMax = nums[0];
  let runningMin = nums[0];
  for (let i = 1; i < numsLength; i++) {
    if (nums[i] < 0) {
      const currentMax = runningMax;
      runningMax = runningMin;
      runningMin = currentMax;
    }

    runningMax = Math.max(runningMax * nums[i], nums[i]);
    runningMin = Math.min(runningMin * nums[i], nums[i]);
    maxValue = Math.max(maxValue, runningMax);
  }
  return maxValue;
};

// console.log(maxProduct([2, 3, -2, 4]));
// console.log(maxProduct([-2, 3, -4]));

const minSubArrayLen = function (target, nums) {
  const numsLength = nums.length;
  if (numsLength === 0) {
    return 0;
  }
  let minimumLength = Infinity;
  let startIndex = 0;
  let runningSum = 0;
  for (let i = startIndex; i < numsLength; i++) {
    runningSum += nums[i];
    while (runningSum >= target) {
      minimumLength = Math.min(minimumLength, i - startIndex + 1);
      runningSum -= nums[startIndex];
      startIndex++;
    }
  }
  return minimumLength === Infinity ? 0 : minimumLength;
};

// console.log(minSubArrayLen(7, [2, 3, 1, 2, 4, 3]));
// console.log(minSubArrayLen(4, [1, 4, 4]));

const characterReplacement = function (s, k) {
  let currentSubstring = s[0];
  let startIndex = 0;

  let runningSubstring = s[0];
  let runningIndex = 0;

  for (let i = 1; i < s.length; i++) {
    if (s[i] === s[i - 1]) {
      runningSubstring += s[i];
    } else {
      runningSubstring = s[i];
      runningIndex = i;
    }
    if (runningSubstring.length > currentSubstring.length) {
      currentSubstring = runningSubstring;
      startIndex = runningIndex;
    }
  }
  let longestSubstring = currentSubstring;
  let currentIndex = 0;
  console.log(longestSubstring);
  for (let j = 0; j < k; j++) {
    while (s[currentIndex] === currentSubstring[0]) {
      if (currentIndex + 1 < s.length) {
        currentIndex++;
        continue;
      } else if (currentIndex - 1 >= 0) {
        currentIndex--;
        continue;
      }
    }
    longestSubstring += currentSubstring[0];
  }
  // return longestSubstring;
};

// console.log(characterReplacement("ABAB", 2));

const numSubarrayProductLessThanK = function (nums, k) {
  const numsLength = nums.length;
  if (k < 1) {
    return 0;
  }
  let startIndex = 0;
  let numSubarrays = 0;
  let runningProduct = 1;

  for (let endIndex = 0; endIndex < numsLength; endIndex++) {
    runningProduct *= nums[endIndex];
    while (runningProduct >= k) {
      runningProduct /= nums[startIndex];
      startIndex++;
    }
    numSubarrays += endIndex - startIndex + 1;
  }
  return numSubarrays;
};

// console.log(numSubarrayProductLessThanK([10, 5, 2, 6], 100));

const sortColors = function (nums) {
  const numsLength = nums.length;
  let lowIndex = 0;
  let midIndex = 0;
  let highIndex = numsLength - 1;

  while (midIndex < highIndex) {
    if (nums[midIndex] === 0) {
      [nums[lowIndex], nums[midIndex]] = [nums[midIndex], nums[lowIndex]];
      lowIndex++;
      midIndex++;
    } else if (nums[midIndex] === 1) {
      midIndex++;
    } else if (nums[midIndex] === 2) {
      [nums[midIndex], nums[highIndex]] = [nums[highIndex], nums[midIndex]];
      highIndex--;
    }
  }
  return nums;
};

// console.log(sortColors([2, 0, 2, 1, 1, 0]));

const maxArea = function (height) {
  const length = height.length;
  let startIndex = 0;
  let endIndex = length - 1;
  let height1 = height[startIndex];
  let height2 = height[endIndex];
  let maxAreaComputed = Math.min(height1, height2) * Math.abs(endIndex - startIndex);

  while (startIndex < endIndex) {
    const currentArea = Math.min(height1, height2) * Math.abs(endIndex - startIndex);
    if (currentArea > maxAreaComputed) {
      maxAreaComputed = currentArea;
    }
    if (height1 > height2) {
      endIndex--;
    } else {
      startIndex++;
    }
    height1 = height[startIndex];
    height2 = height[endIndex];
  }
  return maxAreaComputed;
};

// console.log(maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]));

const findTargetSumWays = function (nums, target) {
  const numsLength = nums.length;
  let totalSum = nums.reduce((acc, current) => acc + current, 0);
  const arr = Array.from({ length: numsLength + 1 }, () => Array(2 * totalSum + 1).fill(0));
  arr[0][totalSum] = 1;
  for (let i = 1; i <= numsLength; i++) {
    for (let sum = -totalSum; sum <= totalSum; sum++) {
      if (arr[i - 1][sum + totalSum] !== 0) {
        arr[i][sum + nums[i - 1] + totalSum] += arr[i - 1][sum + totalSum];
        arr[i][sum - nums[i - 1] + totalSum] += arr[i - 1][sum + totalSum];
      }
    }
  }
  return target > totalSum || target < -totalSum ? 0 : arr[numsLength][target + totalSum];
};

// console.log(findTargetSumWays([1, 1, 1, 1, 1], 3));

const removeDuplicates = function (arr) {
  if (arr.length === 0) {
    return 0;
  }
  let i = 0;
  const new_arr = [];
  for (let j = 0; j < arr.length; j++) {
    if (arr[j] !== arr[i]) {
      i += 1;
      arr[i] = arr[j];
    }
  }
  return i + 1;
};

console.log(removeDuplicates([1, 1, 2, 2, 3, 4, 4]));
