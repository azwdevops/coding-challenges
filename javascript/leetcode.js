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

// ================== START OF SOLUTION FOR QUESTION 3 ==========================
// There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

// For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
// Return true if you can finish all courses. Otherwise, return false.

const canFinish = function (numCourses, prerequisites) {};
// console.log(canFinish(2, [[1, 0]]));
console.log(
  canFinish(5, [
    [1, 4],
    [2, 4],
    [3, 1],
    [3, 2],
  ])
);
// console.log(
//   canFinish(2, [
//     [1, 0],
//     [0, 1],
//   ])
// );

// ================== END OF SOLUTION FOR QUESTION 3 ==========================
