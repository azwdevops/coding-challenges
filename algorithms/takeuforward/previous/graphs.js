"use strict";

class SolutionNumberOfEnclaves {
  numberOfEnclaves(grid) {
    const queue = [];
    const n = grid.length;
    const m = grid[0].length;
    const visited = Array.from({ length: n }, () => Array.from({ length: m }, () => 0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        // first row, first col, last row, last col
        if (i === 0 || j === 0 || i === n - 1 || j === m - 1) {
          if (grid[i][j] === 1) {
            queue.push([i, j]);
            visited[i][j] = 1;
          }
        }
      }
    }
    const directions = [
      [0, 1],
      [0, -1],
      [1, 0],
      [-1, 0],
    ];
    while (queue.length > 0) {
      const [row, col] = queue.shift();
      for (const [deltaRow, deltaCol] of directions) {
        const nrow = row + deltaRow;
        const ncol = col + deltaCol;
        if (nrow >= 0 && nrow < n && ncol >= 0 && ncol < m && visited[nrow][ncol] === 0 && grid[nrow][ncol] === 1) {
          queue.push([nrow, ncol]);
          visited[nrow][ncol] = 1;
        }
      }
    }
    let count = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        if (grid[i][j] === 1 && visited[i][j] === 0) {
          count++;
        }
      }
    }
    return count;
  }
}

// const solution = new SolutionNumberOfEnclaves();
// console.log(
//   solution.numberOfEnclaves([
//     [0, 0, 0, 0],
//     [1, 0, 1, 0],
//     [0, 1, 1, 0],
//     [0, 0, 0, 0],
//   ])
// );

class SolutionCountDistinctIslands {
  countDistinctIslands(grid) {
    const n = grid.length;
    const m = grid[0].length;
    const visited = Array.from({ length: n }, () => Array(m).fill(0));
    const set = new Set();
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        if (!visited[i][j] && grid[i][j] === 1) {
          const vector = [];
          this.dfs(i, j, grid, visited, vector, i, j, n, m);
          set.add(vector);
        }
      }
    }
    return set.size;
  }

  dfs(row, col, grid, visited, vector, row0, col0, n, m) {
    visited[row][col] = 1;
    vector.push([row - row0, col - col0]);
    const directions = [
      [0, 1],
      [0, -1],
      [1, 0],
      [-1, 0],
    ];
    for (const [deltaRow, deltaCol] of directions) {
      const nrow = row + deltaRow;
      const ncol = col + deltaCol;
      if (nrow >= 0 && nrow < n && ncol >= 0 && ncol < m && visited[nrow][ncol] === 0 && grid[nrow][ncol] === 1) {
        this.dfs(nrow, ncol, grid, visited, vector, row0, col0, n, m);
      }
    }
  }
}

// const solution = new SolutionCountDistinctIslands();
// console.log(
//   solution.countDistinctIslands([
//     [1, 1, 0, 0, 0],
//     [1, 1, 0, 0, 0],
//     [0, 0, 0, 1, 1],
//     [0, 0, 0, 1, 1],
//   ])
// );

class SolutionIsBipartiteBFS {
  // colors individual component
  check(start, adj, color) {
    const queue = [];
    queue.push(start);

    color[start] = 0;
    while (queue.length > 0) {
      const node = queue.shift();
      for (let neighbor of adj[node]) {
        // if neighbor is not colored, give the opposite color of the current node i.e if 0, give 1, if 1 give 0
        if (color[neighbor] === -1) {
          color[neighbor] = color[node] === 1 ? 0 : 1;
          queue.push(neighbor);
        }
        // check if neighbor has same color as node, meaning some other adj node already colored it
        else if (color[neighbor] === color[node]) {
          return false;
        }
      }
    }
    return true;
  }

  isBipartiteBFS(V, adj) {
    const color = [];
    for (let i = 0; i < V; i++) {
      color[i] = -1;
    }
    for (let i = 0; i < V; i++) {
      if (color[i] === -1) {
        if (this.check(i, adj, color) === false) {
          return false;
        }
      }
    }
    return true;
  }
}

class SolutionIsBipartiteDFS {
  dfs(node, nodeColor, color, adj) {
    color[node] = nodeColor;
    for (let neighbor of adj[node]) {
      if (color[neighbor] === -1) {
        color[neighbor] = nodeColor === 1 ? 0 : 1;
        if (this.dfs(neighbor, color[neighbor], color, adj) === false) {
          return false;
        }
      } else if (color[neighbor] === nodeColor) {
        return false;
      }
    }
    return true;
  }

  isBipartiteDFS(V, adj) {
    const color = [];
    for (let i = 0; i < V; i++) {
      color[i] = -1;
    }
    for (let i = 0; i < V; i++) {
      if (color[i] === -1) {
        if (this.dfs(i, 0, color, adj) === false) {
          return false;
        }
      }
    }
    return true;
  }
}

class SolutionIsCyclicDFS {
  dfsCheck(node, adj, visited, pathVisited) {
    visited[node] = 1;
    pathVisited[node] = 1;
    // traverse the adjacent nodes
    for (let neighbor of adj[node]) {
      // when the neighbor is not yet visited
      if (!visited[neighbor]) {
        if (this.dfsCheck(neighbor, adj, visited, pathVisited) === true) {
          return true;
        }
      }
      // if the node has been visited already, but it has to be visited on the same path
      else if (pathVisited[neighbor]) {
        return true;
      }
    }

    pathVisited[node] = 0;
    return false;
  }
  isCyclic(V, adj) {
    const visited = Array.from({ length: V }).fill(0);
    const pathVisited = Array.from({ length: V }).fill(0);
    for (let i = 0; i < V; i++) {
      if (!visited[i]) {
        if (this.dfsCheck(i, adj, visited, pathVisited) === true) {
          return true;
        }
      }
    }
    return false;
  }
}

class SolutionEventualSafeNode {
  dfsCheck(node, adj, visited, pathVisited, check) {
    visited[node] = 1;
    pathVisited[node] = 1;
    // traverse the neighbors
    for (let neighbor of adj[node]) {
      if (!visited[neighbor]) {
        if (this.dfsCheck(neighbor, adj, visited, pathVisited, check)) {
          check[node] = 0;
          return true;
        }
      } else if (pathVisited[neighbor]) {
        check[node] = 0;
        return true;
      }
    }
    check[node] = 1;
    pathVisited[node] = 0;
    return false;
  }
  eventualSafeNode(V, adj) {
    const visited = Array.from({ length: V }).fill(0);
    const pathVisited = Array.from({ length: V }).fill(0);
    const safeNodes = [];
    const check = Array.from({ length: V }).fill(0);
    for (let i = 0; i < V; i++) {
      if (!visited[i]) {
        this.dfsCheck(i, adj, visited, pathVisited, check);
      }
    }
    for (let i = 0; i < V; i++) {
      if (check[i] === 1) {
        safeNodes.push(i);
      }
    }
    return safeNodes;
  }
}

class SolutionTopoSortDFS {
  topoSort(V, adj) {
    const visited = Array.from({ length: V }).fill(0);
    const stack = [];
    for (let i = 0; i < V; i++) {
      if (!visited[i]) {
        this.dfs(i, visited, stack, adj);
      }
    }
    const result = [];
    while (stack.length) {
      result.push(stack.pop());
    }
    return result;
  }

  dfs(node, visited, stack, adj) {
    visited[node] = 1;
    for (let neighbor of adj[node]) {
      if (!visited[neighbor]) {
        this.dfs(neighbor, visited, stack, adj);
      }
    }
    stack.push(node);
  }
}

class SolutionTopoSortBFS {
  topoSort(V, adj) {
    const inDegree = Array(V).fill(0);
    const queue = [];
    for (let i = 0; i < V; i++) {
      for (let neighbor of adj[i]) {
        inDegree[neighbor]++;
      }
    }
    for (let i = 0; i < V; i++) {
      if (inDegree[i] === 0) {
        queue.push(i);
      }
    }
    const topo = [];
    while (queue.length) {
      const node = queue.shift();
      topo.push(node);
      // node is in you topo sort so remove it from the inDegree
      for (let neighbor of adj[node]) {
        inDegree[neighbor]--;
        if (inDegree[neighbor] === 0) {
          queue.push(neighbor);
        }
      }
    }
    return topo;
  }
}

class SolutionIsCyclic {
  isCyclic(V, adj) {
    const inDegree = Array(V).fill(0);
    for (let i = 0; i < V; i++) {
      for (let neighbor of adj[i]) {
        inDegree[neighbor]++;
      }
    }
    const queue = [];
    for (let i = 0; i < V; i++) {
      if (inDegree[i] === 0) {
        queue.push(i);
      }
    }
    let count = 0;
    while (queue.length > 0) {
      const node = queue.shift();
      count++;
      for (let neighbor of adj[node]) {
        inDegree[neighbor]--;
        if (inDegree[neighbor] === 0) {
          queue.push(neighbor);
        }
      }
    }
    return count === V;
  }
}

class SolutionTaskCompletePossible {
  isPossible(N, prerequisites) {
    const adjList = Array.from({ length: N }, () => []);
    for (let [task, prereq] of prerequisites) {
      adjList[task].push(prereq);
    }
    const inDegree = Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      for (let prereq of adjList[i]) {
        inDegree[prereq]++;
      }
    }
    const queue = [];
    for (let i = 0; i < N; i++) {
      if (inDegree[i] === 0) {
        queue.push(i);
      }
    }
    let count = 0;
    while (queue.length > 0) {
      const task = queue.shift();
      count++;
      for (let prereq of adjList[task]) {
        inDegree[prereq]--;
        if (inDegree[prereq] === 0) {
          queue.push(prereq);
        }
      }
    }
    return count === N;
  }
}

// const solution = new SolutionTaskCompletePossible();
// console.log(
//   solution.isPossible(4, [
//     [1, 0],
//     [2, 1],
//     [3, 2],
//   ])
// );
// console.log(
//   solution.isPossible(2, [
//     [1, 0],
//     [0, 1],
//   ])
// );

class SolutionTaskCompleteOrder {
  findOrder(n, m, prerequisites) {
    const adjList = Array.from({ length: n }, () => []);
    for (let [task, prereq] of prerequisites) {
      adjList[task].push(prereq);
    }
    const inDegree = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let prereq of adjList[i]) {
        inDegree[prereq]++;
      }
    }
    const queue = [];
    for (let i = 0; i < n; i++) {
      if (inDegree[i] === 0) {
        queue.push(i);
      }
    }
    const result = [];
    while (queue.length > 0) {
      const task = queue.shift();
      result.unshift(task);
      for (let prereq in adjList[task]) {
        inDegree[prereq]--;
        if (inDegree[prereq] === 0) {
          queue.push(prereq);
        }
      }
    }
    return result.length === n ? result : [];
  }
}

// const solution = new SolutionTaskCompleteOrder();
// console.log(solution.findOrder(2, 1, [[1, 0]]));

class SolutionEventualSafeNodeBFS {
  eventualSafeNodes(V, adj) {
    const adjReverse = Array.from({ length: V }, () => []);
    const inDegree = Array(V).fill(0);
    for (let i = 0; i < V; i++) {
      for (let neighbor of adj[i]) {
        adjReverse[neighbor].push(i);
        inDegree[i]++;
      }
    }
    const queue = [];
    const safeNodes = [];
    for (let i = 0; i < V; i++) {
      if (inDegree[i] === 0) {
        queue.push(i);
      }
    }
    while (queue.length > 0) {
      const node = queue.shift();
      safeNodes.push(node);
      for (let neighbor of adjReverse[node]) {
        inDegree[neighbor]--;
        if (inDegree[neighbor] === 0) {
          queue.push(neighbor);
        }
      }
    }
    return safeNodes.sort((a, b) => a - b);
  }
}

// const solution = new SolutionEventualSafeNodeBFS()
// console.log(solution.eventualSafeNodes())

class SolutionAlienFindOrder {
  findOrder(words, n, k) {
    const adjList = Array.from({ length: k }, () => []);

    for (let i = 0; i < n - 1; i++) {
      let s1 = words[i];
      let s2 = words[i + 1];
      let upper_length = Math.min(s1.length, s2.length);
      for (let ptr = 0; ptr < upper_length; ptr++) {
        if (s1[ptr] !== s2[ptr]) {
          adjList[s1[ptr].charCodeAt(0) - "a".charCodeAt(0)].push(s2[ptr].charCodeAt(0) - "a".charCodeAt(0)); // note we want to have zero index and 97 is the ascii value of 'a', since other letters will start from a onwards
          break;
        }
      }
    }
    const inDegree = Array(k).fill(0);
    for (let i = 0; i < k; i++) {
      for (let neighbor of adjList[i]) {
        inDegree[neighbor]++;
      }
    }
    const queue = [];
    for (let i = 0; i < k; i++) {
      if (inDegree[i] === 0) {
        queue.push(i);
      }
    }
    const order = [];
    while (queue.length > 0) {
      let value = queue.shift();
      order.push(String.fromCharCode(value + 97));
      for (let neighbor of adjList[value]) {
        inDegree[neighbor]--;
        if (inDegree[neighbor] === 0) {
          queue.push(neighbor);
        }
      }
    }
    return order;
  }
}

// const solution = new SolutionAlienFindOrder();
// console.log(solution.findOrder(["baa", "abcd", "abca", "cab", "cad"], 5, 4));

class SolutionShortestPathDFS {
  topoSort(node, adjList, visited, stack) {
    visited[node] = true;
    for (let [neighbor, weight] of adjList[node]) {
      if (!visited[neighbor]) {
        this.topoSort(neighbor, adjList, visited, stack);
      }
    }
    stack.push(node);
  }
  shortestPath(n, m, edges) {
    const adjList = Array.from({ length: n }, () => []);
    for (let i = 0; i < m; i++) {
      const u = edges[i][0];
      const v = edges[i][1];
      const weight = edges[i][2];
      adjList[u].push([v, weight]);
    }
    const visited = Array(n).fill(false);
    const stack = [];
    for (let i = 0; i < n; i++) {
      if (!visited[i]) {
        topoSort(i, adjList, visited, stack);
      }
    }
    const distance = Array(n).fill(Infinity);
    distance[0] = 0;
    while (stack.length > 0) {
      const node = stack.pop();
      for (let [neighbor, weight] of adjList[node]) {
        if (distance[node] + weight < distance[neighbor]) {
          distance[neighbor] = distance[node] + weight;
        }
      }
    }
    return distance;
  }
}

class SolutionShortestPathBFS {
  shortestPath(edges, n, src) {
    const adjList = Array.from({ length: n }, () => []);
    for (let [node, neighbor] of edges) {
      adjList[node].push(neighbor);
    }
    const distance = Array(n).fill(Infinity);
    distance[src] = 0;
    const queue = [src];

    while (queue.length > 0) {
      let node = queue.shift();
      for (let neighbor of adjList[node]) {
        if (distance[node] + 1 < distance[neighbor]) {
          distance[neighbor] = distance[node] + 1;
          queue.push(neighbor);
        }
      }
    }
    return distance.map((item) => (item === Infinity ? -1 : item));
  }
}

// const solution = new SolutionShortestPathBFS();
// console.log(
//   solution.shortestPath(
//     [
//       [0, 1],
//       [0, 3],
//       [3, 4],
//       [4, 5],
//       [5, 6],
//       [1, 2],
//       [2, 6],
//       [6, 7],
//       [7, 8],
//       [6, 8],
//     ],
//     9,
//     0
//   )
// );

class SolutionWordLadderLength {
  wordLadderLength(startWord, targetWord, wordList) {
    const queue = [];
    queue.push([startWord, 1]);
    const wordSet = new Set(wordList);
    wordSet.delete(startWord);

    while (queue.length > 0) {
      const [word, steps] = queue.shift();
      if (word === targetWord) {
        return steps;
      }
      for (let i = 0; i < word.length; i++) {
        for (const char of "abcdefghijklmnopqrstuvwxyz") {
          const new_word = word.slice(0, i) + char + word.slice(i + 1);
          // console.log(new_word);
          // return;
          if (wordSet.has(new_word)) {
            wordSet.delete(new_word);
            queue.push([new_word, steps + 1]);
          }
        }
      }
    }
    return 0;
  }
}

// const solution = new SolutionWordLadderLength();
// console.log(solution.wordLadderLength("der", "dfs", ["des", "der", "dfr", "dgt", "dfs"]));
// this raises a time limit execeeded error on leetcode, for another solution see the class after this one below
class SolutionWordLadderSequence {
  findSequences(beginWord, endWord, wordList) {
    const wordSet = new Set(wordList);
    const queue = [];
    queue.push([beginWord]);
    let usedOnLevel = [];
    usedOnLevel.push(beginWord);
    let level = 0;
    const result = [];
    while (queue.length > 0) {
      const vec = queue.shift();
      if (vec?.length > level) {
        level++;
        for (const word of usedOnLevel) {
          wordSet.delete(word);
        }
        usedOnLevel = [];
      }
      const word = vec[vec.length - 1];
      if (word === endWord) {
        if (result.length === 0) {
          result.push(vec.slice());
        } else if (result[0].length === vec.length) {
          result.push(vec.slice());
        }
      }
      for (let i = 0; i < word.length; i++) {
        for (const char of "abcdefghijklmnopqrstuvwxyz") {
          const newWord = word.slice(0, i) + char + word.slice(i + 1);
          if (wordSet.has(newWord)) {
            vec.push(newWord);
            queue.push(vec.slice());
            usedOnLevel.push(newWord);
            vec.pop();
          }
        }
      }
    }
    return result;
  }
}

// const solution = new SolutionWordLadderSequence();
// console.log(solution.findSequences("der", "dfs", ["des", "der", "dfr", "dgt", "dfs"]));

class SolutionWordLadderSequenceWithoutTLE {
  constructor() {
    this.map = new Map();
    this.result = [];
    this.b = "";
  }

  dfs(word, sequence) {
    if (word === this.b) {
      const copySequence = [...sequence].reverse();
      this.result.push(copySequence);
      return;
    }
    const steps = this.map.get(word);
    const sz = word.length;

    for (let i = 0; i < sz; i++) {
      const originalChar = word[i];
      for (let char of "abcdefghijklmnopqrstuvwxyz") {
        if (char === originalChar) {
          continue;
        }
        const newWord = word.slice(0, i) + char + word.slice(i + 1);
        if (this.map.has(newWord) && this.map.get(newWord) + 1 === steps) {
          sequence.push(newWord);
          this.dfs(newWord, sequence);
          sequence.pop();
        }
      }
    }
  }

  findLadders(beginWord, endWord, wordList) {
    const wordSet = new Set(wordList);
    const queue = [beginWord];
    this.b = beginWord;
    this.map.set(beginWord, 1);
    const sz = beginWord.length;
    wordSet.delete(beginWord);
    while (queue.length > 0) {
      const word = queue.shift();
      const steps = this.map.get(word);

      if (word === endWord) {
        break;
      }
      for (let i = 0; i < sz; i++) {
        const originalChar = word[i];
        for (let char of "abcdefghijklmnopqrstuvwxyz") {
          if (char === originalChar) {
            continue;
          }
          const newWord = word.slice(0, i) + char + word.slice(i + 1);
          if (wordSet.has(newWord)) {
            queue.push(newWord);
            wordSet.delete(newWord);
            this.map.set(newWord, steps + 1);
          }
        }
      }
    }
    if (this.map.has(endWord)) {
      const sequence = [endWord];
      this.dfs(endWord, sequence);
    }
    return this.result;
  }
}

// const solution = new SolutionWordLadderSequenceWithoutTLE();
// console.log(solution.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]));
