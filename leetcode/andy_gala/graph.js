// leetcode 323
class Solution323 {
  buildAdjacencyList(n, edges) {
    const adjList = Array.from({ length: n }, () => []);
    for (let edge of edges) {
      let [src, dest] = edge;
      adjList[src].push(dest);
      adjList[dest].push(src);
    }
    return adjList;
  }

  bsf(node, adjList, visited) {
    const queue = [node];
    visited[node] = true;
    while (queue.length) {
      let currentNode = queue.shift();
      for (let neighbor of adjList[currentNode]) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          queue.push(neighbor);
        }
      }
    }
  }

  countComponents = function (n, edges) {
    const adjList = this.buildAdjacencyList(n, edges);
    const visited = {};
    let numComponents = 0;
    for (let vertex = 0; vertex < adjList.length; vertex++) {
      if (!visited[vertex]) {
        numComponents++;
        this.bfs(vertex, adjList, visited);
      }
    }
    return numComponents;
  };
}

// leetcode 200
class Solution200 {
  getAdjNeighbors(i, j, grid, visited) {
    const adjNeighbors = [];

    if (i > 0 && !visited[i - 1][j]) adjNeighbors.push([i - 1, j]);
    if (i < grid.length - 1 && !visited[i + 1][j]) adjNeighbors.push([i + 1, j]);

    if (j > 0 && !visited[i][j - 1]) adjNeighbors.push([i, j - 1]);
    if (j < grid[0].length - 1 && !visited[i][j + 1]) adjNeighbors.push([i, j + 1]);

    return adjNeighbors;
  }

  dfs(row, col, grid, visited) {
    const stack = [[row, col]];
    let islandSize = 0;
    while (stack.length) {
      let currentNode = stack.pop();
      let [i, j] = currentNode;
      // check if visited at i, j
      if (visited[i][j]) {
        continue;
      }
      visited[i][j] = true;
      // check if cell is part of an island
      if (grid[i][j] === "0") {
        continue;
      }
      islandSize++;
      let adjNeighbors = this.getAdjNeighbors(i, j, grid, visited);
      stack.push(...adjNeighbors);
    }

    return islandSize > 0 ? true : false;
  }

  numIslands(grid) {
    const visited = grid.map((row) => row.map((cell) => false));
    let islandCount = 0;

    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i].length; j++) {
        if (dfs(i, j, grid, visited)) {
          islandCount++;
        }
      }
    }

    return islandCount;
  }
}

// leetcode 547
class SolutionProvinces {
  getEdges(index, edge) {
    const edges = [];
    for (let i = 0; i < edge.length; i++) {
      if (index === i) continue;
      if (edge[i] === 0) continue;
      edges.push(i);
    }
    return edges;
  }

  buildAdjacencyList(edges, n = edges.length) {
    const adjList = Array.from({ length: n }, () => []);
    for (let i = 0; i < edges.length; i++) {
      adjList[i].push(...getEdges(i, edges[i]));
    }
    return adjList;
  }

  dfs(node, adjList, visited) {
    visited[node] = true;
    for (let neighbor of adjList[node]) {
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        this.dfs(neighbor, adjList, visited);
      }
    }
  }

  findCircleNum(isConnected) {
    const adjList = this.buildAdjacencyList(isConnected);
    const visited = {};
    let provinces = 0;
    for (let vertex = 0; vertex < adjList.length; vertex++) {
      if (!visited[vertex]) {
        provinces++;
        this.dfs(vertex, adjList, visited);
      }
    }
    return provinces;
  }
}

// leetcode 207

class SolutionCourseSchedule {
  buildAdjacencyList(n, edges) {
    const adjList = Array.from({ length: n }, () => []);
    for (let edge of edges) {
      let [src, dest] = edge;
      adjList[src].push(dest);
    }

    return adjList;
  }

  hasCycleDFS(node, adjList, visited, arrive, depart) {
    arrive[node]++;
    visited[node] = true;
    for (let neighbor of adjList[node]) {
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        if (this.hasCycleDFS(neighbor, adjList, visited, arrive, depart)) {
          return true;
        }
      } else {
        if (depart[neighbor] === 0) {
          return true;
        }
      }
    }
    depart[node]++;

    return false;
  }

  canFinish(numCourses, prerequisites) {
    const adjList = this.buildAdjacencyList(numCourses, prerequisites);
    const visited = {};
    const arrive = Array.from({ length: numCourses }, () => 0);
    const depart = Array.from({ length: numCourses }, () => 0);

    for (let vertex = 0; vertex < adjList.length; vertex++) {
      if (!visited[vertex]) {
        if (this.hasCycleDFS(vertex, adjList, visited, arrive, depart)) {
          return false;
        }
      }
    }
    return true;
  }
}
