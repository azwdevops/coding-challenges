const depthFirstPrint = (graph, source) => {
  const stack = [source];

  while (stack.length > 0) {
    const current = stack.pop();
    console.log(current);
    for (let neighbor of graph[current]) {
      stack.push(neighbor);
    }
  }
};

const depthFirstPrint2 = (graph, source) => {
  console.log(source);
  for (let neighbor of graph[source]) {
    depthFirstPrint2(graph, neighbor);
  }
};

const graph = {
  a: ["c", "b"],
  b: ["d"],
  c: ["e"],
  d: ["f"],
  e: [],
  f: [],
};

// depthFirstPrint(graph, "a");
// depthFirstPrint2(graph, "a");

const breadthFirstPrint = (graph, source) => {
  const queue = [source];
  while (queue.length > 0) {
    const current = queue.shift();
    console.log(current);
    for (let neighbor of graph[current]) {
      queue.push(neighbor);
    }
  }
};

// breadthFirstPrint(graph, "a");

// using depth first traversal
const hasPath = (graph, src, dest) => {
  if ((src = dest)) return true;
  for (let neighbor of graph[src]) {
    if (hasPath(graph, neighbor, dest) === true) {
      return true;
    }
  }
  return false;
};

// using breadth first traversal
const hasPath2 = (graph, src, dest) => {
  const queue = [src];
  while (queue.length > 0) {
    const current = queue.shift();
    if (current === dest) return true;
    for (let neighbor of graph[current]) {
      queue.push(neighbor);
    }
  }
  return false;
};

const undirectedPath = (edges, nodeA, nodeB) => {
  const graph = buildGraph(edges);
  return hasPath3(graph, nodeA, nodeB, new Set());
};

const hasPath3 = (graph, src, dest, visited) => {
  if (src === dest) return true;
  if (visited.has(src)) return false;
  visited.add(src);
  for (let neighbor of graph[src]) {
    if (hasPath3(graph, neighbor, dest, visited) === true) {
      return true;
    }
  }
  return false;
};

const buildGraph = (edges) => {
  const graph = {};
  for (let edge of edges) {
    const [a, b] = edge;
    if (!(a in graph)) graph[a] = [];
    if (!(b in graph)) graph[b] = [];
    graph[a].push(b);
    graph[b].push(a);
  }

  return graph;
};

const edges = [
  ["i", "j"],
  ["k", "i"],
  ["m", "k"],
  ["k", "l"],
  ["o", "n"],
];

// console.log(undirectedPath(edges, "j", "m"));

const connectedComponentsCount = (graph) => {
  const visited = new Set();
  let count = 0;
  for (let node in graph) {
    if (explore(graph, node, visited) === true) {
      count += 1;
    }
  }
  return count;
};

const explore = (graph, current, visited) => {
  if (visited.has(String(current))) return false;

  visited.add(String(current));

  for (let neighbor of graph[current]) {
    explore(graph, neighbor, visited);
  }

  return true;
};

const graph1 = {
  0: [8, 1, 5],
  1: [0],
  5: [0, 8],
  8: [0, 5],
  2: [3, 4],
  3: [2, 4],
  4: [3, 2],
};
// console.log(connectedComponentsCount(graph1));

const largestComponent = (graph) => {
  const visited = new Set();
  let longest = 0;
  for (let node in graph) {
    const size = exploreSize(graph, node, visited);
    if (size > longest) {
      longest = size;
    }
  }
  return longest;
};

const exploreSize = (graph, current, visited) => {
  if (visited.has(current)) return 0;

  visited.add(current);
  let size = 1;
  for (let neighbor of graph[current]) {
    size += exploreSize(graph, neighbor, visited);
  }
  return size;
};

const shortestPath = (edges, nodeA, nodeB) => {
  const graph = buildGraph2(edges);
  const visited = new Set([nodeA]);
  const queue = [[nodeA, 0]];
  while (queue.length > 0) {
    const [node, distance] = queue.shift();
    if (node === nodeB) return distance;

    for (let neighbor of graph[node]) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push([neighbor, distance + 1]);
      }
    }
  }
  return -1;
};

const buildGraph2 = (edges) => {
  const graph = {};
  for (let edge of edges) {
    const [a, b] = edge;
    if (!(a in graph)) graph[a] = [];
    if (!(b in graph)) graph[b] = [];

    graph[a].push(b);
    graph[b].push(a);
  }
};

const islandCount = (grid) => {
  const visited = new Set();
  let count = 0;
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; (c = grid[0].length); c++) {
      if (exploreGrid(grid, r, c, visited) === true) {
        count += 1;
      }
    }
  }
  return count;
};

const exploreGrid = (grid, r, c, visited) => {
  const rowInbounds = 0 <= r && r < grid.length;
  const columnInbounds = 0 <= r && r < grid[0].length;

  if (!rowInbounds || !columnInbounds) return false;

  if (grid[r][c] === "W") return false;

  const pos = r + "," + c;
  if (visited.has(pos)) return false;
  visited.add(pos);

  exploreGrid(grid, r - 1, c, visited);
  exploreGrid(grid, r + 1, c, visited);
  exploreGrid(grid, r, c - 1, visited);
  exploreGrid(grid, r, c + 1, visited);

  return true;
};

const minimumIsland = (grid) => {
  const visited = new Set();
  let minSize = Infinity;
  for (let r = 0; r < grid.length; r++) {
    for (let r = 0; r < grid[0].length; r++) {
      const size = exploreIslandSize(grid, r, c, visited);
      if (size > 0 && size < minSize) {
        minSize = size;
      }
    }
  }
  return minSize;
};

const exploreIslandSize = (grid, r, c, visited) => {
  const rowInbounds = 0 <= r && r < grid.length;
  const columnInbounds = 0 <= c && c < grid[0].length;
  if (!rowInbounds || !columnInbounds) return 0;

  if (grid[r][c] === "W") return 0;

  const pos = r + "," + c;
  if (visited.has(pos)) return 0;

  visited.add(pos);

  let size = 1;

  size += exploreIslandSize(grid, r - 1, c, visited);
  size += exploreIslandSize(grid, r + 1, c, visited);
  size += exploreIslandSize(grid, r, c - 1, visited);
  size += exploreIslandSize(grid, r, c + 1, visited);
  return size;
};
