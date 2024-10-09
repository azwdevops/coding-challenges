import heapq
from typing import List
from collections import deque, defaultdict
from copy import deepcopy
from string import ascii_lowercase as lowercase


class SolutionBFS:
    # Function to return Breadth First Traversal of given graph.
    def bfsOfGraph(self, V: int, adj: List[List[int]]) -> List[int]:
        visited = [0] * V
        visited[0] = 1
        queue = deque()
        queue.append(0)
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append(neighbor)

        return result


class SolutionDFS:
    # Function to return a list containing the DFS traversal of the graph.
    def dfsOfGraph(self, V, adj):
        visited = [0] * V
        start = 0
        result = []
        self.dfs(start, adj, visited, result)

        return result

    def dfs(self, node, adjList, visited, result):
        visited[node] = 1
        result.append(node)
        for neighbor in adjList[node]:
            if not visited[neighbor]:
                self.dfs(neighbor, adjList, visited, result)


class SolutionProvinces:
    def numProvinces(self, adj, V):
        adjList = [[] for i in range(V)]
        for i in range(V):
            for j in range(V):
                if adj[i][j] == 1 and i != j:
                    adjList[i].append(j)
                    adjList[j].append(i)

        visited = [0] * V
        count = 0
        for i in range(V):
            if not visited[i]:
                count += 1
                self.dfs(i, adjList, visited)
        return count

    def dfs(self, node, adjList, visited):
        visited[node] = 1
        for neighbor in adjList[node]:
            if not visited[neighbor]:
                self.dfs(neighbor, adjList, visited)


class SolutionIslands:
    def numIslands(self, grid):
        n = len(grid)
        m = len(grid[0])
        visited = [[0 for _ in range(m)] for n in range(n)]
        count = 0
        for row in range(n):
            for col in range(m):
                if not visited[row][col] and grid[row][col] == 1:
                    count += 1
                    self.bfs(row, col, visited, grid, n, m)

        return count

    def bfs(self, row, col, visited, grid, n, m):

        visited[row][col] = 1
        queue = deque()
        queue.append([row, col])
        while queue:
            row, col = queue.popleft()

            # traverse the neighbors and mark them
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbor_row = row + i
                    neighbor_col = col + j
                    if (
                        0 <= neighbor_row < n
                        and 0 <= neighbor_col < m
                        and not visited[neighbor_row][neighbor_col]
                        and grid[neighbor_row][neighbor_col] == 1
                    ):
                        visited[neighbor_row][neighbor_col] = 1
                        queue.append([neighbor_row, neighbor_col])


class SolutionFloodFill:
    def floodFill(self, image, sr, sc, newColor):
        initialColor = image[sr][sc]
        result = deepcopy(image)
        n = len(image)
        m = len(image[0])
        self.dfs(sr, sc, result, image, newColor, initialColor, n, m)

        return result

    def dfs(self, row, col, result, image, newColor, initialColor, n, m):
        result[row][col] = newColor
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        for deltaRow, deltaCol in directions:
            nrow = row + deltaRow
            ncol = col + deltaCol
            if 0 <= nrow < n and 0 <= ncol < m and image[nrow][ncol] == initialColor and result[nrow][ncol] != newColor:
                self.dfs(nrow, ncol, result, image, newColor, initialColor, n, m)


class SolutionOrangesRotting:
    # Function to find minimum time required to rot all oranges.
    def orangesRotting(self, grid):
        n = len(grid)
        m = len(grid[0])
        queue = deque()
        visited = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 2:
                    queue.append([[i, j], 0])
                    visited[i][j] = 1

        result = 0
        while queue:
            [row, col], currentTime = queue.popleft()
            result = max(result, currentTime)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for deltaRow, deltaCol in directions:
                nrow = row + deltaRow
                ncol = col + deltaCol
                if 0 <= nrow < n and 0 <= ncol < m and visited[nrow][ncol] == 0 and grid[nrow][ncol] == 1:
                    queue.append([[nrow, ncol], currentTime + 1])
                    visited[nrow][ncol] = 1

        for i in range(n):
            for j in range(m):
                if visited[i][j] == 0 and grid[i][j] == 1:
                    return -1

        return result


class SolutionBFS:
    # Function to detect cycle in an undirected graph.
    def isCycle(self, V: int, adj: List[List[int]]) -> bool:
        visited = [0] * V
        for i in range(V):
            if not visited[i]:
                if self.detect(i, adj, visited):
                    return True

        return False

    def detect(self, src, adj, visited):
        visited[src] = 1
        queue = deque()
        queue.append((src, -1))
        while queue:
            node, parent = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = 1
                    queue.append((neighbor, node))
                elif parent != neighbor:
                    return True

        return False


class SolutionDFS:
    # Function to detect cycle in an undirected graph.
    def isCycle(self, V: int, adj: List[List[int]]) -> bool:
        visited = [0] * V
        for i in range(V):
            if not visited[i]:
                if self.dfs(i, -1, visited, adj):
                    return True
        return False

    def dfs(self, node, parent, visited, adjList):
        visited[node] = 1
        for neighbor in adjList[node]:
            if not visited[neighbor]:
                if self.dfs(neighbor, node, visited, adjList):
                    return True
            elif neighbor != parent:
                return True

        return False


class SolutionNearestCell:

    # Function to find distance of nearest 1 in the grid for each cell.
    def nearest(self, grid):
        n = len(grid)
        m = len(grid[0])
        visited = [[0] * m for _ in range(n)]
        distance = [[0] * m for _ in range(n)]
        queue = deque()
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    queue.append([[i, j], 0])
                    visited[i][j] = 1
                else:
                    visited[i][j] = 0

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while queue:
            [row, col], steps = queue.popleft()
            distance[row][col] = steps
            for deltaRow, deltaCol in directions:
                nrow = row + deltaRow
                ncol = col + deltaCol
                if 0 <= nrow < n and 0 <= ncol < m and visited[nrow][ncol] == 0:
                    visited[nrow][ncol] = 1
                    queue.append([[nrow, ncol], steps + 1])
        return distance


class SolutionFill:
    def fill(self, n, m, mat):
        visited = [[0] * m for _ in range(n)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # traverse first row and last row
        for j in range(m):
            # first row
            if not visited[0][j] and mat[0][j] == "O":
                self.dfs(0, j, visited, mat, directions, n, m)
            # last row
            if not visited[n - 1][j] and mat[n - 1][j] == "O":
                self.dfs(n - 1, j, visited, mat, directions, n, m)

        for i in range(n):
            # first column
            if not visited[i][0] and mat[i][0] == "O":
                self.dfs(i, 0, visited, mat, directions, n, m)
            if not visited[i][m - 1] and mat[i][m - 1] == "O":
                self.dfs(i, m - 1, visited, mat, directions, n, m)

        for i in range(n):
            for j in range(m):
                if not visited[i][j] and mat[i][j] == "O":
                    mat[i][j] = "X"
        return mat

    def dfs(self, row, col, visited, mat, directions, n, m):
        visited[row][col] = 1
        # check for right, left, top, bottom
        for deltaRow, deltaCol in directions:
            nrow = row + deltaRow
            ncol = col + deltaCol
            if 0 <= nrow < n and 0 <= ncol < m and not visited[nrow][ncol] and mat[nrow][ncol] == "O":
                self.dfs(nrow, ncol, visited, mat, directions, n, m)


class SolutionDijkstraUsingPriorityQueue:
    def dijkstra(self, V, adj, S):
        priority_queue = []
        distance = [float("inf")] * V
        distance[S] = 0
        heapq.heappush(priority_queue, (0, S))
        while priority_queue:
            currentDistance, node = heapq.heappop(priority_queue)
            for neighbor, weight in adj[node]:
                if currentDistance + weight < distance[neighbor]:
                    distance[neighbor] = currentDistance + weight
                    heapq.heappush(priority_queue, (distance[neighbor], neighbor))

        return distance


class SolutionDijkstraUsingSet:
    def dijkstra(self, V, adj, S):
        nodes_set = set()
        distance = [float("inf")] * V
        nodes_set.add((0, S))
        distance[S] = 0
        while nodes_set:
            currentDistance, node = min(nodes_set)
            nodes_set.remove((currentDistance, node))

            for neighbor, weight in adj[node]:
                if currentDistance + weight < distance[neighbor]:
                    if distance[neighbor] != float("inf"):
                        nodes_set.discard((distance[neighbor], neighbor))

                    distance[neighbor] = currentDistance + weight
                    nodes_set.add((distance[neighbor], neighbor))

        return distance
