from typing import List, Optional
from collections import defaultdict


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


# leetcode 417
class Solution417:
    def check(self, row, col, heights, ocean):
        ocean[row][col] = True
        lst = [[row + 1, col], [row - 1, col], [row, col - 1], [row, col + 1]]
        for r, c in lst:
            if 0 <= r < len(heights) and 0 <= c < len(heights[0]) and not ocean[r][c] and heights[r][c] >= heights[row][col]:
                self.check(r, c, heights, ocean)
        return

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        result = []
        rows, cols = len(heights), len(heights[0])
        pacific = [[False for i in range(cols)] for j in range(rows)]
        atlantic = [[False for i in range(cols)] for j in range(rows)]
        for i in range(rows):
            self.check(i, 0, heights, pacific)
            self.check(i, cols - 1, heights, atlantic)

        for i in range(cols):
            self.check(0, i, heights, pacific)
            self.check(rows - 1, i, heights, atlantic)
        for r in range(rows):
            for c in range(cols):
                if atlantic[r][c] and pacific[r][c]:
                    result.append([r, c])
        return result


# leetcode 207
class Solution207:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        prereqs = defaultdict(list)
        for c, p in prerequisites:
            prereqs[c].append(p)

        def cycle(course, seen):
            if course in seen:
                return True
            seen.add(course)
            for p in prereqs[course]:
                if cycle(p, seen):
                    return True
            prereqs[course] = []
            seen.remove(course)
            return False

        seen = set()

        for course in range(numCourses):
            if cycle(course):
                return False

        return True


# leetcode 133
class Solution133:
    def cloneGraph(self, node: Optional[Node]) -> Optional[Node]:
        if not node:
            return
        visited = {}
        return self.clone(node, visited)

    def clone(self, node, visited):
        if node in visited:
            return visited[node]
        cloned_node = Node(node.val)
        visited[node] = cloned_node
        for neighbor in node.neighbors:
            cloned_node.neighbors.append(self.clone(neighbor, visited))
        return cloned_node


# leetcode 200
class Solution200:
    def numIslands(self, grid: List[List[str]]) -> int:
        islands = 0
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                if grid[r][c] == "1":
                    self.dfs(grid, r, c)
                    islands += 1
        return islands

    def dfs(self, grid, r, c):
        grid[r][c] = "0"
        lst = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for row, col in lst:
            if row >= 0 and col >= 0 and row < len(grid) and col < len(grid[row]) and grid[r][c] == "1":
                self.dfs(grid, row, col)
