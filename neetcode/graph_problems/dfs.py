from typing import List
from collections import defaultdict

# leetcode 207

class Solution207:
  def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    # map each course to prereq list
    preMap = defaultdict(list)
    for course, pre in prerequisites:
      preMap[course].append(pre)

    # visitSet = all courses along the current DFS path
    visitSet = set()

    def dfs(course):
      if course in visitSet:
        return False
      if preMap[course] == []:
        return True
      visitSet.add(course)

      for pre in preMap[course]:
        if not dfs(pre):
          return False
      visitSet.remove(course)
      preMap[course] = []
      return True 
    
    for course in range(numCourses):
      if not dfs(course):
        return False
      
    return True
  
# solution = Solution207()
# print(solution.canFinish(2, [[1,0]]))
# print(solution.canFinish(2, [[1,0],[0,1]]))


# leetcode 210
class Solution210:
  def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # build adjacency list of preqs
    prereq_adjacency = defaultdict(list)
    for course, pre in prerequisites:
      prereq_adjacency[course].append(pre)
    # a course has three possible states
    # 1 visited - course has been added to output
    # 2 visiting - course not added to output but added to cycle
    # 3 unvisited - course not added to output or cycle
    output = []
    visit, cycle = set(), set()
    def dfs(course):
      if course in cycle:
        return False
      if course in visit:
        return True
      cycle.add(course)
      for pre in prereq_adjacency[course]:
        if not dfs(pre):
          return False
      cycle.remove(course)
      visit.add(course)
      output.append(course)

      return True
    for c in range(numCourses):
      if not dfs(c):
        return []
    return output

# solution = Solution210()
# print(solution.findOrder(2, [[1,0]]))
# print(solution.findOrder(4, [[1,0],[2,0],[3,1],[3,2]]))