from collections import deque, defaultdict
import heapq


class TreeNode:
    def __init__(self, val=None, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def sameTree(root1: TreeNode, root2: TreeNode):
    if root1 is None and root is None:
        return True
    if root1 is None or root2 is None:
        return False
    if root1.val != root2.val:
        return False

    return sameTree(root1.left, root2.left) and sameTree(root1.right, root2.right)


class SolutionZigZagLevelOrder:
    def zigZagLevelOrder(self, root):
        result = []
        if not root:
            return result
        nodesQueue = deque([root])
        leftToRight = True

        while nodesQueue:
            n = len(nodesQueue)
            row = [0] * n
            for i in range(n):
                node = nodesQueue.popleft()
                index = i if leftToRight else (n - 1 - i)
                row[index] = node.data

                if node.left:
                    nodesQueue.append(node.left)
                if node.right:
                    nodesQueue.append(node.right)
            leftToRight = not leftToRight

            result.extend(row)

        return result


class SolutionTraverseBoundary:
    def traverseBoundary(root: TreeNode):
        if not root:
            return []
        result = [root.val]
        if not root.right and not root.left:
            return result

        def isLeaf(node):
            return not node.right and not node.left

        def addLeftBoundary(node):
            current = node.left
            while current:
                if not isLeaf(current):
                    result.append(current.val)
                if current.left:
                    current = current.left
                else:
                    current = current.right

        def addLeaves(node):
            if not node:
                return
            if isLeaf(node):
                result.append(node.val)
            addLeaves(node.left)
            addLeaves(node.right)

        def addRightBoundary(node):
            temp = deque()
            current = current.right
            while current:
                if not isLeaf(current):
                    temp.appendleft(current.val)
                if current.right:
                    current = current.right
                else:
                    current = current.left
            result.extend(temp)

        addLeftBoundary(root)
        addLeaves(root)
        addRightBoundary(root)

        return result


def solutionRightView(root: TreeNode):
    result = []

    def recursiveCall(node, level):
        if node is None:
            return
        if level == len(result):
            result.append(node.val)
        recursiveCall(node.right, level + 1)
        recursiveCall(node.left, level + 1)

    recursiveCall(root, 0)

    return result


def printNodesAtDistance(root, target, k):
    def bfs_to_find_parents(node):
        parent_map = {}
        if not node:
            return parent_map
        queue = deque()
        queue.append(root)

        while queue:
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
                parent_map[node.left] = node
            if node.right:
                queue.append(node.right)
                parent_map[node.right] = node
        return parent_map

    def find_nodes_with_distance_k(target, parent_map, k):
        queue = deque()
        queue.append(target)
        visited = set()
        visited.add(target)
        distance = 0

        while distance != k:
            n = len(queue)
            for _ in range(n):
                node = queue.popleft()
                # check if parent exists and it has not been visited before
                parent_node = parent_map[node]
                if parent_node and parent_node not in visited:
                    queue.append(parent_node)
                if node.left and node not in visited:
                    queue.append(node.left)
                    visited.add(node.left)
                if node.right and node not in visited:
                    queue.append(node.right)
                    visited.add(node.right)
            distance += 1

        return queue

    parent_map = bfs_to_find_parents(root)

    return find_nodes_with_distance_k(target, parent_map, k)


s = "N,parent,1-2-3;N1,parent1,4-5;N2,parent2,6-7-8-9;"

# for item in s.split(";"):
#     if item != "":
#         nodeValue, parent, children = item.split(",")
#         for child in children.split("-"):
#             print(child)


def kthSmallest(root, k):
    stack = []
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if k == 0:
            return root
        root = root.right
