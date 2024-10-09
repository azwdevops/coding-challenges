from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# leetcode 2326
class Solution2326:
    def spiralMatrix(self, m: int, n: int, head: Optional[ListNode]) -> List[List[int]]:
        # step 1 initialize matrix with -1
        matrix = [[-1 for _ in range(n)] for _ in range(m)]

        # step 2 define directions: right (0, 1), down (1,0), left (0,-1), up (-1, 0)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        current_direction = 0
        row, col = 0, 0

        # step 3: traverse the matrix and linked list
        current_node = head
        while current_node:
            # place current node value in the matrix
            matrix[row][col] = current_node.val
            current_node = current_node.next
            # calculate the next position
            next_row = row + directions[current_direction][0]
            next_col = col + directions[current_direction][1]

            # step 4 check if the next position is valid
            if 0 <= next_row < m and 0 <= next_col < n and matrix[next_row][next_col] == -1:
                row, col = next_row, next_col
            else:
                # change direction
                current_direction = (current_direction + 1) % 4
                row += directions[current_direction][0]
                col += directions[current_direction][1]

        # step 5 return the matrix
        return matrix


# leetcode 2058
class Solution2058:
    def nodesBetweenCriticalPoints(self, head: Optional[ListNode]) -> List[int]:
        # step 1 initialize variables
        critical_points = []
        prev_node = None
        current_node = head
        index = 0
        # step 2 traverse the linked list
        while current_node and current_node.next:
            next_node = current_node.next
            if prev_node:
                if (current_node.val > prev_node.val and current_node.val > next_node.val) or (
                    current_node.val < prev_node.val and current_node.val < next_node.val
                ):
                    critical_points.append(index)
            prev_node = current_node
            current_node = next_node
            index += 1
        # step 3 check if there are fewer than two critical points
        if len(critical_points) < 2:
            return [-1, -1]
        # step 4 claculate minDistance and maxDistance
        min_distance = float("inf")
        max_distance = critical_points[-1] - critical_points[0]
        for i in range(1, len(critical_points)):
            min_distance = min(min_distance, critical_points[i] - critical_points[i - 1])
        return [min_distance, max_distance]
