from copy import deepcopy


class SolutionSolveNQueens:
    def solveNQueens(self, n):
        board = [["."] * n for _ in range(n)]
        results = []
        leftrow = [0] * n
        upperDiagonal = [0] * (2 * n - 1)
        lowerDiagonal = [0] * (2 * n - 1)

        def solve(col, leftrow, upperDiagonal, lowerDiagonal):
            if col == n:
                results.append(["".join(row) for row in board])  # append a copy of the current board version to results
                return
            for row in range(n):
                if leftrow[row] == 0 and lowerDiagonal[row + col] == 0 and upperDiagonal[n - 1 + col - row] == 0:
                    # place the queen
                    board[row][col] = "Q"
                    leftrow[row] = 1
                    lowerDiagonal[row + col] = 1
                    upperDiagonal[n - 1 + col - row] = 1

                    # recurse to place the next queen
                    solve(col + 1, leftrow, upperDiagonal, lowerDiagonal)

                    # backtrack by removing the queen
                    board[row][col] = "."
                    leftrow[row] = 0
                    lowerDiagonal[row + col] = 0
                    upperDiagonal[n - 1 + col - row] = 0

        solve(0, leftrow, upperDiagonal, lowerDiagonal)
        return results


# solution = SolutionSolveNQueens()
# print(solution.solveNQueens(4))


class SolutionSolveSudoku:
    def solveSudoku(self, board):

        def isValid(board, row, col, c):
            for i in range(9):
                if board[i][col] == c or board[row][i] == c:
                    return False
                if board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == c:
                    return False
            return True

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == ".":
                    for c in "123456789":
                        if isValid(board, i, j, c):
                            board[i][j] = c
                            if self.solveSudoku(board):
                                return True
                            else:
                                board[i][j] = "."
                    return False
        return True


def searchMaze(arr, n):
    # Write your code here.
    if arr[0][0] == 0:
        return []
    result = []
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # [up, down, left, right]
    visited = [[0] * n for _ in range(n)]

    def recursiveCall(row, col, path):
        visited[row][col] = 1
        if row == n - 1 and col == n - 1:
            result.append("".join(path[::]))
            return

        for index, [deltaRow, deltaCol] in enumerate(directions):
            nrow = row + deltaRow
            ncol = col + deltaCol
            currentDirection = None
            if index == 0:
                currentDirection = "U"
            elif index == 1:
                currentDirection = "D"
            elif index == 2:
                currentDirection = "L"
            elif index == 3:
                currentDirection = "R"

            if 0 <= nrow < n and 0 <= ncol < n and visited[nrow][ncol] == 0 and arr[nrow][ncol] == 1:
                visited[nrow][ncol] = 1
                path.append(currentDirection)
                recursiveCall(nrow, ncol, path)
                visited[nrow][ncol] = 0
                path.pop()

    recursiveCall(0, 0, [])
    result.sort()
    return result


# searchMaze([[1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1], [0, 0, 0, 1]], 4)


def kthPermutation(arr):
    n = len(arr)
    count = 0
    result = []

    def recursiveCall(startIndex, arr):
        if startIndex == n:
            nonlocal count, result
            count += 1
            if count == 3:
                result = arr[::]
            return
        for i in range(startIndex, n):
            arr[i], arr[startIndex] = arr[startIndex], arr[i]
            recursiveCall(startIndex + 1, arr)
            arr[i], arr[startIndex] = arr[startIndex], arr[i]

    recursiveCall(0, arr)
    return result


# print(kthPermutation([1, 2, 3]))
