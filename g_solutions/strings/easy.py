from typing import List
import string


# leetcode 2194
class Solution2194:
    def cellsInRange(self, s: str) -> List[str]:
        def col_to_index(col: str) -> int:
            # convert a column letter to a zero based index
            return ord(col) - ord("A")

        def index_to_col(index: int) -> str:
            return chr(index + ord("A"))

        # parse the input string
        col1, row1, col2, row2 = s[0], int(s[1]), s[3], int(s[4])
        # convert column letters to indices
        col1_index = col_to_index(col1)
        col2_index = col_to_index(col2)

        result = []

        # iterate over the range of columns and rows
        for col_index in range(col1_index, col2_index + 1):
            for row in range(row1, row2 + 1):
                # convert column index back to letter and form the cell string
                cell = index_to_col(col_index) + str(row)
                result.append(cell)

        return result


# leetcode 1974
class Solution1974:
    def minTimeToType(self, word: str) -> int:
        letters = string.ascii_lowercase
        current_index = 0
        seconds = 0

        for char in word:
            char_index = letters.index(char)
            clockwise_distance = (char_index - current_index) % 26
            counterclockwise_distance = (current_index - char_index) % 26

            seconds += min(clockwise_distance, counterclockwise_distance) + 1

            current_index = char_index

        return seconds


# leetcode 1047
class Solution1047:
    def removeDuplicates(self, s: str) -> str:
        stack = []
        for char in s:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        return "".join(stack)


sum(["1", "2"])
