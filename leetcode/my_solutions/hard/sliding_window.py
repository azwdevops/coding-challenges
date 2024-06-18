
# solution not working yet
class LongestValidParentheses:
  def longestValidParentheses(self, s: str) -> int:
    opening = []
    valid = []

    for item in s:
      if item == '(':
        opening.append('(')
      elif item == ')' and len(opening) > 0:
        valid.append(')')
        valid.insert(0, '(')
        opening.pop()
    return len(valid)

new_solution = LongestValidParentheses()
# print(new_solution.longestValidParentheses("(()"))
print(new_solution.longestValidParentheses(")()())"))