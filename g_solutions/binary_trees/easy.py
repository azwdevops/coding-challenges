from typing import Optional

class TreeNode:
  def __init__(self, val = 0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

# leetcode 222
class Solution222:
  def countNodes(self, root: Optional[TreeNode]) -> int:
    if not root:
      return 0

    leftHeight = self.getHeight(root.left)
    rightHeight = self.getHeight(root.right)

    if leftHeight == rightHeight:
      # left subtree is a perfect binary tree
      return (1 << leftHeight) + self.countNodes(root.right)
    else:
      # right subtree is a perfect binary tree minus one level
      return (1 << rightHeight) + self.countNodes(root.left)


  def getHeight(self, node):
    height = 0
    while node:
      height += 1
      node = node.left
    return height
  
# solution = Solution222()
# print(solution.countNodes([1,2,3,4,5,6]))


# leetcode 1379
class Solution1379:
  def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
    if original is None:
      return None
    if original == target:
      return cloned
    # traverse the left subtree
    leftResult = self.getTargetCopy(original.left, cloned.left, target)
    if leftResult:
      return leftResult
    return self.getTargetCopy(original.right, cloned.right, target)
  
# leetcode 2331
class Solution2331:
  def evaluateTree(self, root: Optional[TreeNode]) -> bool:
    if root is None:
      return False
    if root.left is None:
      return bool(root.val)
    left_result = self.evaluateTree(root.left)
    right_result = self.evaluateTree(root.right)

    if root.val == 2:
      return left_result or right_result
    elif root.val == 3:
      return left_result and right_result
    
# leetcode 206
class ListNode:
  def __init__(self, val=0, next=None):
    self.val = val
    self.next = next

  def __repr__(self) -> str:
    return str(self.val)
  
class Solution206:
  def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if head is None or head.next is None:
      return head
    new_head = self.reverseList(head.next)
    head.next.next = head
    head.next = None

    return new_head
  
# node5 = ListNode(5)
# node4 = ListNode(4, node5)
# node3 = ListNode(3, node4)
# node2 = ListNode(2, node3)
# head = ListNode(1, node2)

# solution = Solution206()
# print(solution.reverseList(head))

# leetcode 21
class Solution21:
  def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    if list1 is None:
      return list2
    if list2 is None:
      return list1
    if list1.val < list2.val:
      list1.next = self.mergeTwoLists(list1.next, list2)
      return list1
    else:
      list2.next = self.mergeTwoLists(list1, list2.next)
      return list2
    

# leetcode 234
class Solution234:
  def isPalindrome(self, head: Optional[ListNode]) -> bool:
    if head is None or head.next is None:
      return True
    slow, fast = head, head
    while fast and fast.next:
      slow = slow.next
      fast = fast.next.next

    second_half = self.reverseList(slow)

    first_half = head
    while second_half:
      if first_half.val != second_half.val:
        return False
      first_half = first_half.next
      second_half = second_half.next
    return True

  def reverseList(self, head):
    prev = None
    current = head
    while current:
      next_node = current.next
      current.next = prev
      prev = current
      current = next_node
    return prev