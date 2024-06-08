# from lessons by teamtreehouse algorithms course on freecodecamp youtube channel

# sample binary search, to determine how many steps it takes to find a number
def sample_binary_search(my_list, target):
  my_list.sort()
  start = 0
  end = len(my_list) - 1
  if target > my_list[end] or target < my_list[start]:
    return False
  steps = 0
  while start <= end:
    midpoint = (end + start) // 2
    if my_list[midpoint] == target:
      return steps, midpoint
    elif my_list[midpoint] > target:
      end = midpoint - 1
    elif my_list[midpoint] < target:
      start = midpoint + 1
    steps += 1
  return False

# print(sample_binary_search([2,6,5,1,3,7,9,10], 1))

def sample_recursive_binary_search(my_list, target):
  my_list.sort()
  if len(my_list) == 0:
    return False
  else:
    midpoint = (len(my_list))//2
    if my_list[midpoint] == target:
      return True
    if my_list[midpoint] < target:
      return sample_recursive_binary_search(my_list[midpoint + 1:], target)
    else:
      return sample_recursive_binary_search(my_list[:midpoint], target)
    

# print(sample_recursive_binary_search([2,6,5,1,3,7,9,10], 1))

# creating a singly linked list
class Node:
  """
    an object for storing a single node of a linked list
    models two attributes data and the link to the next node in the list
  """
  data = None
  next_node = None

  def __init__(self, data):
    self.data = data

  def __repr__(self):
    return f"<Node data: {self.data}>"
  
class LinkedList:
  """
    singly linked list
  """
  def __init__(self):
    self.head = None

  def is_empty(self):
    return self.head == None
  
  def size(self):
    """
      returns the number of nodes in the list takes O(n) time
    """
    current = self.head
    count = 0
    while current:
      count += 1
      current = current.next_node
    return count
  
  def add(self, data) -> None:
    """
      adds a new node containing data at the head of the list, it takes O(1) time
    """
    new_node = Node(data)
    new_node.next_node = self.head
    self.head = new_node

  
  def search(self, key):
    """
      search for the first node containing data that matches the given key, returns node or None if not found, takes O(n) time
    """
    current = self.head
    while current:
      if current.data == key:
        return current
      current = current.next_node
    return None

  def insert(self, data, index):
    """
      inserts a new node containing data at index position, insertion takes O(1) time but finding the node at the insertion point takes O(n) time
      takes overall O(n) time
    """
    if index == 0:
      self.add(data)
    if index > 0:
      new = Node(data)
      position = index
      current = self.head

      while position > 1:
        current = current.next_node
        position -= 1
      previous = current
      next = current.next_node
      previous.next_node = new
      new.next_node = next

  def remove(self, key):
    """
      removes node containing data that matches the key, returns the node or None if key does not exist. It takes O(n) time
    """
    current = self.head
    previous = None
    found = False

    while current and not found:
      if current.data == key and current is self.head:
        found = True
        self.head = current.next_node
      elif current.data == key:
        found = True
        previous.next_node = current.next_node
      else:
        previous = current
        current = current.next_node

    return current
  
  def node_at_index(self, index):
    if index == 0:
      return self.head
    
    current = self.head
    position = 0

    while position < index:
      current = current.next_node
      position += 1

    return current

  def __repr__(self) -> str:
    """
      return a string representation of the list, it takes O(n) time
    """
    nodes = []
    current = self.head
    while current:
      if current is self.head:
        nodes.append(f"[Head: {current.data}]")
      elif current.next_node is None:
        nodes.append(f"[Tail: {current.data}]")
      else:
        nodes.append(f"[{current.data}]")
      current = current.next_node

    return '-> '.join(nodes)


# linked_list_instance = LinkedList()
# linked_list_instance.add(1)
# linked_list_instance.add(2)
# linked_list_instance.add(3)
# print(linked_list_instance)
# linked_list_instance.remove(2)
# print(linked_list_instance)


# =================== START OF MERGE AND SORT FOR NORMAL LIST ========================
# =================== START OF MERGE AND SORT FOR NORMAL LIST ========================
# =================== START OF MERGE AND SORT FOR NORMAL LIST ========================
def merge_sort(my_list):
  """
    sorts a list in ascending order. Returns a new sorted list
    Divide: find the midpoint of the list and divide into sublists
    Conquer: recursively sort the sublists created in previous step
    Combine: merge the sorted sublists into a single sorted list
    overall it takes O(n log n) time
  """
  if len(my_list) <= 1:
    return my_list
  left_half, right_half = split_list(my_list)
  left = merge_sort(left_half)
  right = merge_sort(right_half)

  return merge(left, right) 

def split_list(list_to_split):
  """
    divide the unsorted list at midpoint into sublists. Return two sublists - left and right
    takes overall O(log n) time
  """
  midpoint = len(list_to_split) // 2
  left_half = list_to_split[:midpoint]
  right_half = list_to_split[midpoint:]

  return left_half, right_half

def merge(left, right):
  """
    merges two lists (arrays) sorting them in the process. returns a new merged list
    takes O(n) time
  """
  new_list = []
  i = 0
  j = 0

  while i < len(left) and j < len(right):
    if left[i] < right[j]:
      new_list.append(left[i])
      i += 1
    else:
      new_list.append(right[j])
      j += 1

  while i < len(left):
    new_list.append(left[i])
    i += 1

  while j < len(right):
    new_list.append(right[j])
    j += 1

  return new_list


def verify_sorted(my_list):
  n = len(my_list)
  if n == 0 or n == 1:
    return True
  return my_list[0] <= my_list[1] and verify_sorted(my_list[1:])


# sorted_list = merge_sort([10,2,3,5,1,8,2,1,0,6,9,8])

# print(verify_sorted(sorted_list))


# =================== END OF MERGE AND SORT FOR NORMAL LIST ========================


# =================== START OF MERGE AND SORT FOR LINKED LIST ========================
def merge_sort_linked_list(linked_list):
  """
    sorts linked list in ascending order
    - recursively divide the linked list into sublists containing a single node
    - repeatedly merge the sublists to product sorted sublists until one remains
    returns a sorted linked list
    runs in O(kn log n)
  """
  if linked_list.size() == 1:
    return linked_list
  elif linked_list.head is None:
    return linked_list
  left_half, right_half = split_linked_list(linked_list)
  left = merge_sort_linked_list(left_half)
  right = merge_sort_linked_list(right_half)

  return merge_linked_list(left, right)

def split_linked_list(linked_list):
  """
    divide the unsorted list at midpoint into sublists
    takes O(k log n)
  """
  if linked_list == None or linked_list.head == None:
    left_half, right_half = linked_list, None

    return left_half, right_half
  else:
    size = linked_list.size()
    midpoint = size // 2
    mid_node = linked_list.node_at_index(midpoint - 1)
    left_half = linked_list
    right_half = LinkedList()
    right_half.head = mid_node.next_node
    mid_node.next_node = None

    return left_half, right_half
  
def merge_linked_list(left, right):
  """
    merges two linked lists sorting by data in the nodes and returns a new merge linked list
    runs in O(n) time
  """
  # create a new linked list that contains node from merging left and right
  merged = LinkedList()

  # add a fake head that is discarded later
  merged.add(0)
  # set current to the head of the linked list
  current = merged.head
  # obtain head nodes for left and right linked lists
  left_head = left.head
  right_head = right.head

  # iterate over left and right until we reach the tail node of either
  while left_head or right_head:
    # if the head node of left is None, we're past the tail
    # add the node from right to merged linked list
    if left_head is None:
      current.next_node = right_head
      # call next on right to set loop condition to False
      right_head = right_head.next_node
    # if the head node of right is None, we're past the tail
    # add the tail node from left to merged linked list
    elif right_head is None:
      current.next_node = left_head
      # call next on left to set loop condition to False
      left_head = left_head.next_node
    else:
      # not at either tail node, obtain node data to perform comparison operations
      left_data = left_head.data
      right_data = right_head.data
      # if data on left is less than right, set current to left node
      if left_data < right_data:
        current.next_node = left_head
        # move left head to next node
        left_head = left_head.next_node
      # if data on left is greater than right, set current to right node
      else:
        current.next_node = right_head
        # move right head to next node
        right_head = right_head.next_node
    # move current to next node
    current = current.next_node

  # discard fake head and set first merged node as head
  head = merged.head.next_node
  merged.head = head

  return merged


# new_linked_list = LinkedList()
# new_linked_list.add(10)
# new_linked_list.add(2)
# new_linked_list.add(44)
# new_linked_list.add(15)
# new_linked_list.add(200)


# print(new_linked_list)

# sorted_linked_list = merge_sort_linked_list(new_linked_list)

# print(sorted_linked_list)




# =================== END OF MERGE AND SORT FOR LINKED LIST ==========================
