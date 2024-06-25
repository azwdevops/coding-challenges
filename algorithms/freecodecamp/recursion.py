def reverse_string(str):
  if str == '':
    return ''
  return reverse_string(str[1:]) + str[0]

# print(reverse_string('hello there and welcome to recursion'))

def palidrome(str):
  if len(str) == 0 or len(str) == 1:
    return True
  if str[0] == str[len(str) - 1]:
    return palidrome(str[1:len(str) - 1])
  return False

# print(palidrome('kayak'))
# print(palidrome('kayaka'))

def decimal_to_binary(number, result=''):
  if number == 0:
    return result
  result = str(number % 2) + result
  return decimal_to_binary(number // 2, result)

# print(decimal_to_binary(233))

def recursiveSummation(num):
  if num <= 1:
    return num
  return num + recursiveSummation(num - 1)

# print(recursiveSummation(2))

def binary_search(nums, left, right, target):
  if left > right:
    return -1
  mid = (right + left) // 2
  if nums[mid] == target:
    return mid
  if target < nums[mid]:
    return binary_search(nums, left, mid - 1, target)
  
  return binary_search(nums, mid + 1, right, target)

# nums = [-1,0,1,2,3,4,7,9,10,20]
# print(binary_search(nums, 0, len(nums) - 1, 10))

class MergeSortSolution:
  def merge_sort(self, nums, start, end):
    if start < end:
      mid = (start + end) // 2
      self.merge_sort(nums, start, mid)
      self.merge_sort(nums, mid + 1, end)
      self.merge(nums, start, mid, end)
    return nums

  def merge(self, nums, start, mid, end):
    # build a temp array to avoid modifying original contents
    temp = [0] * (end - start + 1)
    i = start
    j = mid + 1 
    k = 0
    # while both sub-arrays have values, then try and merge them in sorted order
    while i <= mid and j <= end:
      if nums[i] <= nums[j]:
        temp[k] = nums[i]
        i += 1
        k += 1
      else:
        temp[k] = nums[j]
        k += 1
        j += 1
      
    #  add the rest of the values from the left subarray into the result
    while i <= mid:
      temp[k] = nums[i]
      k += 1
      i += 1

    #  add the rest of the values from the right subarray into the result
    while j <= end:
      temp[k] = nums[j]
      k += 1
      j += 1

    for i in range(start, end + 1):
      nums[i] = temp[i - start]

# nums = [-5,20,10,3,2,0]
# solution = MergeSortSolution()
# print(solution.merge_sort(nums, 0, len(nums) - 1))


def int_to_binary(n, count):
  if n == 1:
    return '1', count + 1
  remainder = n % 2
  if remainder == 1:
    count += 1
  result = int_to_binary(n//2, count)
  return result[0] + str(remainder), result[1]


# print(int_to_binary(1378, 0))
# print(int_to_binary(45264, 0))

def count_bits(n):
  ans = []
  def ones_in_num(num, count):
      if num == 0:
          return count
      if num % 2 == 1:
          count += 1
      return ones_in_num(num // 2, count)

  for num in range(n+1):
      ans.append(ones_in_num(num, 0))
  return ans


# print(count_bits(5))