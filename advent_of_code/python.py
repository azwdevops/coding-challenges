def twoSum(nums, target):
  my_dict = {}

  for item in nums:
    if item in my_dict:
      return item, my_dict[item]
    difference = target - item
    my_dict[difference] = item

  return -1

# print(twoSum([1721,979,366,299,675,1456], 2020))