# ======================== START OF QUESTION 1 ===========================
def maximum_sum(nums, k):
  max_sum = float('-inf')
  for i in range(len(nums) - k):
    current_sum = sum(nums[i:k+1])
    if current_sum > max_sum:
      max_sum = current_sum
  return max_sum


print(maximum_sum([-1,2,3,1,-3,2], 2))


# ========================= END OF QUESTION 1 ============================