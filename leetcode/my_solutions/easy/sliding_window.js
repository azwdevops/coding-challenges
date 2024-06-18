// ========= problem 643 =======
// int array nums of n elements, and int k
// find contingous subarray of length k that has max average value
// this is not working yet
const findMaxAverage = function (nums, k) {
  const numsLength = nums.length;
  if (k > numsLength) {
    return 0;
  } else if (k === numsLength) {
    return nums.reduce((acc, item) => acc + item, 0) / k;
  }
  let leftIndex = 0;
  let rightIndex = k;
  let maxAverage = -Infinity;
  let currentAverage = 0;
  while (rightIndex <= numsLength) {
    console.log(nums.slice(leftIndex, rightIndex));
    currentAverage = nums.slice(leftIndex, rightIndex).reduce((acc, item) => acc + item, 0) / k;
    maxAverage = Math.max(maxAverage, currentAverage);
    leftIndex += 1;
    rightIndex += 1;
  }
  return maxAverage;
};

// console.log(findMaxAverage([1, 12, -5, -6, 50, 3], 4));
// console.log(findMaxAverage([5], 1));
console.log(findMaxAverage([0, 1, 1, 3, 3], 4));
