const removeDuplicates = function (arr) {
  if (arr.length === 0) {
    return 0;
  }
  let i = 0;
  const new_arr = [];
  for (let j = 0; j < arr.length; j++) {
    if (arr[j] !== arr[i]) {
      i += 1;
      arr[i] = arr[j];
    }
  }
  return i + 1;
};

console.log(removeDuplicates([1, 1, 2, 2, 3, 4, 4]));
