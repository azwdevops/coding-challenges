// can sum
const can_sum = (target, nums, memo = {}) => {
  if (target === 0) return true;
  if (target < 0) return false;
  if (target in memo) return memo[target];

  for (const num of nums) {
    if (can_sum(target - num, nums, memo) === true) {
      memo[target] = true;
      return true;
    }
  }
  memo[target] = false;
  return false;
};

// console.log(can_sum(7, [5, 3, 4, 7]));
// console.log(can_sum(7, [2, 4]));
// console.log(can_sum(300, [7, 14]));

// can construct
const can_construct = (target, wordBank, memo) => {
  if (target === "") return true;
  if (target in memo) return memo[target];
  for (const word of wordBank) {
    if (target.indexOf(word) === 0) {
      if (can_construct(target.slice(word.length), wordBank, memo)) {
        memo[target] = true;
        return true;
      }
    }
  }
  memo[target] = false;
  return false;
};

// console.log(can_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(can_construct("leetcode", ["leet", "code"]));

const count_construct = (target, wordBank) => {
  if (target === "") return 1;
  let count = 0;
  for (const word of wordBank) {
    if (target.indexOf(word) === 0) {
      const result = count_construct(target.slice(word.length), wordBank);
      count += result;
    }
  }
  return count;
};

// console.log(count_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(count_construct("purple", ["purp", "p", "ur", "le", "purpl"]));

const count_sum = (target, nums, memo = {}) => {
  if (target === 0) return 1;
  if (target < 0) return 0;
  let count = 0;
  if (target in memo) return memo[target];
  for (let i = 0; i < nums.length; i++) {
    const result = count_sum(target - nums[i], nums.slice(i), memo);
    memo[target] = result;
    count += result;
  }
  return count;
};

// console.log(count_sum(7, [2, 3]));
// console.log(count_sum(7, [5, 3, 4, 7]));
// console.log(count_sum(7, [2, 4]));
// console.log(count_sum(8, [2, 3, 5]));
// console.log(count_sum(300, [7, 14]));

const how_sum = (target, nums, memo = {}) => {
  if (target === 0) return [];
  if (target < 0) return null;
  if (target in memo) return memo[target];
  for (const num of nums) {
    const result = how_sum(target - num, nums, memo);
    memo[target] = result;
    if (result !== null) {
      return [...result, num];
    }
  }
  memo[target] = null;
  return null;
};

// console.log(how_sum(7, [2, 3]));
// console.log(how_sum(7, [5, 3, 4, 7]));
// console.log(how_sum(7, [2, 4]));
// console.log(how_sum(8, [2, 3, 5]));
// console.log(how_sum(300, [7, 14]));

const how_construct = (target, wordBank) => {
  if (target === "") return [];

  for (const word of wordBank) {
    if (target.indexOf(word) === 0) {
      const result = how_construct(target.slice(word.length), wordBank);
      if (result !== null) {
        return [word, ...result];
      }
    }
  }
  return null;
};

// console.log(how_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(how_construct("purple", ["purp", "p", "ur", "le", "purpl"]));

// the minimum length of array to sum
const best_sum = (target, nums, memo = {}) => {
  if (target === 0) return [];
  if (target < 0) return null;
  if (target in memo) return memo[target];
  let shortestCombination = null;
  for (const num of nums) {
    const result = best_sum(target - num, nums, memo);
    if (result !== null) {
      if (shortestCombination === null || result.length < shortestCombination.length) {
        shortestCombination = [...result, num];
      }
    }
  }
  memo[target] = shortestCombination;
  return shortestCombination;
};

// console.log(best_sum(7, [5, 3, 4, 7]));
// console.log(best_sum(7, [5, 3, 1]));
// console.log(best_sum(8, [2, 3, 5]));
// console.log(best_sum(8, [1, 4, 5]));
// console.log(best_sum(100, [1, 2, 5, 25]));

const best_construct = (target, wordBank, memo = {}) => {
  if (target === "") return [];
  let shortestCombination = null;
  if (target in memo) return memo[target];
  for (const word of wordBank) {
    if (target.indexOf(word) === 0) {
      const result = best_construct(target.slice(word.length), wordBank, memo);
      if (result !== null) {
        if (!shortestCombination || result.length < shortestCombination.length) {
          shortestCombination = [word, ...result];
        }
      }
    }
  }
  memo[target] = shortestCombination;
  return shortestCombination;
};

// console.log(best_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(best_construct("purple", ["purp", "p", "ur", "le", "purpl"]));

const all_sum = (target, nums, memo = {}) => {
  if (target === 0) return [[]];
  if (target < 0) return [];
  if (target in memo) return memo[target];
  const result = [];

  for (let i = 0; i < nums.length; i++) {
    const currentResult = all_sum(target - nums[i], nums.slice(i), memo);
    if (currentResult.length > 0) {
      const updatedResult = currentResult.map((arr) => [...arr, nums[i]]);
      result.push(...updatedResult);
    }
  }

  memo[target] = result;
  return result;
};

// console.log(all_sum(7, [2, 3]));
// console.log(all_sum(7, [5, 3, 4, 7]));
// console.log(all_sum(7, [2, 4]));
// console.log(all_sum(8, [2, 3, 5]));
// console.log(all_sum(300, [7, 14]));

const all_construct = (target, wordBank, memo = {}) => {
  if (target === "") return [[]];
  const result = [];
  if (target in memo) return memo[target];

  for (const word of wordBank) {
    if (target.indexOf(word) === 0) {
      const currentResult = all_construct(target.slice(word.length), wordBank, memo);
      const updatedCurrentResult = currentResult.map((arr) => [word, ...arr]);
      result.push(...updatedCurrentResult);
    }
  }
  memo[target] = result;
  return result;
};

// console.log(all_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(all_construct("purple", ["purp", "p", "ur", "le", "purpl"]));
