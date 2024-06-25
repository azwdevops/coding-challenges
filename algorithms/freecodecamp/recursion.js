const reverse_string = function (str) {
  if (str === "") {
    return "";
  }
  return reverse_string(str.slice(1)) + str[0];
};

// console.log(reverse_string("hello there and welcome to recursion"));

const can_sum = function (target, nums, cache = {}) {
  if (target === 0) return true;
  if (target < 0) return false;
  if (target in cache) {
    return cache[target];
  }
  for (const item of nums) {
    const remainder = target - item;
    cache[target] = can_sum(remainder, nums, cache);
    if (cache[target] === true) {
      return true;
    }
  }
  cache[target] = false;
  return false;
};

// console.log(can_sum(7, [5, 3, 4, 7]));
// console.log(can_sum(7, [2, 4]));
// console.log(can_sum(300, [7, 14]));

const how_sum = (target, nums, cache = {}) => {
  if (target === 0) {
    return [];
  }
  if (target < 0) {
    return null;
  }
  if (target in cache) return cache[target];

  for (const item of nums) {
    const remainder = target - item;
    const remainderResult = how_sum(remainder, nums, cache);

    if (remainderResult !== null) {
      currentResult = [...remainderResult, item];
      cache[target] = currentResult;
      return currentResult;
    }
  }
  cache[target] = null;
  return null;
};

// console.log(how_sum(7, [5, 3, 4, 7]));
// console.log(how_sum(7, [2, 4]));
// console.log(how_sum(300, [7, 14]));

const best_sum = (target, nums, cache = {}) => {
  if (target === 0) return [];
  if (target < 0) return null;

  if (target in cache) {
    return cache[target];
  }

  let shortestCombination = null;

  for (const num of nums) {
    const remainder = target - num;
    const remainderCombination = best_sum(remainder, nums, cache);
    if (remainderCombination !== null) {
      const combination = [...remainderCombination, num];
      cache[target] = combination;
      // if the combination is shorten than the current shortestCombination update it
      if (shortestCombination === null || combination.length < shortestCombination.length) {
        shortestCombination = combination;
      }
    }
  }
  cache[target] = shortestCombination;
  return shortestCombination;
};

// console.log(best_sum(7, [5, 3, 4, 7]));
// console.log(best_sum(8, [2, 3, 5]));
// console.log(best_sum(8, [1, 4, 5]));
// console.log(best_sum(100, [1, 2, 5, 25]));

const can_construct = function (target, wordBank, cache = {}) {
  if (target === "") return true;
  if (target in cache) {
    return cache[target];
  }
  for (const word of wordBank) {
    if (target.indexOf(word) === 0) {
      const suffix = target.slice(word.length);
      if (can_construct(suffix, wordBank, cache) === true) {
        cache[target] = true;
        return true;
      }
    }
  }
  cache[target] = false;
  return false;
};

// console.log(can_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(can_construct("leetcode", ["leet", "code"]));

const count_construct = function (target, wordBank, cache = {}) {
  if (target === "") return 1;
  if (target in cache) {
    return cache[target];
  }

  let totalCount = 0;
  for (const word of wordBank) {
    if (target.indexOf(word) === 0) {
      const result = count_construct(target.slice(word.length), wordBank, cache);
      totalCount += result;
    }
  }
  cache[target] = totalCount;
  return totalCount;
};

// console.log(count_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(count_construct("purple", ["purp", "p", "ur", "le", "purpl"]));

const all_construct = function (target, wordBank, memo = {}) {
  if (target === "") return [[]];
  if (target in memo) return memo[target];
  const variations = [];
  for (const word of wordBank) {
    if (target.indexOf(word) === 0) {
      const result = all_construct(target.slice(word.length), wordBank, memo);
      const targetWays = result.map((way) => [word, ...way]);
      variations.push(...targetWays);
    }
  }
  memo[target] = variations;
  return variations;
};

// console.log(all_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(all_construct("purple", ["purp", "p", "ur", "le", "purpl"]));
