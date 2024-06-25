const can_sum = (target, nums) => {
  const table = Array(target + 1).fill(false);
  table[0] = true;
  for (let i = 0; i < nums.length; i++) {
    if (table[i] === true) {
      for (const num of nums) {
        table[i + num] = true;
      }
    }
  }
  return table[target];
};

// console.log(can_sum(7, [5, 3, 4, 7]));
// console.log(can_sum(7, [2, 4]));
// console.log(can_sum(300, [7, 14]));

const can_construct = (target, wordBank) => {
  const table = Array(target.length + 1).fill(false);
  table[0] = true;
  for (let i = 0; i < target.length; i++) {
    if (table[i] === true) {
      for (const word of wordBank) {
        if (target.slice(i, i + word.length) === word) {
          table[i + word.length] = true;
        }
      }
    }
  }
  return table[target.length];
};

console.log(can_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
console.log(can_construct("leetcode", ["leet", "codde"]));

// console.log(count_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(count_construct("purple", ["purp", "p", "ur", "le", "purpl"]));

// console.log(count_sum(7, [2, 3]));
// console.log(count_sum(7, [5, 3, 4, 7]));
// console.log(count_sum(7, [2, 4]));
// console.log(count_sum(8, [2, 3, 5]));
// console.log(count_sum(300, [7, 14]));

// console.log(how_sum(7, [2, 3]));
// console.log(how_sum(7, [5, 3, 4, 7]));
// console.log(how_sum(7, [2, 4]));
// console.log(how_sum(8, [2, 3, 5]));
// console.log(how_sum(300, [7, 14]));

// console.log(how_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(how_construct("purple", ["purp", "p", "ur", "le", "purpl"]));

// console.log(best_sum(7, [5, 3, 4, 7]));
// console.log(best_sum(7, [5, 3, 1]));
// console.log(best_sum(8, [2, 3, 5]));
// console.log(best_sum(8, [1, 4, 5]));
// console.log(best_sum(100, [1, 2, 5, 25]));

// console.log(best_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(best_construct("purple", ["purp", "p", "ur", "le", "purpl"]));

// console.log(all_sum(7, [2, 3]));
// console.log(all_sum(7, [5, 3, 4, 7]));
// console.log(all_sum(7, [2, 4]));
// console.log(all_sum(8, [2, 3, 5]));
// console.log(all_sum(300, [7, 14]));

// console.log(all_construct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(all_construct("purple", ["purp", "p", "ur", "le", "purpl"]));
