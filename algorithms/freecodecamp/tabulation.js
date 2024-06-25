const fib = (n) => {
  const table = Array(n + 1).fill(0);
  table[1] = 1;
  for (let i = 0; i <= n; i++) {
    table[i + 1] += table[i];
    table[i + 2] += table[i];
  }
  return table[n];
};

// console.log(fib(6));
// console.log(fib(7));
// console.log(fib(8));
// console.log(fib(50));

const gridTraveler = (m, n) => {
  const table = Array(m + 1)
    .fill()
    .map(() => Array(n + 1).fill(0));
  table[1][1] = 1;
  for (let i = 0; i <= m; i++) {
    for (let j = 0; j <= n; j++) {
      const current = table[i][j];
      if (j + 1 <= n) table[i][j + 1] += current;
      if (i + 1 <= m) table[i + 1][j] += current;
    }
  }
  return table[m][n];
};

// console.log(gridTraveler(1, 1));
// console.log(gridTraveler(2, 3));
// console.log(gridTraveler(3, 2));
// console.log(gridTraveler(3, 3));
// console.log(gridTraveler(18, 18));

const canSum = (target, nums) => {
  const table = Array(target + 1).fill(false);
  table[0] = true;
  for (let i = 0; i <= target; i++) {
    if (table[i] === true) {
      for (const num of nums) {
        table[i + num] = true;
      }
    }
  }
  return table[target];
};

// console.log(canSum(7, [2, 3]));
// console.log(canSum(7, [5, 3, 4, 7]));
// console.log(canSum(7, [2, 4]));
// console.log(canSum(8, [2, 3, 5]));
// console.log(canSum(300, [7, 14]));

const howSum = (target, nums) => {
  const table = Array(target + 1).fill(null);
  table[0] = [];
  for (let i = 0; i <= target; i++) {
    if (table[i] !== null) {
      for (const num of nums) {
        table[i + num] = [...table[i], num];
      }
    }
  }
  return table[target];
};

// console.log(howSum(7, [2, 3]));
// console.log(howSum(7, [5, 3, 4, 7]));
// console.log(howSum(7, [2, 4]));
// console.log(howSum(8, [2, 3, 5]));
// console.log(howSum(300, [7, 14]));

const bestSum = (target, nums) => {
  const table = Array(target + 1).fill(null);
  table[0] = [];
  for (let i = 0; i <= target; i++) {
    if (table[i] !== null) {
      for (const num of nums) {
        const combination = [...table[i], num];
        // if this current combination is shorter than what is already stored
        if (!table[i + num] || table[i + num].length > combination.length) {
          table[i + num] = combination;
        }
      }
    }
  }
  return table[target];
};

// console.log(bestSum(7, [5, 3, 4, 7]));

const canConstruct = (target, wordBank) => {
  const table = Array(target.length + 1).fill(false);
  table[0] = true;

  for (let i = 0; i <= target.length; i++) {
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

// console.log(canConstruct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));

const countConstruct = (target, wordBank) => {
  const table = Array(target.length + 1).fill(0);
  table[0] = 1;
  for (let i = 0; i <= target.length; i++) {
    for (const word of wordBank) {
      if (target.slice(i, i + word.length) === word) {
        table[i + word.length] += table[i];
      }
    }
  }
  return table[target.length];
};

// console.log(countConstruct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(countConstruct("purple", ["purp", "p", "ur", "le", "purpl"]));

const allConstruct = (target, wordBank) => {
  const table = Array(target + 1).fill(null);
  table[0] = [[]];
  for (let i = 0; i <= target.length; i++) {
    for (const word of wordBank) {
      if (target.slice(i, i + word.length) === word) {
        // console.log(table[i]);
        if (!table[i] || table[i].length === 0) {
          table[i + word.length] = [word];
        } else {
          table[i + word.length] = table[i].map((arr) => {
            // console.log(arr);
            [...arr, word];
          });
        }
      }
    }
  }

  return table[target.length];
};

// console.log(allConstruct("abcdef", ["ab", "abc", "cd", "def", "abcd"]));
// console.log(allConstruct("purple", ["purp", "p", "ur", "le", "purpl"]));
