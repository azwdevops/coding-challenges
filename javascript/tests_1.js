// reverse a string
const reverseString = (my_string) => {
  // option 1
  let my_reversed_string = "";
  for (let character of my_string) {
    my_reversed_string = character + my_reversed_string;
  }
  console.log(my_reversed_string);

  // option 2
  const arr = my_string.split("");
  arr.reverse();
  my_reversed_string = arr.join("");

  console.log(my_reversed_string);
};

// reverseString("Greetings!");

// check if string is palidrome
const isPalidrome = (my_string) => {
  const my_reversed_string = my_string.split("").reverse().join("");

  console.log(my_string === my_reversed_string ? "String is palidrome" : "String is not palidrome");

  // console.log(my_string)
  // console.log(rever)
};

// isPalidrome("racecafgfgfr");

// check if a string is a palidrome permutation, note we are ignoring the case

const palidromePermutation = function (str) {
  let str_arr = str.toLowerCase().split(" ").join("").split("");

  for (const item of str_arr) {
    const currentCharArray = str_arr.filter((currentItem) => currentItem === item);
    if (currentCharArray.length % 2 === 0) {
      str_arr = str_arr.filter((currentItem) => currentItem !== item);
    }
  }
  return str_arr.length > 1 ? console.log("String is not a palidrome permutation") : console.log("String is a palidrome permutation");
};

// palidromePermutation("tact coa");
// palidromePermutation("TPOPTO");
// palidromePermutation("PUIPIP");
// palidromePermutation("Tact Coat");
// palidromePermutation("no lemon no melon");
// palidromePermutation("A man a plan a canal Panama");

const oneWayAlgorithm = function (str1, str2) {
  // new_str1 should be the longer string
  let new_str1, new_str2;
  if (str1.length > str2.length) {
    new_str1 = str1;
    new_str2 = str2;
  } else {
    new_str1 = str2;
    new_str2 = str1;
  }
  for (const char of new_str2) {
    if (new_str1.includes(char)) {
      new_str1 = new_str1.replace(`${char}`, "");
    }
  }
  return str1.length > 1 && new_str1.length === 1 ? console.log("One way algorithm") : console.log("Not one way algorithm");
};

// oneWayAlgorithm("pale", "ple");
// oneWayAlgorithm("pales", "pale");
// oneWayAlgorithm("pale", "kale");
// oneWayAlgorithm("pale", "pales");
// oneWayAlgorithm("pale", "bake");

// function to compress a string checking for continous characters
const compressString1 = function (str) {
  let currentCount = 1;
  let new_str = "";
  let uniqueCount = 0;
  for (let i = 0; i < str.length; i++) {
    if (str[i] === str[i + 1]) {
      currentCount++;
    } else {
      new_str += str[i] + currentCount;
      currentCount = 1;
      uniqueCount++;
    }
  }
  if (uniqueCount === str.length) {
    return str;
  } else {
    return new_str;
  }
};

// console.log(compressString1("aabcccccaaa"));
// console.log(compressString1("abcd"));

const zeroMatrix = function (matrix, n) {
  // mark the positions
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (matrix[r][c] === 0) {
        matrix[r][c] = true;
      }
    }
  }
  // find the locations and zero them
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (matrix[r][c] === true) {
        // zero the row
        for (let i = 0; i < n; i++) {
          matrix[r][i] = 0;
        }

        // zero the column
        for (let i = 0; i < n; i++) {
          matrix[i][c] = 0;
        }
      }
    }
  }
  return matrix;
};

// console.log(
//   zeroMatrix(
//     [
//       [4, 0, 6],
//       [2, 8, 1],
//       [5, 9, 2],
//     ],
//     3
//   )
// );
// console.log(
//   zeroMatrix(
//     [
//       [4, 1, 3],
//       [2, -4, 0],
//       [5, 9, 2],
//     ],
//     3
//   )
// );
// console.log(
//   zeroMatrix(
//     [
//       [5, 2, 0],
//       [9, 0, 1],
//       [2, 9, 3],
//     ],
//     3
//   )
// );

// scoping problem
// var x = Math.floor(Math.random());
// if (x > 0.5) {
//   var x = 1;
// } else {
//   x = 2;
// }

// console.log(x);

// flattening array
// console.log(
//   [
//     [5, 2, 0],
//     [9, 0, 1],
//     [2, 9, 3],
//   ].flat()
// );

// find the largest prime number in an array
const largestPrimeNumber = function (arr) {
  const primeTesters = [2, 3, 5, 7];
  let largestPrime = false;
  const filteredNumbers = arr
    .filter(
      (number) =>
        (number % 2 !== 0 || number === 2) &&
        (number % 3 !== 0 || number === 3) &&
        (number % 5 !== 0 || number === 5) &&
        (number % 7 !== 0 || number === 7)
    )
    .sort((a, b) => a - b);
  return filteredNumbers.length > 0 ? filteredNumbers.at(-1) : "No prime number found";
};
// const arr = [4, 5, 7, 8, 9, 11, 2, 12, 121, 17, 97, 47];

// console.log(largestPrimeNumber(arr));

const removeDuplicates = function (arr) {
  return Array.from(new Set(arr));
};

// const my_arr = ["Mike", "John", "Nancy", "Thomas", "Nancy", "Peter", "Mike"];
// console.log(removeDuplicates(my_arr));

const intArray = [1, 2, 3];
intArray[50] = 50;
// console.log(intArray.length);
// console.log(intArray[50]);

const generateAlphaNumberic = function (newLength) {
  const aplhas = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxy";
  const numeric = "0123456789";
  const alphaNumeric = aplhas + numeric;
  let captcha = "";
  const strLength = alphaNumeric.length;
  for (let i = 0; i < newLength; i++) {
    captcha += alphaNumeric[Math.floor(Math.random() * strLength)];
  }
  return captcha;
};

// console.log(generateAlphaNumberic(10));

// const result = Array.from({ length: 13 }, (v, i) => i);
// console.log(result);

let place = "US";
let department = { [getStudentPlace()]: "India", [place + "Department"]: "English", standard: 100 };
function getStudentPlace() {
  return place;
}
let division = "standard";
let { [getStudentPlace()]: placeDetails, [place + "Department"]: departmentDetails, [division]: standard } = department;

// console.log(placeDetails + " & " + departmentDetails + " & " + standard);

const num1 = [[1], [2]];
const num2 = [3, [4]];
const num3 = 5;
// console.log([[...num1[0], num3], [...num1[1]], ...num2]);

// const arr = Array(2).fill({});
// arr[1].product = "laptop";
// console.log(arr);

// const arr = [1, 2, 3];
// const newArray = [];
// for (const [item, index] of arr.entries()) {
//   newArray.push(item + index);
// }

// console.log(newArray);

const compose = function (functions) {
  return function (x) {
    if (functions.length === 0) {
      return x;
    } else if (functions.length === 1) {
      return functions[0](x);
    }
    let functionParameterValue = functions[functions.length - 1](x);
    for (let i = functions.length - 1; i > 0; i--) {
      functionParameterValue = functions[i - 1](functionParameterValue);
    }
    return functionParameterValue;
  };
};

// functions = [(x) => x + 1, (x) => x * x, (x) => 2 * x];
// console.log(compose(functions)(4));

const memoize = function (fn) {
  const cache = {};
  return function (...args) {
    const key = JSON.stringify(args);
    if (cache.hasOwnProperty(key)) {
      console.log("cached value");
      return cache[key];
    } else {
      console.log("calculation run");
      const result = fn.apply(this, args);
      cache[key] = result;
      return result;
    }
  };
};

// Given an object or an array, return if it is empty.

const isEmpty = function (obj) {
  if (obj.length && obj.length === 0) {
    return true;
  } else if (!obj.length && Array.from(Object.entries(obj)).length === 0) {
    return true;
  } else {
    return false;
  }
};

// console.log(isEmpty([4]));

const chunk = function (arr, size) {
  const chunkedArray = [];
  let remainingElementsCount = arr.length;
  let currentIndex = 0;
  while (remainingElementsCount >= 0) {
    if (arr.length === 0) {
      break;
    } else if (currentIndex + size < arr.length) {
      chunkedArray.push(arr.slice(currentIndex, currentIndex + size));
      remainingElementsCount -= size;
      currentIndex += size;
    } else {
      chunkedArray.push(arr.slice(currentIndex, arr.length));
      remainingElementsCount = 0;
      currentIndex = arr.length - 1;
      break;
    }
  }
  return chunkedArray;
};

console.log(chunk([1, 2, 3, 4, 5], 1));
