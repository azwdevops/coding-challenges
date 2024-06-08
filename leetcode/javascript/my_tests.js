// using decorators

// const splitString = function (target, key, descriptor) {
//   const originalMethod = descriptor.value;
//   descriptor.value = function (...args) {
//     args[0] = args[0].split("");
//     originalMethod.apply(this, args);
//   };
// };

// const reverseString = function (target, key, descriptor) {
//   const originalMethod = descriptor.value;
//   descriptor.value = function (...args) {
//     args[0] = args[0].reverse();
//     originalMethod.apply(this, args);
//   };
// };

// const joinString = function (target, key, descriptor) {
//   const originalMethod = descriptor.value;
//   descriptor.value = function (...args) {
//     args[0] = args[0].join("");
//     originalMethod.apply(this, args);
//   };
// };

// // decorators seem to work in typescript
// class StringManager {
//   // @reverseString
//   // @splitString
//   print(str) {
//     console.log(str);
//   }
// }

// const stringManager = new StringManager();
// stringManager.print("hello world");

// using function composition
function splitString(str) {
  return str.split("");
}

function reverseArr(arr) {
  return arr.reverse();
}

function joinArr(arr, joinChar = "") {
  return arr.join(joinChar);
}

function compose(...functions) {
  return (str) => functions.reduceRight((acc, currentFunction) => currentFunction(acc, "."), str);
}

const composedFunction = compose(joinArr, reverseArr, splitString);

console.log(composedFunction("hello"));
