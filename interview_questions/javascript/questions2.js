"use strict";

const test = function () {
  const arr = [1, 2, 1, 3, 2, 6, 4, 2, 1, 3, 5, 7];
  const arr2 = arr.filter((value, index) => arr.indexOf(value) === Number(index));
  const arr3 = [1, 2, 3];
  // console.log(arr2);
  // console.log(arr.slice(2));
  console.log(arr3.splice(3, 1, 4));
  console.log(arr3);
};
// test();

const test2 = function () {
  const url = "employee?name=azw&occupation=ssoftware engineer";
  const encodedURL = encodeURI(url);
  console.log(encodedURL);
  const decodedURL = decodeURI(encodedURL);
  console.log(decodedURL);
};

// test2();

const usingCallbacks = () => {
  setTimeout(() => {
    console.log("timeout executed");
  }, 0);
  console.log("without timeout");
};

// usingCallbacks();

const object_operations = () => {
  const person = {
    name: "azw",
  };
  // console.log(person.age);
  console.log(Object.keys(person).length);
};

// object_operations();

const test3 = () => {
  // const str_arr = "capitalize";
  // console.log(str_arr[0].toUpperCase() + str_arr.slice(1));
  const dateObj = new Date();
  // const month = String(dateObj.getMonth() + 1).padStart(2, "0");
  // console.log(`${month}/${dateObj.getDate()}/${dateObj.getFullYear()}`);
  // console.log(dateObj.getTime()); // this gives a timestamp
  // console.log("test".startsWith("te"));
  const newObj = {
    name: "John",
  };
  newObj.last_name = "Doe";
  // console.log(newObj);
};

// test3();

const sortStr = function (arr) {
  return arr.sort();
};
const sortNumbers = function (arr) {
  return arr.sort((a, b) => a - b);
};

// console.log(sortStr(["b", "e", "f", "a", "c"]));
// console.log(sortNumbers([6, 2, 5, 177, 9, 1, 3, 4]));

class Vehicle {
  constructor(fuel) {
    this.fuel = fuel;
  }
}

class Toyota extends Vehicle {
  constructor(fuel, model) {
    super(fuel);
    this.model = model;
  }
}

// const car = new Toyota("petrol", "LC300");
// console.log(car);

// const newPrototype = {};
// const newObject = Object.create(newPrototype);
// console.log(Object.getPrototypeOf(newObject)); // true

// const person = {};
// Object.defineProperty(person, "firstName", {
//   value: "test",
// });
// console.log(person.firstName);

// const arr1 = [1, 2, 3, 4];
// const arr2 = [6, 7, 8];
// console.log([...arr1, ...arr2]);

// const obj = {
//   name: "azw",
//   test: "test",
//   test1: "test1",
// };

// const { name, ...rest } = obj;
// console.log(rest);

function test4() {
  console.log("A");
  setTimeout(function print() {
    console.log("B");
  }, 0);
  console.log("C");
}
// test4();

// console.log(true && 1);
// console.log(
//   "#" +
//     Math.floor(Math.random() * 0xffffff)
//       .toString(16)
//       .padStart(6, "0")
// );

// const numbers = [11, 25, 31, 23, 33, 18, 200];
// numbers.sort();
// console.log(numbers);

// let message = "Hello World!";
// message[0] = "J";
// console.log(message);

// let x = 7;
// let y = !!x && !!!x;
// console.log(y);

// const x = [31, 2, 8, 200];
// x.sort();
// console.log(x);

// let a = [1, 2, 3];
// let b = [4, 5, 6];

// console.log(a + b);

// const obj1 = {
//   name: "azw",
//   name: "zach",
// };

// console.log(obj1.name);

// const s = "racecar";
// s_reverse = s.split().reverse().join("");
// console.log(s_reverse);

// const d = {};
// ["zebra", "horse"].forEach(function (k) {
//   d[k] = undefined;
// });
// console.log(d);

// function getPersonInfo(one, two, three, four) {
//   console.log(one);
//   console.log(two);
//   console.log(three);
//   console.log(four);
// }

// const person = "Lydia";
// const age = 21;

// getPersonInfo`${person} is ${age} years old`;

// function* generator(i) {
//   yield i;
//   yield i * 2;
//   yield i * 2;
// }

// const gen = generator(10);
// console.log(gen.next().value);
// console.log(gen.next().value);
// console.log(gen.next().value);

// (function () {
//   console.log("IIFE function this");
// })();
