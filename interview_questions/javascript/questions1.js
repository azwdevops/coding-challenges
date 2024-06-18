const test = function () {
  if (true) {
    var a = "variable";
    let b = "variable";
  }

  console.log(a);
  console.log(b);
};

// test();

const test2 = function () {
  const arr = [1, 2, 3, 5, 6];
  console.log(arr);
  arr.length = 0;
  console.log(arr);
};

const test3 = function () {
  let a = 10;
  if (function abc() {}) {
    a += typeof abc;
  }
  console.log(a);
};

const test4 = function () {
  const student = {
    college: "abc",
  };
  const student1 = Object.create(student);
  delete student1.college;

  console.log(student1.company);
};

const bar = function foo() {
  return 11;
};

// console.log(typeof foo());

const test5 = function () {
  let i = 0;
  do {
    console.log(i);
    i++;
  } while (i < 10);
};

// test5();

const test6 = function () {
  console.log("log 1");
  setTimeout(() => {
    console.log("log 2");
  }, 0);
  setTimeout(() => {
    console.log("log 3");
  }, 0);

  console.log("log 4");
};

// test6();

const typeCoersion = function () {
  const a = "10";
  const b = 5;
  console.log(a + b);
  console.log(a - b);
  console.log(a * b);
  console.log(a / b);
};

// typeCoersion();

const arr_operations = function () {
  const arr = [1, 2, 3, 4, 5, 6];

  // arr[0] = 0;
  // console.log(arr[0]);
  // console.log(arr.at(-2));
  arr.splice(2, 1);
  console.log(arr);
};

// arr_operations();

const sort_arr = function () {
  const arr = ["c", "e", "a", "t"];
  arr.sort();
  console.log(arr);
};

// sort_arr();

const arrayLikeToArray = function () {
  const str = "abajjadfjadlfad";
  // console.log(Array.from(str));
  // console.log([...str]);
  const obj = { 1: "assaads", 2: "addasgs" };
  // console.log(Object.entries(obj));
};

// arrayLikeToArray();

const argumentsFunc = function () {
  console.log(arguments[0].name);
};

// argumentsFunc({ name: "azw", profession: "software developer" });

const dom_operations = function () {
  console.log(document.getElementById("new"));
};

// dom_operations();

const object_operations = function () {
  const person = {
    name: "azw",
    profession: "software engineer",
    hobbies: {
      music: "guitar",
    },
  };
  // person.hobby = "guitar playing";
  // delete person.hobby;
  // const career = "profession";
  // console.log(person[career]);
  // console.log(person.hasOwnProperty("name"));
  // const person1 = Object.assign({}, person);
  // person1.hobbies.music = "zach";
  // console.log(person);
};

// object_operations();

// document.getElementById("new").addEventListener("click", () => {
//   console.log("parent clicked");
// });
// document.getElementById("btn").addEventListener("click", () => {
//   console.log("child clicked");
// });

const webStorage = () => {
  if (typeof Storage !== undefined) {
    console.log("Storage supported");
  } else {
    console.log("storage not supported");
  }
};
// webStorage();

const webWorkers = () => {
  if (typeof Worker !== undefined) {
    console.log("workers support");
  } else {
    console.log("workers not support");
  }
};

// webWorkers();

const sse_events = () => {
  if (typeof EventSource !== undefined) {
    console.log("supports sse");
  } else {
    console.log("sse not supported");
  }
};

// sse_events();
