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
