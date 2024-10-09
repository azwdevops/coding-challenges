// "use strict";
const output = (function (x) {
  delete x;
  return x;
})(0);

// console.log(output);

function multiplicationTable(n, m) {
  for (let i = 1; i <= n; i++) {
    let row = "";
    for (let j = 1; j <= m; j++) {
      row += j * i + " ";
    }
    console.log(row);
    console.log("\n");
  }
}
// multiplicationTable(5, 10);

class Vehicle {
  constructor(fuel) {
    this.fuel = fuel;
  }
}
