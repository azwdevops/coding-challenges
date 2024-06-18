if (typeof w == "undefined") {
  w = new Worker("counter.js");
}

// console.log(parent);
// console.log(window);

w.onmessage = function (e) {
  document.getElementById("worker").innerHTML = e.data;
};

// we use terminate to stop a web worker
// w.terminate();
