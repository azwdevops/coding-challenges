let i = 0;

function timedCount() {
  // console.log(document); // is undefined
  // console.log(window); // is undefined
  // console.log(parent); // is undefined
  i = i + 1;
  setTimeout(() => {
    postMessage("This message is from a web worker and will display on the html page");
  }, 4000);
}

timedCount();
