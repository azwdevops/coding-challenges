const breath_first = (root) => {
  if (root === null) return [];

  const values = [];
  const queue = [root];

  while (queue.length > 0) {
    const current = queue.shift();
    values.push(current.val);
    if (current.left !== null) queue.push(current.left);
    if (current.right !== null) queue.push(current.right);
  }
  return values;
};

const treeIncludes = (root, target) => {
  if (root === null) return false;
  const queue = [root];
  while (queue.length > 0) {
    const current = queue.shift();

    if (current.val === target) return true;
    if (current.left) queue.push(current.left);
    if (current.right) queue.push(current.right);
  }
  return false;
};

const treeSum = (root) => {
  if (root === null) return 0;
  let totalSum = 0;
  const queue = [root];
  while (queue.length > 0) {
    const current = queue.shift();
    totalSum += current.val;
    if (current.left !== null) queue.push(current.left);
    if (current.right !== null) queue.push(current.right);
  }
  return totalSum;
};

const treeMinValue = (root) => {
  let smallest = Infinity;
  const queue = [root];
  while (stack.length > 0) {
    const current = queue.shift();
    if (current.left !== null) queue, push(current.left);
    if (current.right !== null) queue, push(current.right);
  }
  return smallest;
};
