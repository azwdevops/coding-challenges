// solution alternative 1
const depthFirst = (root) => {
  if (root === null) return [];
  const result = [];
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop();
    result.push(current.val);

    if (current.right) stack.push(current.right);
    if (current.left) stack.push(current.left);
  }
  return result;
};

// depthFirst(a);

// solution alternative 2
const depthFirst2 = (root) => {
  if (root === null) return [];
  const leftValues = depthFirst2(root.left);
  const rightValues = depthFirst2(root.right);

  return [root.val, ...leftValues, ...rightValues];
};

const treeIncludes = (root, target) => {
  if (root === null) return false;
  if (root.val === target) return true;
  return treeIncludes(root.left, target) || treeIncludes(root.right, target);
};

const treeSum = (root) => {
  if (root === null) return 0;
  return root.val + treeSum(root.left) + treeSum(root.right);
};

const treeMinValue = (root) => {
  let smallest = Infinity;
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop();
    if (current.val < smallest) {
      smallest = current.val;
    }
    if (current.left !== null) stack.push(current.left);
    if (current.right !== null) stack.push(current.right);
  }

  return smallest;
};

const treeMinValue2 = (root) => {
  if (root === null) return Infinity;
  const leftMin = treeMinValue2(root.left);
  const rightMin = treeMinValue2(root.right);

  return Math.min(root.val, leftMin, rightMin);
};

const maxPathValue = (root) => {
  if (root === null) return -Infinity;
  if (root.left === null && root.right === null) return root.val;
  const maxChildPathSum = Math.max(maxPathValue(root.left), maxPathValue(root.right));
  return root.val + maxChildPathSum;
};
