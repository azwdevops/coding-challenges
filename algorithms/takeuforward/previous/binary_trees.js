class TreeNode {
  constructor(val) {
    this.val = val;
    this.left = null;
    this.right = null;
  }
}

class SolutionTraverseBoundary {
  // function to check if a node is a leaf
  isLeaf(node) {
    return !node.left && !node.right;
  }

  // function to add the left boundary of the tree
  addLeftBoundary(node, result) {
    let current = node.left;
    while (current) {
      // if the current node is not a leaf, add its value to the result
      if (!this.isLeaf(current)) {
        result.push(current.val);
      }
      // move to the left child if it exists, otherwise move to the right child
      if (current.left) {
        current = current.left;
      } else {
        current = current.right;
      }
    }
  }

  addRightBoundary(node, result) {
    let current = node.right;
    const temp = [];
    while (current) {
      // ensure the current node is not a leaf, and add it's value to the temp array
      if (!this.isLeaf(current)) {
        temp.unshift(current.val);
      }
      // move to the right child if it exists, else move to the left
      if (current.right) {
        current = current.right;
      } else {
        current = current.left;
      }
    }
    result.push(...temp);
  }
  // function to add leaves of the tree
  addLeaves(node, result) {
    if (this.isLeaf(node)) {
      result.push(node.val);
      return;
    }
    // recursively add leaves of the left and right subtrees
    if (node.left) {
      this.addLeaves(node.left, result);
    }
    if (node.right) {
      this.addLeaves(node.right, result);
    }
  }
  traverseBoundary(root) {
    const result = [];
    if (!root) {
      return result;
    }
    // if the root is not a leaf, add it's value to the result
    if (!this.isLeaf(root)) {
      result.push(root.val);
    }
    // add the left boundary, leaves and right boundary in order
    this.addLeftBoundary(root, result);
    this.addLeaves(root, result);
    this.addRightBoundary(root, result);

    return result;
  }
}

class SolutionVerticalOrderTraversal {
  // function to perform vertical order traversal and return a 2D array of node values
  verticalOrderTraversal(root) {
    // map to store nodes based on vertical and level information
    const nodes = new Map();
    // queue for BFS traversal, each element is an array containing node and its vertical and level information
    const todo = [];
    // push the root node with initial vertical and level values [0,0]
    todo.push([root, [0, 0]]);

    // BFS traversal
    while (todo.length > 0) {
      // retrieve the node and its vertical and level information from the front of the queue
      const [temp, [vertical, level]] = todo.shift();

      // insert the node value into the corresponding vertical and level in the map
      if (!nodes.has(x)) {
        nodes.set(x, new Map());
      }
      if (!nodes.get(x).has(y)) {
        nodes.get(x).set(y, new Set());
      }
      nodes.get(x).get(y).add(temp.data);

      // process left child
      if (temp.left) {
        todo.push([temp.left, [x - 1, y + 1]]); // x - 1  =>  move left in terms of vertical, y + 1 => move down in terms of level
      }

      // process the right child
      if (temp.right) {
        todo.push([temp.right, [x + 1, y + 1]]); // x + 1 => move right in terms of vertical, y + 1 => move down in terms of level
      }
    }
    // prepare the final result array by combining values from the map
    const result = [];
    for (const vertical of nodes) {
      const col = [];
      for (const [level, nodeValues] of nodes[vertical]) {
        // insert node values into the column array
        col.push(...nodeValues?.sort((a, b) => a - b));
      }
      result.push(col);
    }
    return result;
  }
}

// to return the top view of a binary tree
class SolutionTopView {
  topView(root) {
    const result = [];
    if (root === null) {
      return result;
    }
    // map to store the top view nodes based on their vertical positions
    const nodesMap = new Map();
    // queue for BFS traversal, each element is a pair containing node and it's vertical position
    const queue = [];
    // push the root node with it's vertical position 0 into the queue
    queue.push([root, 0]);

    // BFS traversal
    while (queue.length > 0) {
      const [node, line] = queue.shift();
      // if the vertical position is not already in the map, we add the nodes data to the map
      if (!nodesMap.has(line)) {
        nodesMap.set(line, node.data);
      }
      // process the left child
      if (node.left !== null) {
        // push the left child with a decreased vertical position into the queue
        queue.push([node.left, line - 1]);
      }

      // process the right child
      if (node.right !== null) {
        // push the right child with an increased vertical position into the queue
        queue.push([node.right, line + 1]);
      }
    }
    // transfer values from the map to the result vector
    for (let [key, value] of nodesMap) {
      result.push(value);
    }

    return result;
  }
}

const solutionLeftView = function (root) {
  const result = [];
  const recursiveCall = function (node, level) {
    if (node === null) {
      return;
    }
    if (level === result.length) {
      result.push(node.val);
    }
    recursiveCall(node.left, level + 1);
    recursiveCall(node.right, level + 1);
  };

  recursiveCall(root, 0);

  return result;
};

const isSymmetricTree = function (root) {
  if (root === null) {
    return true;
  }
  const isSameTree = function (node1, node2) {
    if (node1 === null || node2 === null) {
      return node1 === node2;
    }
    return node1.val === node2.val && isSameTree(node1.left, node2.right) && isSameTree(node1.right, node2.left);
  };

  return isSameTree(root.left, root.right);
};

const solutionAllRootToLeaf = function (root) {
  const result = [];

  const findPath = function (node, currentPath) {
    if (node === null) {
      return;
    }
    currentPath.push(node.val);
    if (node.right === null && node.left === null) {
      result.push(currentPath.join(" "));
    }
    findPath(node.left, currentPath);
    findPath(node.right, currentPath);

    currentPath.pop();
  };
  findPath(root, []);
  return result;
};

const rootToNodePath = function (root, x) {
  const result = [];
  const findPath = function (node, currentPath) {
    if (node === null) {
      return;
    }
    currentPath.push(node.data);
    if (node.data === x) {
      result.append(currentPath);
      return;
    }
    findPath(node.left, currentPath);
    findPath(node.right, currentPath);

    currentPath.pop();
  };

  findPath(root, x);

  return result[0];
};

const getMaxWidth = function (root) {
  if (!root) {
    return 0;
  }
  let result = 0;
  let queue = [{ node: root, position: 0 }];
  while (queue.length > 0) {
    const n = queue.length;
    let startIndex = queue[0].position;
    let first, last;

    for (let i = 0; i < n; i++) {
      let currentIndex = q[0].position - startIndex;
      let node = q[0].node;

      queue.shift();

      if (i === 0) {
        first = currentIndex;
      }
      if (i === n - 1) {
        last = currentIndex;
      }
      if (node.left) {
        queue.push({ node: node.left, position: currentIndex * 2 + 1 });
      }
      if (node.right) {
        queue.push({ node: node.right, position: currentIndex * 2 + 2 });
      }
    }
    result = Math.max(result, last - first + 1);
  }
  return result;
};

const changeTree = function (root) {
  if (root === null) {
    return null;
  }
  const recursiveCall = function (node) {
    if (node === null) {
      return;
    }
    // moving down the tree, update each child node (left and right) with the parent values
    if (node.left) {
      node.left.data = Math.max(node.left.data, node.data);
    }
    if (node.right) {
      node.right.data = Math.max(node.right.data, node.data);
    }
    // make the recursive call here
    recursiveCall(node.left);
    recursiveCall(node.right);

    // after hitting the base case, we no pop calls from the stack, we will update the parent with children's sum, since currently the children hold
    // the maximum value possible which was done before recursion above
    const leftChild = node.left ? node.left.data : 0;
    const rightChild = node.right ? node.right.data : 0;

    node.data = Math.max(node.data, leftChild + rightChild);
  };

  recursiveCall(root);

  return root;
};

// given preorder and inorder build binary tree
class SolutionBuildTreeFromPreInOrder {
  buildTree(preorder, inorder) {
    const inOrderMap = new Map();
    inorder.forEach((item, index) => inOrderMap.set(item, index));

    const root = this.buildTreeHelper(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1, inOrderMap);

    return root;
  }

  buildTreeHelper(preorder, preorderStart, preorderEnd, inorder, inorderStart, inorderEnd, inOrderMap) {
    if (preorderStart > preorderEnd || inorderStart > inorderEnd) {
      return null;
    }
    const root = new TreeNode(preorder[preorderStart]);
    const rootIndex = inOrderMap.get(root.val);
    const leftNumsCount = rootIndex - inorderStart;

    root.left = this.buildTreeHelper(preorder, preorderStart + 1, preorderStart + leftNumsCount, inorder, inorderStart, rootIndex - 1, inOrderMap);
    root.right = this.buildTreeHelper(preorder, preorderStart + leftNumsCount + 1, preorderEnd, inorder, rootIndex + 1, inorderEnd, inOrderMap);

    return root;
  }
}

class SolutionBuildTreeFromPostInOrder {
  buildTree(inorder, postorder) {
    if (inorder.length !== postorder.length) {
      return null;
    }
    const inMap = new Map();
    inorder.map((value, index) => inMap.set(value, index));

    return this.buildTreeHelper(inorder, 0, inorder.length - 1, postorder, 0, postorder.length - 1, inMap);
  }

  buildTreeHelper(inorder, inStart, inEnd, postorder, postStart, postEnd, inMap) {
    if (postStart > postEnd || inStart > inEnd) {
      return null;
    }
    const root = new TreeNode(postorder[postEnd]);
    const rootIndex = inMap.get(root.val);
    const numsLeft = rootIndex - inStart;

    root.left = this.buildTreeHelper(inorder, inStart, rootIndex - 1, postorder, postStart, postStart + numsLeft - 1, inMap);
    root.right = this.buildTreeHelper(inorder, rootIndex + 1, inEnd, postorder, postStart + numsLeft, postEnd - 1, inMap);

    return root;
  }
}

class SolutionSerializeDeserialize {
  serialize(root) {
    if (!root) {
      return "";
    }
    let tree_string = "";
    const queue = [];
    queue.push(root);

    while (queue.length > 0) {
      const current = queue.shift();
      if (current === null) {
        tree_string += "#,";
      } else {
        tree_string += current.val + ",";
        queue.push(current.left);
        queue.push(current.right);
      }
    }
    return tree_string;
  }

  deserialize(data) {
    if (data === "") {
      return null;
    }
    let data_list = data.split(",");
    let rootVal = data_list.shift();
    let rootNode = new TreeNode(parseInt(rootVal));
    const queue = [];
    queue.push(rootNode);
    while (queue.length > 0) {
      let node = queue.shift();
      let leftValue = data_list.shift();
      if (leftValue !== "#") {
        let leftNode = new TreeNode(parseInt(leftValue));
        node.left = leftNode;
        queue.push(leftNode);
      }
      let rightValue = data_list.shift();
      if (rightValue !== "#") {
        let rightNode = new TreeNode(parseInt(rightValue));
        node.right = rightNode;
        queue.append(rightNode);
      }
    }
    return node;
  }
}

const flattenTreeToLinkedList = function (root) {
  const stack = [];
  stack.push(root);
  while (stack.length > 0) {
    const node = stack.pop();
    if (node.right) {
      stack.append(node.left);
    }
    if (node.left) {
      stack.append(node.left);
    }
    node.left = null;
    if (stack.length > 0) {
      node.right = stack[stack.length - 1];
    }
  }
};

const searchBST = function (root, x) {
  while (root !== null && root.val != x) {
    root = x < root.val ? root.left : root.right;
  }
  return root;
};

const deleteNode = function (root, key) {
  const recursiveCall = function (node) {
    if (!node) {
      return None;
    }
    if (node.data < key) {
      node.right = deleteNode(node.right, key);
    } else if (node.data > key) {
      node.left = deleteNode(node.left, key);
    } else {
      if (!node.left && !node.right) {
        return None;
      } else if (!node.right) {
        return node.left;
      } else if (!node.left) {
        return node.right;
      } else {
        let temp = node.right;
        while (temp.left) {
          temp = temp.left;
        }
        temp.left = node.left;

        return node.right;
      }
    }
  };
  recursiveCall(root, key);
  return root;
};

class SolutionKthSmallest {
  inorder(node, count, k, kSmallest) {
    if (!node || count >= k) {
      return;
    }
    this.inorder(node.left, count, k, kSmallest);
    count++;
    if (count === k) {
      kSmallest = node.val;
      return;
    }
    this.inorder(node.right, count, k, kSmallest);
  }
  findKthSmallest(root, k) {
    let kSmallest = null;
    let count = 0;
    this.inorder(root, count, k, kSmallest);
    return kSmallest;
  }
}

const lowestLCA = function (root, p, q) {
  if (!root) {
    return null;
  }
  const currentVal = root.val;
  if (currentVal < p.val && currentVal < q.val) {
    return lowestLCA(root.right, p, q);
  } else if (currentVal > p.val && currentVal > q.val) {
    return lowestLCA(root.left, p, q);
  }
  return root;
};

const constructFromPreOrder = function (preorder) {
  const build = function (preorder, currentIndex, bound) {
    if (currentIndex === preorder.length || preorder[currentIndex] > bound) {
      return null;
    }
    const root = new TreeNode(preorder[currentIndex++]);
    root.left = build(preorder, currentIndex, root.val);
    root.right = build(preorder, currentIndex, bound);
    return root;
  };
  let currentIndex = 0;
  return build(preorder, currentIndex, Infinity);
};

const inOrderSuccessor = function (root, p) {
  let successor = null;
  while (root !== null) {
    if (p.val >= root.val) {
      root = root.right;
    } else {
      successor = root;
      root = root.left;
    }
  }
  return successor;
};

const inOrderPredecessorSuccessor = function (root, key) {
  let successor = null;
  let predecessor = null;
  let current = root;
  while (current !== null) {
    if (key < current.data) {
      successor = current;
      current = current.left;
    } else if (key > current.data) {
      predecessor = current;
      current = current.right;
    } else {
      if (current.left) {
        let temp = current.left;
        while (temp.right) {
          temp = temp.right;
        }
        predecessor = temp;
      }
      if (current.right) {
        let temp = temp.right;
        while (temp.left) {
          temp = temp.left;
        }
        successor = temp;
      }
      break;
    }
  }
  const predecessor_data = predecessor !== null ? predecessor.data : -1;
  const successor_data = successor !== null ? successor.data : -1;

  return predecessor_data, successor_data;
};

class BSTIterator {
  constructor(root) {
    this.stack = [];
    this.pushAll(root);
  }
  hasNext() {
    return this.stack.length !== 0;
  }
  next() {
    const node = this.stack.pop();
    this.pushAll(node.right);
    return node.val;
  }
  pushAll(node) {
    while (node !== null) {
      this.stack.push(node);
      node = node.left;
    }
  }
}

class BSTIteratorTwoSum {
  constructor(root) {
    this.stack = [];
    this.reverse = true;
  }

  next() {
    const node = this.stack.pop();
    if (this.reverse) {
      this.pushAll(node.right);
    } else {
      this.pushAll(node.left);
    }
    return node.data;
  }

  hasNext() {
    return this.stack.length > 0;
  }

  pushAll(node) {
    while (node) {
      this.stack.push(node);
      if (this.reverse) {
        node = node.right;
      } else {
        node = node.left;
      }
    }
  }
}

const twoSumBinarySearchTree = function (root, k) {
  if (!root) {
    return false;
  }
  const left = new BSTIteratorTwoSum(root, false);
  const right = new BSTIteratorTwoSum(root, true);
  let i = left.next();
  let j = right.next();
  while (i < j) {
    if (i + j == k) {
      return true;
    } else if (i + j < k) {
      i = left.next();
    } else {
      j = right.next();
    }
  }
  return false;
};

class SolutionRecoverBST {
  constructor() {
    this.first = null;
    this.prev = null;
    this.middle = null;
    this.last = null;
  }
  inorder(root) {
    if (root === null) {
      return;
    }
    this.inorder(root.left);
    if (this.prev !== null && root.val < this.prev.val) {
      // if this is the first violation, mark these two nodes as first and middle
      if (this.first === null) {
        this.first = prev;
        this.middle = root;
      }
      // if this is the second violation, mark this node as last
      else {
        this.last = root;
      }
    }
    // mark this node as previous
    this.prev = root;
    this.inorder(root.right);
  }

  recoverBST(root) {
    this.inorder(root);
    if (this.first !== null && this.last !== null) {
      const temp = this.first;
      this.first = this.last;
      this.last = temp;
    } else if (this.first !== null && this.middle !== null) {
      const temp = this.first;
      this.first = this.middle;
      this.middle = temp;
    }
  }
}

class NodeValue {
  constructor(minNode, maxNode, maxSize) {
    this.minNode = minNode;
    this.maxNode = maxNode;
    this.maxSize = maxSize;
  }
}

const largestBST = function (root) {
  const helper = function (node) {
    if (node === null) {
      return new NodeValue(Infinity, -Infinity, 0);
    }
    // recursively call for left and right subtrees
    const left = helper(root.left);
    const right = helper(root.right);

    // check if the current node is a valid BST node
    if (left.maxNode < node.val && node.val < right.minNode) {
      // this means the current subtree is a valid BST
      return new NodeValue(Math.min(left.minNode, node.val), Math.max(right.maxNode, node.val), right.maxSize + left.maxSize + 1);
    }
    // if it's not a BST, return an invalid range but keep the max size of the largest BST found
    return new NodeValue(Infinity, -Infinity, Math.max(right.maxSize, left.maxSize));
  };

  return helper(root).maxSize;
};
