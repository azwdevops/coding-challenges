class Node {
  constructor() {
    this.links = new Array(26);
    this.endOfWord = false;
  }
  containsKey(char) {
    return this.links[char.charCodeAt(0) - "a".charCodeAt(0)] != undefined;
  }

  put(char, node) {
    this.links[char.charCodeAt(0) - "a".charCodeAt(0)] = node;
  }
  get(char) {
    return this.links[char.charCodeAt(0) - "a".charCodeAt(0)];
  }
  setEnd() {
    this.endOfWord = true;
  }

  isEnd() {
    return this.endOfWord;
  }
}

class Trie {
  constructor() {
    this.root = new Node();
  }

  insert(word) {
    let node = this.root;
    for (let i = 0; i < word.length; i++) {
      if (!node.containsKey(word[i])) {
        node.put(word[i], new Node());
      }
      // moves to the reference trie
      node.get(word[i]);
    }
    node.setEnd();
  }

  search(word) {
    let node = this.root;
    for (let i = 0; i < word.length; i++) {
      if (!node.containsKey(word[i])) {
        return false;
      }
      node = node.get(word[i]);
    }

    return node.isEnd();
  }

  startsWith(prefix) {
    let node = this.root;
    for (let i = 0; i < word.length; i++) {
      if (!node.containsKey(word[i])) {
        return false;
      }
      node = node.get(word[i]);
    }
    return true;
  }
}
