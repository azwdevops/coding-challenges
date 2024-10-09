class Node {
  constructor(val, nextNode = null) {
    this.val = val;
    this.next = nextNode;
  }
}

class SolutionSortLinkedList {
  mergeSortedLists(list1, list2) {
    let dummyNode = new Node(-1);
    let temp = dummyNode;

    while (list1 !== null && list2 !== null) {
      if (list1.data <= list2.data) {
        temp.next = list1;
        list1 = list1.next;
      } else {
        temp.next = list2;
        list2 = list2.next;
      }
      temp = temp.next;
    }
    if (list1 !== null) {
      temp.next = list1;
    } else {
      temp.next = list2;
    }
    return dummyNode.next;
  }

  findMiddle(node) {
    if (head === null || head.next === null) {
      return head;
    }
    let slow = node;
    let fast = node;

    while (fast && fast.next) {
      slow = slow.next;
      fast = fast.next.next;
    }
    return slow;
  }
  sortLinkedList(head) {
    let middle = this.findMiddle(head);
    let right = middle.next;
    middle.next = null;
    let left = head;

    leftList = this.sortLinkedList(left);
    rightList = this.sortLinkedList(right);

    return this.mergeSortedLists(leftList, rightList);
  }
}
