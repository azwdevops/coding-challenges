import heapq


# couple of points to note
# parent_index = (child_index - 1) // 2
# left child index = ((2 * parent_index) + 1)
# right child index = ((2 * parent_index) + 2)


# min heap implentation
class MinHeap:
    def __init__(self, capacity):
        self.storage = [0] * capacity
        self.capacity = capacity
        self.size = 0

    def get_parent_index(self, index):
        return (index - 1) // 2

    def get_left_child_index(self, index):
        return (2 * index) + 1

    def get_right_child_index(self, index):
        return (2 * index) + 2

    def has_parent(self, index):
        return self.get_parent_index(index) >= 0

    def has_left_child(self, index):
        return self.get_left_child_index(index) < self.size

    def has_right_child(self, index):
        return self.get_right_child_index(index) < self.size

    def get_parent(self, index):
        return self.storage[self.get_parent_index(index)]

    def get_left_child(self, index):
        return self.storage[self.get_left_child_index(index)]

    def get_right_child(self, index):
        return self.storage[self.get_right_child_index(index)]

    def isFull(self):
        return self.size == self.capacity

    def swap(self, index1, index2):
        self.storage[index1], self.storage[index2] = self.storage[index2], self.storage[index1]

    def insert(self, data):
        if self.isFull():
            raise "Heap if full"
        self.storage[self.size] = data
        self.size += 1
        self.heapifyUpIterative()
        # self.heapifyUpRecursive(self.size - 1)

    # heapifyUp when using iteration
    def heapifyUpIterative(self):
        index = self.size - 1
        while self.has_parent(index) and self.get_parent(index) > self.storage[index]:
            self.swap(self.get_parent_index(index), index)
            index = self.get_parent_index(index)

    # heapifyUp when using recursion
    def heapifyUpRecursive(self, index):
        if self.has_parent(index) and self.get_parent(index) > self.storage[index]:
            self.swap(self.get_parent_index(index), index)
            self.heapifyUpRecursive(self.get_parent_index(index))

    def remove(self):
        if self.size == 0:
            raise "empty heap"
        data = self.storage[0]
        self.storage[0] = self.storage[self.size - 1]
        self.size -= 1
        self.heapifyDown()
        return data

    def heapifyDownIterative(self):
        index = 0
        while self.has_left_child(index):
            smallerChildIndex = self.get_left_child_index(index)
            if self.has_right_child(index) and self.get_right_child(index) < self.get_left_child(index):
                smallerChildIndex = self.get_right_child_index(index)
            if self.storage[index] < self.storage[smallerChildIndex]:
                break
            else:
                self.swap(index, smallerChildIndex)
            index = smallerChildIndex
