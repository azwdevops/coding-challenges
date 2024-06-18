# =========================== START OF QUESTION 208. Implement Trie (Prefix Tree) ====================================

class TrieSolution:

    def __init__(self):
        self.trie = {}
        

    def insert(self, word: str) -> None:
        self.trie[word] = word
        
    def search(self, word: str) -> bool:
        if self.trie.get(word):
            return self.trie[word]
        return False

    def startsWith(self, prefix: str) -> bool:
        length = len(prefix)
        for key, value in self.trie.items():
            if prefix == value[:length]:
                return value
        return False
    
# =========================== END OF QUESTION 208. Implement Trie (Prefix Tree) ====================================