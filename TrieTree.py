"""
字典树相关题目：
LeetCode208 实现前缀树
LeetCode648 单词替换
LeetCode211 添加与搜索单词
LeetCode677 键值映射
"""


import collections

# TrieSet LeetCode208 实现前缀树
# 只需要判断是否是结尾，不需要存储value
# TrieNode 写法
class TrieNode:

    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.isEnd = None


class Trie:

    def __init__(self):
        self.root = TrieNode()

    @staticmethod
    def insert(self, word):
        node = self.root
        for c in word:
            node = node.children[c]
        node.isEnd = True

    def search(self, word):
        node = self.root
        for c in word:
            node = node.children.get(c)
            if node is None:
                return False
        return node.isEnd

    def startswith(self, prefix):
        node = self.root
        for c in prefix:
            node = node.children.get(c)
            if node is None:
                return False
        return True


# 字典写法
class TrieDic:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['#'] = {}

    def search(self, word):
        node = self.root
        for c in word:
            if c not in node:
                return False
            node = node[c]
        return True

    def startswith(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node:
                return False
            node = node[c]
        return True


# LeetCode648 单词替换
# TrieNode写法
class Trie1(Trie):

    def __init__(self):
        super().__init__()
        self.root = TrieNode()
        self.isEnd = False

    def insert(self, word):
        node = self.root
        for c in word:
            node = node.children[c]
        node.isEnd = True

    def shortestPrefix(self, prefix):
        node = self.root
        for i in range(len(prefix)):
            if node is None:
                return ""
            if node.isEnd:
                return prefix[i]
            node = node.children[prefix[i]]
        if node and node.isEnd:
            return prefix
        return ""

class Solution:

    def replaceWords(self, dictionary: list[str], sentence: str) -> str:
        trie = Trie1()
        for d in dictionary:
            trie.insert(d)
        words = sentence.split(' ')
        for i in range(len(words)):
            prefix = trie.shortestPrefix(words[i])
            if prefix:
                words[i] = prefix
        return ' '.join(words)

