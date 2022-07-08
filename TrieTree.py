"""
字典树相关题目：
LeetCode208 实现前缀树
LeetCode648 单词替换
LeetCode211 添加与搜索单词
LeetCode677 键值映射
"""


import collections
from collections import deque

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


# 字典写法
class TrieDic1:

    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['#'] ={}

    def find_sp(self, word):
        node = self.root
        for i in range(len(word)):
            if '#' in node:
                return word[:i]
            if word[i] in node:
                node = node[word[i]]
            else:
                break
        return word

class Solution:

    def replaceWords(self, dictionary: list[str], sentence: str) -> str:
        trie = TrieDic1()
        for dic in dictionary:
            trie.insert(dic)
        words = sentence.split(' ')
        return ' '.join(trie.find_sp(word) for word in words)


# LeetCode 211 添加与搜索单词
# TrieNode 方法
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()


    def addWord(self, word: str) -> None:
        node = self.root
        for c in word:
            node = node.children[c]
        node.isEnd = True


    def search(self, word: str) -> bool:
        return self.match(word, 0, self.root)

    def match(self, word, idx, root):
        if not root:
            return False
        if idx == len(word):
            return root.isEnd
        if word[idx] != '.':
            return self.match(word, idx + 1, root.children.get[word[idx]])
        else:
            for child in root.children.values():
                if self.match(word, idx + 1, child.child):
                    return True
        return False


# 字典方法
class WordDictionary:

    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        node = self.root
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['#'] = {}

    def search(self, word: str) -> bool:
        word += '#'
        bfs = deque([(0, self.root)])
        while bfs:
            idx, cur = bfs.popleft()
            if idx == len(word):
                return True
            if word[idx] == '.':
                for nxt in cur.values():
                    bfs.append((idx + 1, nxt))
            elif word[idx] in cur:
                bfs.append((idx + 1, cur[word[idx]]))
        return False


# 677 键值映射
# 字典方法
class MapSum:

    def __init__(self):
        self.root = {}

    def insert(self, key: str, val: int) -> None:
        node = self.root
        for c in key:
            if c not in node:
                node[c] = {}
            node = node[c]
        node['val'] = val


    def sum(self, prefix: str) -> int:
        node = self.root
        for c in prefix:
            if c not in node:
                return 0
            else:
                node = node[c]
        ans = 0
        def dfs(node):
            for c in node:
                if c == 'val':
                    nonlocal ans
                    ans += node[c]
                else:
                    dfs(node[c])
        dfs(node)
        return ans
