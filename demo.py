from functools import lru_cache

class Trie:
    def __init__(self):
        self.children = {}

    def add(self, word, index):
        cur = self.children
        for c in word:
            if c not in cur:
                cur[c] = {}
            cur = cur[c]
        if 'val' not in cur:
            cur['val'] = []
        cur['val'].append(index)

    def search(self, word):
        cur = self.children
        for c in word:
            if c not in cur:
                return []
            cur = cur[c]
        ans = []
        if 'val' in cur:
            ans.extend(cur['val'])

        def dfs(node):
            for c in node:
                if c == 'val':
                    nonlocal ans
                    ans.extend(node['val'])
                else:
                    dfs(node[c])

        dfs(cur)
        return ans


class WordFilter:

    def __init__(self, words):
        self.prefTrie = Trie()
        self.suffTrie = Trie()
        for i, word in enumerate(words):
            self.prefTrie.add(word, i)
            self.suffTrie.add(word[::-1], i)
        self.memo = {}

    @lru_cache(None)
    def f(self, pref: str, suff: str) -> int:
        a = self.prefTrie.search(pref)
        b = self.suffTrie.search(suff[::-1])
        if a and b:
            return max(set(a) & set(b))
        return -1

# Your WordFilter object will be instantiated and called as such:
# obj = WordFilter(words)
# param_1 = obj.f(pref,suff)


a = WordFilter(['abbba', 'abba'])
print(a.f('ab', 'ba'))
