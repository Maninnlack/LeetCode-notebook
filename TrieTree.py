class TrieNode:
    """
    Trie 树节点的实现
    class TrieNode<V> {
        V val = null;
        TrieNode<V>[] children = new TrieNode[256];
    }
    """
    def __init__(self):
        self.val = None
        self.children = [None] * 256


class TrieMap:

    def __init__(self):
