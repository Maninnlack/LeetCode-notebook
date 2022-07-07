dictionary = ["cat","bat","rat"]
sentence = "the cattle was rattled by the battery"
words = sentence.split(' ')
for i in range(len(words)):
    for root in dictionary:
        if root in words[i] and len(root) < len(words[i]):
            words[i] = root
print(' '.join(words))