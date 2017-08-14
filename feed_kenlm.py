import codecs
import mmap

fpath = "./data/wikipedia/cn_wiki.txt"
with codecs.open(fpath, 'r', encoding='utf-8') as f:
    for i in range(10):
        text = f.readline()
        print(text)
        print(' '.join(text))
