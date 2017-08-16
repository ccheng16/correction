import codecs
import mmap

fpath = "./data/wikipedia/cn_wiki.txt"
with codecs.open(fpath, 'r', encoding='utf-8') as f:
    text = f.readlines()

for line in text[:10]:
    print(' '.join(line.strip()), end=' ')
    # print(' '.join(line.strip().split(' / ')), end=' ')
