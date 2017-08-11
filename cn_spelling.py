import kenlm
import codecs
import os
import re
import time
import pypinyin
import pickle
import wubi
import numpy as np
from pypinyin import pinyin, lazy_pinyin
from collections import Counter
from langconv import *

print('Loading models...')

model_path = './kenmodels/zhwiki.klm'
model = kenlm.Model(model_path)

text_path = './data/wikipedia/cn_wiki.txt'
counter_path = './data/wikipedia/cn_wiki_counter.pkl'
if os.path.exists(counter_path):
    print('Loading Counter from file: {}'.format(counter_path))
    counter = pickle.load(open(counter_path, 'rb'))
else:
    print('Generating Counter from text file: {}'.format(text_path))
    counter = Counter((codecs.open(text_path, 'r', encoding='utf-8').read()))
    pickle.dump(counter, open(counter_path, 'wb'))
    print('Dumped Counter to {}'.format(counter_path))

total = sum(counter.values())

xjz_dict_path = './data/xjz.pkl'
if os.path.exists(xjz_dict_path):
    xingjinzi = pickle.load(open(xjz_dict_path, 'rb'))
else:
    xingjinzi = {}
    pickle.dump(xingjinzi, open(xjz_dict_path, 'wb'))

common_dict_path = './data/common.pkl'
if os.path.exists(common_dict_path):
    common = pickle.load(open(common_dict_path, 'rb'))
else:
    common = {}
    pickle.dump(common, open(common_dict_path, 'wb'))
common_mistakes = common.keys()

print('Models loaded.')

# 输入汉字，查询同音字
def samechr(in_char):
    n = 0
    char_list = []
    for i in range(0x4e00, 0x9fa6):
        if(pinyin([chr(i)], style=pypinyin.NORMAL)[0][0] == pinyin([in_char], style=pypinyin.NORMAL)[0][0]):
            char_list.append(chr(i))
    return char_list

# 输入拼音，查询同音字                
def sametone(in_pinyin):
    n = 0
    char_list = []
    for i in range(0x4e00, 0x9fa6):
        if(pinyin([chr(i)], style=pypinyin.TONE2)[0][0] == in_pinyin):
            # TONE2: zho1ng
            char_list.append(chr(i))
    return char_list

def simtone(in_pinyin):
    pass

# 五笔形近字
def simshape(in_char, frac=5):
    in_wubi = wubi.get(in_char, 'cw') # Get wubi code
    edit_wubi = edits1(in_wubi)
    valid_char_list = list(wubi_known(edit_wubi))
    return sorted(valid_char_list, key=lambda k: getf(k), reverse=True)[:len(valid_char_list)//frac+1]
    # return list(wubi_known(edit_wubi))

# Get the frequency of a single character in the text
def getf(char):
    return counter[char] / total
    
def wubi_known(words): 
    # The subset of `words` that appear in the dictionary.
    return set(wubi.get(w, 'wc') for w in words if w is not wubi.get(w, 'wc'))

def edits1(word):
    # All edits that are one edit away from `word`.#
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    # transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    results = [e for e in set(deletes + inserts + replaces) if len(e) > 0 and len(e) < 6]
    return results

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def add_common_mistake(mistake_word, correct_word):
    common[mistake_word] = correct_word
    common_mistakes.append(mistake_word)
    pickle.dump(common, open(common_dict_path, 'wb'))

def add_xingjinzi(line, delimeter=','):
    chars = line.split(delimeter)
    for i, char in enumerate(chars):
        xingjinzi[char] = chars[:i] + chars[i+1:]
    pickle.dump(xingjinzi, open(xjz_dict_path, 'wb'))

def get_xingjinzi(in_char):
    return xingjinzi.get(in_char, [])

def gen_chars(in_char, frac=10):
    # char_list = samechr(in_char) + simshape(in_char, frac=10)
    char_list = samechr(in_char)
    if not char_list:
        char_list = [in_char]
    return sorted(char_list, key=lambda k: getf(k), reverse=True)[:len(char_list)//frac+1] + get_xingjinzi(in_char)

def get_score(s):
    return model.score(' '.join(s), bos=False, eos=False)

def merge_ranges(ranges):
    saved = ranges[0][:]
    results = []
    for st, en in ranges:
        if st <= saved[1]:
            saved[1] = max(saved[1], en)
        else:
            results.append(saved[:])
            saved[0] = st
            saved[1] = en
    results.append(saved[:])
    return results

def score_sentence(ss, k):
    ngrams = []
    scores = []
    mini = 0
    minscore = 0
    for i in range(len(ss) - k + 1):
        ngram = ss[i:i+k]
        ngrams.append(ngram)
        score = get_score(ngram)
        if score < minscore:
            minscore = score
            mini = i
        scores.append(score)
    mad_based_outlier(np.array(scores), threshold=2.8)
    outindices,_ = percentile_based_outlier(np.array(scores), threshold=93)
    outranges = merge_ranges([[outindex, outindex+k] for outindex in outindices])
    print('outranges are {}'.format(outranges))
    zipped = zip(ngrams, [round(score, 3) for score in scores])
    print(list(zipped))
    return mini, outranges, ss[mini:mini+k], zipped

def mad_based_outlier(points, threshold=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    outliers = points[modified_z_score > threshold]
    print('Mad based outlier scores are {}'.format(outliers))
    return outliers

def percentile_based_outlier(points, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(points, [diff, 100 - diff])
    outindices = np.where(points < minval) # returns a tuple 
    outliers = points[outindices]
    print('Percentile based outlier scores are {}'.format(outliers))
    return list(outindices[0]), outliers

def correct_common(ss):
    for mistake in common_mistakes:
        ss = re.sub(mistake, common[mistake], ss)
    return ss

def correct_ngram(ss, st, en):
    mingram = ss[st:en]
    candidates = {''}
    for g in mingram:
        gchars = gen_chars(g)
        print('Number of possible replacement for {} is {}'.format(g, len(gchars)))
        cand = candidates.copy()
        for c in cand:
            for gc in gchars:
                candidates.add(c + gc)
            candidates.remove(c)
    print('Number of candidate ngrams is {}'.format(len(candidates)))
    cgram = max(candidates, key=lambda k: get_score(ss[:st] + k + ss[en:]))
    return cgram

def correct(ss, k):
    ss = correct_common(ss)
    mini, outranges, mingram, _ = score_sentence(ss, k)
    for outrange in outranges:
        st, en = outrange
        print('Possible wrong gram is {}'.format(ss[st:en]))
        cgram = correct_ngram(ss, st, en)
        print('Corrected ngram is {}'.format(cgram))
        ss = ss[:st] + cgram + ss[en:]
    return ss

def main():
    line = '我们现今所使用的大部分舒学福号'
    print('The sentence is {}'.format(line))
    print('Corrected sentence is {}'.format(correct(line, 4)))
    print('-------------------------------------------------------------------------------------')
    DryInput_path = './data/sighan/clp14csc_release1.1/Dryrun/CLP14_CSC_DryRun_Input.txt'
    with open(DryInput_path, 'r') as f:
        lines = f.read()
    for line in lines.splitlines():
        sentence = Converter('zh-hans').convert(line.split()[1])
        print('The sentence is {}'.format(sentence))
        print('Corrected sentence is {}'.format(correct(sentence, 5)))
        print('-------------------------------------------------------------------------------------')

if __name__=='__main__':
    main()
