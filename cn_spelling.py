import kenlm
import codecs
import os
import re
import time
import pypinyin
import jieba
import pickle
import math
import wubi
import numpy as np
from pypinyin import pinyin, lazy_pinyin
from collections import Counter
from langconv import *

print('Loading models...')

jieba.initialize()

bimodel_path = './kenmodels/zhwiki_bigram.klm'
bimodel = kenlm.Model(bimodel_path)
print('Loaded bigram language model from {}'.format(bimodel_path))

trimodel_path = './kenmodels/zhwiki_trigram.klm'
trimodel = kenlm.Model(trimodel_path)
print('Loaded trigram language model from {}'.format(trimodel_path))

_4model_path = './kenmodels/zhwiki_4gram.klm'
_4model = kenlm.Model(_4model_path)
print('Loaded 4-gram language model from {}'.format(_4model_path))

wordmodel_path = './kenmodels/zhwiki_word_bigram.klm'
wordmodel = kenlm.Model(wordmodel_path)
print('Loaded word-level bigram language model from {}'.format(wordmodel_path))

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

sims_dict_path = './data/sims.pickle' # Path of similar shape characters dict
if os.path.exists(sims_dict_path):
    sims = pickle.load(open(sims_dict_path, 'rb'))
    print('Loaded similar shape dict from file: {}'.format(sims_dict_path))
else:
    sims = {}
    pickle.dump(sims, open(sims_dict_path, 'wb'))

simp_dict_path = './data/simp_simplified.pickle' # Path of similar pronunciation characters dict
if os.path.exists(simp_dict_path):
    simp = pickle.load(open(simp_dict_path, 'rb'))
    print('Loaded similar pronunciation dict from file: {}'.format(simp_dict_path))
else:
    simp = {}
    pickle.dump(simp, open(simp_dict_path, 'wb'))

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
def simshape(in_char, frac=1):
    in_wubi = wubi.get(in_char, 'cw') # Get wubi code
    edit_wubi = edits1(in_wubi)
    simshape_list = list(wubi_known(edit_wubi))
    return sorted(simshape_list, key=lambda k: getf(k), reverse=True)[:len(simshape_list)//frac]
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
    return set(xingjinzi.get(in_char, []))

def get_simshape(in_char):
    return sims.get(in_char, set())

def get_simpronunciation(in_char):
    return simp.get(in_char, set())

def gen_chars(in_char, frac=2):
    # chars_set = get_simshape(in_char).union(get_simpronunciation(in_char))
    # chars_set = get_simpronunciation(in_char)
    chars_set = get_simpronunciation(in_char).union(get_xingjinzi(in_char))
    if not chars_set:
        chars_set = {in_char}
    chars_set.add(in_char)
    chars_list = list(chars_set)
    return sorted(chars_list, key=lambda k: getf(k), reverse=True)[:len(chars_list)//frac+1]

def get_score(s, model=bimodel):
    return model.score(' '.join(s), bos=False, eos=False)

def get_wordmodel_score(ss):
    return wordmodel.score(' '.join(jieba.lcut(ss)), bos=False, eos=False)

def get_model(k):
    return {
        2: bimodel,
        3: trimodel,
        4: _4model,
    }.get(k, bimodel) # Return the bigram model as default

def overlap(l1, l2): # Detect whether two intervals l1 and l2 overlap
    if l1[0] < l2[0]:
        if l1[1] <= l2[0]:
            return False
        else:
            return True
    elif l1[0] == l2[0]:
        return True
    else:
        if l1[0] >= l2[1]:
            return False
        else:
            return True

def get_ranges(outranges, segranges):
    overlap_ranges = set()
    for segrange in segranges:
        for outrange in outranges:
            if overlap(outrange, segrange):
                overlap_ranges.add(tuple(segrange))
    return [list(overlap_range) for overlap_range in overlap_ranges]

def merge_ranges(ranges):
    print('Length of ranges is {}'.format(len(ranges)))
    ranges.sort()
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

def score_sentence(ss):
    hngrams = []
    hscores = [] # hierachical scores
    houtranges = []
    havg_scores = []
    for k in [2, 3, 4]:
        ngrams = []
        scores = []
        for i in range(len(ss) - k + 1):
            ngram = ss[i:i+k]
            ngrams.append(ngram)
            score = get_score(ngram, model=get_model(k))
            # for _ in range(k):
            scores.append(score)
        percentile_based_outlier(np.array(list(scores)), threshold=93)
        outindices, _ = mad_based_outlier(np.array(list(scores)), threshold=1.2)
        if outindices:
            outranges = merge_ranges([[outindex, outindex+k] for outindex in outindices])
            print('outranges are {}'.format(outranges))
        else:
            outranges = []
            print('No outranges.')
        hngrams.append(ngrams)
        houtranges.append(outranges)
        zipped = zip(ngrams, [round(score, 3) for score in scores])
        print(list(zipped))
        hscores.append(scores)
        for _ in range(k-1):
            scores.insert(0, scores[0])
            scores.append(scores[-1])
        avg_scores = [sum(scores[i:i+k]) / len(scores[i:i+k]) for i in range(len(ss))]
        havg_scores.append(avg_scores)
    per_word_scores = list(np.average(np.array(havg_scores), axis=0))
    outindices, _ = mad_based_outlier(np.array(list(per_word_scores)), threshold=1.2)
    if outindices:
        outranges = merge_ranges([[outindex, outindex+k] for outindex in outindices])
        print('outranges are {}'.format(outranges))
    else:
        outranges = []
        print('No outranges.')
    return per_word_scores, houtranges, hscores, outranges

def mad_based_outlier(points, threshold=1.2):
    points = np.array(points)
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0) # get the median of all points
    diff = np.sqrt(np.sum((points - median)**2, axis=-1)) # deviation from the median
    med_abs_deviation = np.median(diff) # median absolute deviation (MAD)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    points = points.flatten()
    outindices = np.where((modified_z_score > threshold) & (points < median))
    outliers = points[outindices]
    print('Mad based outlier scores are {}'.format(outliers))
    return list(outindices[0]), outliers

def percentile_based_outlier(points, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(points, [diff, 100 - diff])
    outindices = np.where(points < minval) # returns a tuple 
    outliers = points[outindices]
    print('Percentile based outlier scores are {}'.format(outliers))
    return list(outindices[0]), outliers

def detect_final_particle(ss):
    # Sentence final-particle detection
    last_char = ss[-2]
    # 马 码 -> 吗      把 巴 -> 吧      阿 -> 啊
    if last_char == '马' or last_char == '码':
        ss = ss[:-2] + '吗' + ss[-1]
    elif last_char == '把' or last_char == '巴':
        ss = ss[:-2] + '吧' + ss[-1]
    elif last_char == '阿':
        ss = ss[:-2] + '啊' + ss[-1]
    return ss

def preprocess(ss):
    rs = ''
    for s in ss:
        code = ord(s)
        if code == 12288:
            code = 32
        elif code == 8216 or code == 8217:
            code = 39
        elif (code >= 65281 and code <= 65374):
            code -= 65248
        rs += chr(code)
    punc = ['「', '」'] # punctuations to remove from the sentence
    rs = re.sub('|'.join(punc), '', rs)
    rs = detect_final_particle(rs)
    return rs

def correct_common(ss):
    for mistake in common_mistakes:
        ss = re.sub(mistake, common[mistake], ss)
    return ss

def correct_ngram(ss, st, en):
    mingram = ss[st:en]
    candidates = {''}
    for g in mingram:
        gchars = gen_chars(g)
        print('Number of possible replacements for {} is {}'.format(g, len(gchars)))
        cand = candidates.copy()
        for c in cand:
            for gc in gchars:
                candidates.add(c + gc)
            candidates.remove(c)
    print('Number of candidate ngrams is {}'.format(len(candidates)))
    cgram = max(candidates, key=lambda k: get_score(ss[:st] + k + ss[en:]) + get_score(k) + math.log(135)**(k == mingram)) # get_score(ss[:st] + k + ss[en:])
    return cgram

def correct_ngram_2(ss, st, en):
    mingram = ss[st:en]
    for i, m in enumerate(mingram):
        mc = gen_chars(m) # Possible corrections for character m in mingram
        print('Number of possible replacements for {} is {}'.format(m, len(mc)))
        mg = max(mc, key=lambda k: get_score(ss[:st] + mingram[:i] + k + mingram[i:] + ss[en:]) + math.log(12)**(k == m))
        mingram = mingram[:i] + mg + mingram[i+1:]
    return mingram

def correct(ss, k):
    ss = preprocess(ss)
    ss = correct_common(ss)
    tokens = list(jieba.tokenize(ss)) # Returns list of tuples (word, st, en)  mode='search'?
    print('Segmented sentence is {}'.format(''.join([str(token) for token in tokens])))
    segranges = [[token[1], token[2]] for token in tokens]
    # outranges, hscores = score_sentence(ss, k)
    per_word_scores, houtranges, hscores, outranges = score_sentence(ss)
    if outranges:
        correct_ranges = get_ranges(outranges, segranges)
        for correct_range in correct_ranges:
            st, en = correct_range
            print('Possible wrong ngram is {}'.format(ss[st:en]))
            cgram = correct_ngram_2(ss, st, en)
            print('Corrected ngram is {}'.format(cgram))
            ss = ss[:st] + cgram + ss[en:]
    else:
        print('No ngram to correct.')
    return ss

def main():
    line = '我们现今所使用的大部分舒学福号'
    print('The sentence is {}'.format(line))
    print('Corrected sentence is {}'.format(correct(line, 3)))
    print('-------------------------------------------------------------------------------------')
    DryInput_path = './data/sighan/clp14csc_release1.1/Dryrun/CLP14_CSC_DryRun_Input.txt'
    with open(DryInput_path, 'r') as f:
        lines = f.read()
    for line in lines.splitlines():
        sentence = Converter('zh-hans').convert(line.split()[1])
        print('The sentence is {}'.format(sentence))
        print('Corrected sentence is {}'.format(correct(sentence, 3)))
        print('-------------------------------------------------------------------------------------')

if __name__=='__main__':
    main()
