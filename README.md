# Correction

## Overview
A project to correct spelling errors in Chinese texts
中文纠错任务

### 大致思路
* 使用语言模型计算句子或序列的合理性
* bigram, trigram, 4-gram 结合，并对每个字的分数求平均以平滑每个字的得分
* 根据Median Absolute Deviation算出outlier分数，并结合jieba分词结果确定需要修改的范围
* 根据形近字、音近字构成的混淆集合列出候选字，并对需要修改的范围逐字改正
* 句子中的错误会使分词结果更加细碎，结合替换字之后的分词结果确定需要改正的字
* 探测句末语气词，如有错误直接改正

### TODO
* 使用RNN语言模型推算每个字的合理概率（正反双向），以加强长距离前后文关系
* 构建更小更贴近现实的混淆集合（形近字和近音字）
* 从现实中收集更多的有语病或错别字的句子并标注

### 文件结构
```
data/
* sighan/: SIGHAN contests data
* bcmi_data/: 源自生活的语病数据集
* wikipedia/: 中文维基数据集 xml文件、纯文本、提取工具
* simp.pickle: similar pronunciation characters dictionary
* sims.pickle: similar shape characters dictionary
* simp_simplified.pickle: 过滤掉字频100一下的非常用字的版本
* xjz.pickle: 简明形近字dictionary
* 
```
```
kenlm/: library to generate statistical language models
kenmodels/: trained language models *.klm are binary files
```
```
nlm/: various neural language models
* tf_char_rnn: Character-level RNN language model implemented using TensorFlow
* cn_char_rnn: RNN language model for Chinese
* lstm_char_cnn: A CNN language model, not useful for this project
* char_rnn: Character-level RNN language model gists
```
```
spells/: useful tools for English spelling check
```
```
langconv.py: 简繁转换工具
zh_wiki.py: 简繁转换dictionary
```
```
tf_char_rnn/:
* checkpoints/: 17 is backward model, 10 is forward model
* logs/: summaries of training runs
* data/: text input to train models
* model.py: script describing the model
* train.py: script to train the model
* sample.py: script to sample texts and calculate per-char probabilities of a sequence
* utils.py: utilities for reading data and generating batches
```
```
results.out: 输出log
```
### 参考链接
* RNN语言模型: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
* 音型码: http://mabusyao.iteye.com/blog/2267661 
* Spellchecking by computer: http://www.dcs.bbk.ac.uk/~roger/spellchecking.html
* GNU Aspell: http://aspell.net/
* 搜狗实验室数据: http://www.sogou.com/labs/resource/list_pingce.php
* 基于seq2seq模型的中文纠错任务: http://media.people.com.cn/n1/2017/0112/c409703-29018801.html
* char-rnn-tensorflow: https://github.com/fujimotomh/char-rnn-tensorflow
* 语言模型KenLM的训练及使用: http://www.cnblogs.com/zidiancao/p/6067147.html
* KenLM: https://github.com/kpu/kenlm
* 基于语言模型的无监督分词: http://spaces.ac.cn/archives/3956/
* 中文句結構樹資料庫: http://turing.iis.sinica.edu.tw/treesearch/
* CKIP数据集: http://rocling.iis.sinica.edu.tw/CKIP/engversion/index.htm
* 达观数据搜索引擎的Query自动纠错技术和架构: http://www.datagrand.com/blog/search-query.html

