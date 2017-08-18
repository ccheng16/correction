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
data
* sighan: SIGHAN contests data
* bcmi_data: 源自生活的语病数据集
```

### 参考链接
