# 注意力机制小记

人的注意力一般分为两种：

- saliency-based attention 显著性注意力

- focus attention 聚焦式注意力

## 显著性注意力

**显著性注意力**， 自下而上的无意识的注意力，对信息的处理是被动的。这个被动体现在注意力获取信息上是无意识的、受外界刺激而驱动的、没有目的性的。

像 CNN 中的最大汇聚（Max Pooling）和门控机制（Gating），本质上是降采样过操作。它们的目的都是筛选最显著的信息（某个度量值最大），因此这类模型可以看做是显著性注意力的运用。

## 聚集式注意力

**聚集式注意力**，自上而下的有意识的注意力，对信息的处理是主动的、有选择的，因此也可以称为选择性注意力。这个主动性体现在注意力获取信息上是有目的性的、有意识的、被任务驱动的。例如我们要寻找视野中的苹果，聚焦式注意力能够帮助我们快速聚焦在苹果和苹果有关的特征上，进而快速找到苹果。再比如人类在做阅读理解时，一般都是带着问题去阅读去寻找答案，这个“带着问题阅读”的过程就体现了聚焦式注意力的主动性。

聚焦式注意力和显著性注意力的最大差别是获取信息的主动性上，前者获取信息是被动的，后者则是主动的。



## Attention机制的基本思路

假设一个向量序列
$$
X = [X_1, X_2 ....X_n]\in R^{n\times d}
$$
。我们有一个和任务相关的向量**q**，可以根据任务生成，也可以是可学习的参数。那么注意力机制所要做的事情可以分为三步：

- 查询向量 **q** 与每个 x_i 计算相关性 αi，相关性通过评分函数（也称为相关性函数）获得
- 使用softmax归一化相关性αi ，称为注意力分布
- 根据注意力分布计算向量序列的均值

假设我们有评分函数s用于计算查询向量**q**与每个Xi的相关性，那么有：
$$
α_i = s(q, x_i)
$$
使用softmax函数进行归一化相关性，获得注意力分布，
$$
p(z=i|X, q)=softmax(α_1...α_n)\\
           =\frac{exp(α_i)}{\sum_{i=1}^{n}(α_i)}\\
           =\frac{exp(s(q, x_i))}{\sum_{i=1}^{n}exp(s(s,x_i))}
$$
加权平均，即注意力分布下的均值:
$$
Attention(X, q) = \sum_{i=1}^{n}p(z=i|X, q)x_i
$$

| 注意力机制 | 阅读理解任务                     |
| ---------- | -------------------------------- |
| 向量序列   | 阅读材料                         |
| 查询向量   | 问题                             |
| 注意力分布 | 阅读材料中与答案有关的材料部分   |
| 加权平均   | 根据阅读材料整合与答案相关的内容 |



## 查询向量

 在注意力中，所谓的主动与被动的关键是，我们是否提前知道自己需要什么信息。知道自己需要什么信息体现在查询向量$q$上。经过不断的折腾，我们找到一个”标的”向量，如果一个向量和它越”相似”，那么这个向量对任务的贡献就更大。即，如果$x_i$和某个标的向量 $q$越相似，它的重要性越高。又要重新回去折腾 $s$ 这个函数。即用 $s(p,x_i)$ 来表示 $x_i$ 的重要性。于是，每个 $X_i$ 的权重 $λ_i$ 计算如下:
$$
λ_i = \frac{e^{s(q,x_i)}}{\sum_{i=1}^{n}s(q,x_i)}
$$
那么获得的注意力机制为：
$$
C=\sum_{i=1}^{n}λ_ix_i
$$
直观图表示为：

![additive-attention-query](D:\blog_file\imgs\additive-attention-query.png)

这就是简单的有查询向量参与的注意力机制





## 简述自注意力机制(Self-Attention)

自注意力机制可以解决序列中$K,V,Q$数据结构的问题，但是其作用远不止这些

前面写道，Attention是注意力分布$λ_i=p(z=i|X, q)$下的期望：
$$
Attention(X,q) = \sum_{i=1}^{n}p(z=i|X, q)x_i
$$
对于键值对类型的数据结构可以改为：
$$
Attention(X,q)=\sum_{i=1}^{n}p(z=i|K,q)v_i
$$
Google用了简单直接方法得到$K,V,Q$，即将初始向量乘上一个矩阵进行线性变换
$$
Q = W^qX\\
K = W^kX\\
V = W^vX
$$
由于$Q,V,K$都是来自于初始序列$X$，故称为自注意力机制。

### 多头注意力机制(Muti-head Attention)

简单来说就是把得到的$K,V,Q$进行简单的分裂为n个就是n头注意力机制了。公认多头注意力机制的效果是好于单头的，因为前者可以捕获更多维度的信息，示意图如下：

![muti-head-attention](D:\blog_file\imgs\muti-head-attention.jpeg)