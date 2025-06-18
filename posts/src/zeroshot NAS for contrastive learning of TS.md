# zeroshot NAS for contrastive learning of TS

## 文献调研

- **Automated Contrastive Learning Strategy Search for Time Series**：

  这个工作是NAS(Neural  Architecture Search)在时序对比学习中的应用，特别是它提出了对数据增强方法的搜索

  缺陷：它没有对 encoder 的模块 比如 CNN lstm spatial selfattention temporal self attention 进行搜索，且耗时长，搜一次十个小时

- **AutoCTS++: Zero-shot Joint Neural Architecture and Hyperparameter Search for Correlated Time Series Forecasting**:

  这个工作有考虑不同时间空间模块，但是不是对比学习的 是做预测的，而且他是 zero shot search，耗时很小

## 我们的工作

将这两个工作进行结合做**zeroshot NAS for contrastive learning of TS**将AutoCTS++改成对比学习方法，使用AutoCTS++中的zero shot NAS方法



## Neural Architecture Search

传统的NAS主要含有三个维度：搜索空间，搜索策略和性能评估方法，三者关系如图：

![取自论文“Neural Architecture Search: Insight from 1000 papers"”](D:\blog_file\imgs\微信图片_20250419153404.png)

搜索策略在设定的搜索空间$\mathcal{A}$中迭代选择架构（通常使用架构编码方法Architecture encoding method）,架构随后被传递至性能估计策略中，并将性能评估返回给搜索策略。对于one-shot方法来说，搜索策略和性能评估策略是耦合的。

- **Search Space**是允许NAS算法选择的所有架构的集合，一般的NAS的搜索空间在几千到$10^{20}$不等，虽然搜索空间原则上十分通用，但也可以使用相关领域方法简化搜索。但是使用过多方法可能加入认为偏见。从而降低NAS找到新颖架构的可能性。
- **Search Strategy**是一种优化技术，用于在搜索空间寻找到合适的高性能架构
- **Performance Estimation Strategy**是用于快速预测神经架构性能以避免完全训练架构的任何方法。使用学习曲线外推等性能估计策略可以大大提高搜索速度。

### Search Space

在NAS中最重要的架构应该就是搜索空间了，虽然AutoML的一些设计和NAS有重合，但是NAS的搜索空间设计是十分独特新颖的。

搜索空间的设计其实就是人为偏见和搜索效率的权衡：如果搜索空间设计的很小且都是精选设计的模块，那么NAS将快速地找到高性能的模型；如果搜索空间设计的很大且包含许多原始构筑块，那么NAS更有可能找到真正新颖的架构。

### Search Space中的术语

- **Operation/primitive**这代表了在搜索空间中的最小单元。对于几乎所有的搜索空间，这是一个由固定的激活函数，operation和固定的归一化操作所构成的三元组。像是ReLU-conv_1x1-batchnorm，其中ReLU和batchnorm是固定的，其中的operation是在不同的多个operations中的一个选择。
- **Layer**通常用于chain-structured or macro search space，用来指代作为operation的相同的事物。但是又是也指代一些operations的组合
- **Block/Module**有时用于表示遵循大多数chain-structured and macro 搜索空间中的序列化的堆栈
- **Cell**用于表示基于单元格的搜索空间中作的有向无环图。单元格中的最大作数通常是确定的。

### Architecture Encoding

在大多数的搜索空间中，其架构可以被完全由有向无环图(DAG)表示，其中每个节点或者边都可以被表示为operation。像是cell-based search space。但是对于一些层次搜索空间，其架构不能完全使用DAG表示，并且需要使用一种条件结构编码(conditionally-structured encoding)。





## AutoCTS++阅读

**AutoCTS++**是一个采用了zero-shot的NAS架构去解决目前的**Automated CTS**方法只能寻找到对某一数据集预设定超参数的最优模型，这种限制不符合工业界面临的复杂度情况，为了解决这个问题，**AutoCTS**提出了：

- 提出了一种新颖的搜索空间：**architecture-hyperparameter joint search space**，通过将预选的架构和其超参数编码进一个图展示( graph representation)，看图上其实是一个特殊处理的双重DAG。
- 提出了一种零样本学习(**zero-shot**)的搜索策略：**Task-aware Architecture-Hyperparameter Comparator (T-AHC)**，这种架构可以在未见过的数据集中为Architecture-Hyperparameter进行排名。

<img src="D:\blog_file\imgs\AutoCTS_architecture.png" alt="AutoCTS Architecture" style="zoom: 200%;" />

### Problem definition

任务的目标是从一个预定义的联合architecture-hyperparameter搜索空间$S$中为任务$\mathcal{T}$自动构建一个最优的**ST-Block**$\mathcal{F}^*$ ,以在验证集$\mathcal{D}_{val}$中取得最小的预测误差。
$$
\mathcal{F}^*=argmin_{\mathcal{F}^* \in S}ErrorMetric_{\mathcal{T}}(\mathcal{F}, \mathcal{D}_{val})
$$
通过T-AHC选择出来的最优ST Block模块通过一系列的拓扑序列化堆叠在一起，以获取数据中的时空特征，如图：

<img src="D:\blog_file\imgs\CTS_forecastingpng.png" alt="CTS_forecatsing" style="zoom: 150%;" />

### Zero-shot joint search

#### Joint search space

这个搜索空间关注时空模块组件，从两个角度考量ST Block：(1).架构角度，包括其中的操作原件(也就是CNN，GCN，Transformer这些模块化的组件)和操作原件之间的组合。他们详细的介绍了这个组合搜索空间中的架构和超参数的细节：

- **Architecture search space**：**AutoCTS**将ST Block中的**S/T operator**从两个角度考量，时间组件和空间组件。

  - 在时间组件上通过研究手动设计的 CTS 预测模型和现有自动 CTS 预测框架的搜索空间，他们确定并包括两个引人注目的候选 T  operator。   
  - 在空间组件上，同理，他们也是选择了两个S operator以提取空间特征
  - 他们也加入了一个**identity operator**以支持节点之间的跳跃连接。

  将这五个operator组合成一个operator集合$O$。作者认为这样的架构能够很容易地加入其他的operator。具体上是：当需要加入新的operator时，先将这个新的operator加入到$O$中，再对这个包含了新operator的集合进行采样，然后重新训练T-AHC。之前收集到的原先的集合样本是可以被重新运用的，这是很高效的设计。
  
- **Hyperparameter search space**： 作者将超参数从两个角度考量：结构超参数和训练超参数。同Architecture search space一样，作者认为这种架构可以有效地兼容额外的超参数。

  - 结构超参数(**Structural hyperparameters**) :结构超参数就是指代ST Block中的结构参数，如$B$(number of ST-blocks), $C$(number of nodes in an ST-block), $U$指代ST-block中的哪个节点产生输出.....,超参数在图中显示：

    <img src="D:\blog_file\imgs\hyperparameters.png" alt="hyperparameters in search space" style="zoom:150%;" />

  - 训练超参数(**Training hyperparameters**):包含了是否使用dropout的$δ$等等。

- **Encoding of the joint search space**：由于两个搜索空间采用了不同的编码形式(DAG和向量形式)，采用简单的结合形式是不可行的。作者采用了联合双DAG的方式将这两个搜索空间结合。搜索空间的图像表示为：

  <img src="D:\blog_file\imgs\Search_space_architectuer.png" alt="architecture of search sapce" style="zoom:150%;" />

### Task-aware architecture-hyperparameter comparator

作者认为，不需要计算所有待选模块的准确度，他们设计了一个比较器，用于在两个模块之间进行比较，这样可以高效地找到精确度最高的模块。作者设计的T-AHC以两个arch-hyper $ah1$和$ah2$作为输入，并输入二进制数$y$，$y$用于表示哪个arch-hyper可能具有更高的验证精度。

- 文章提出了Task embedding  learning module，这个模块可以捕获到不同任务中的相似性，它通过学习到任务中的隐藏表示，并鼓励相似的任务都有着相似的隐藏表示。（这也是实现了zero-shot的重要组件了,毕竟在AHC这个模块中AutoCTS++就多了这个模块）

在每一个任务中都为其找到合适的$(ah, R(ah))$对需要大量的GPU计算，这在现实工程上是不现实的。于是作者考虑了将任务进行相似度对比，具体上说是，每个ah模块对于相似的任务其实精确度上的表现其实会是相近的。

为满足这项需求，作者采用了一些新颖的嵌入技术，这些嵌入需要满足：(1).能有效地将预测设置(输入长度$P$和输出长度$Q$)与CTS的datasets的$D$作为统一嵌入。(2).能够学习预测任务和模型性能排名之间的关系。

- 为满足第一个需求，作者设计了一个滑动窗口$S=P+Q$用来分割数据集，在每个时间序列窗口中$\mathcal{D}_i=\mathcal{R}^{N\times S\times F}$,将此嵌入进arch-hyper的模块中。作者还使用了TS2Vec的对比学习方法来提供时间序列上的通用嵌入，具体上说是给定一个数据窗口$\mathcal{D}_i$，TS2Vec将特征编码为一个$F^`$维的隐藏空间，公式表示为：
  $$
  E_i=TS2Vec(\mathcal{D}_i) where E_i \in R^{N\times S\times F^`}
  $$
  
- 为了满足第二个需求，作者认为两个相近rank分数的arch-hyper应该也具有相同的嵌入。于是他们采用了一个双堆叠的Set-Transformer 来将TS2Vec返回的嵌入向量$E_i$进一步编码。文中将这个two Set-Transformer layers称为IntraSetPool和InterSetPool。

T-AHC的结构框架如下：

<img src="D:\blog_file\imgs\T-AHC.png" alt="T-AHC" style="zoom:150%;" />

### Pre-training a T-AHC

预训练T-AHC需要大量的$(t_i, ah_1, ah_2, y)$这样格式的输入。对于特定任务$\mathcal{T_i}$，需要训练和评估该任务中的$ah_1,ah_2$。需要为考虑足够的训练样本，以充分预训练T-AHC。这带来了两个挑战：(1).尽管存在多个相关时间序列数据集，但是仍然可能缺失足够数量的相关时间序列。(2).对于每个任务，获取样本$(t_i, ah_1, ah_2, y)$需要大量的时间，因为需要训练和评估$ah_1, ah_2$。

- 为解决第一个挑战：作者建议将一般的CTS数据集分割为多个CTS预测任务，以此为T-AHC提供足够的训练样本。
- 为解决第二个挑战：第一个做法是采用了the early-validation metric $R(.)$，这个R(.)的作用就是生成一个输出$y$的一个伪标签。这么做可行的原因是由于需要的是两个arch-hyper之间的比较，故不需要特别准确的值。但是这样造成了训练样本,第二个做法是选择共享样本。

## Search strategy

这篇文章在Search strategy上的篇幅很短，毕竟这个框架的重点在于这个预训练的T-AHC模块。

这里的Search strategy指的是在T-AHC预训练好后，部署在进行未见过的任务上所使用的算法。作者这里采用了启发式方法，也就是基于遗传算法的进化算法。



## 总结与思考

我认为作者主要的贡献在于提出了一种与训练模块T-AHC，既可以对未见过的任务进行操作，也极大的提高了运算速度（作者在T-AHC大篇幅地讲了如何削减运算用时）。还提出了一种新颖地搜索空间架构，将超参数和operator进行结合。

对于将这项任务改为对比学习任务，鉴于我现在还没完整看过几篇对比学习的文章，目前是没有头绪进行这个任务。







## Automated Contrastive Learning Strategy Search for Time Series阅读

这篇文章提出了AutoCL，这个模型使用了超大型搜索空间，包含了数据增强，嵌入转化，对比对结构以及对比损失。在搜索策略上，作者采用了强化学习的方法在验证集中进行优化。

### Solution Space

这项工作的搜索空间还是很有意思的，作者认为，CL(Contrastive Learining)的搜索空间必须满足两个条件：(1).包含基于先前的认为设计策略的CLS的关键维度。(2).每个维度的选择必须适中。

CL的本质是提取不受小数据扰动影响的语义保留嵌入，作者因此认为，数据增强是CLS(Contrastive Learning Strategy)的基石。

总的来说，作者考虑了四个维度：数据增强，嵌入转换，对比对构建和对比损失。摘要显示如图：

<img src="D:\blog_file\imgs\solution_space_of_AutoCL.png" alt="solution space" style="zoom:150%;" />

- **Data Augmentations**. 数据增强阶段将输入转化为不同但是相关联的视图。作者这里采用了6个常用的数据增强方法作为子维度，包括**resizing (length)， rescaling  (amplitude)， jittering， point masking， frequency  masking，random cropping **。在数据增强阶段，应用数据增强的顺序也会影响学习到的嵌入。所以，作者设计了5中不同的应用数据增强顺序。于是一共有$5\times 11^6$个数据增强的选项。
- **Embedding Transformation**. 这个阶段包含两个模块，嵌入增强(**Embedding augmentation**)和归一化处理(**Normalization**)。在嵌入增强阶段，作者考虑了嵌入抖动(**embedding  jittering**)和嵌入掩码(**embedding masking**)作为子维度。这个阶段和数据增强阶段是相同的。在归一化阶段，考虑三个归一化策略：无归一化，$l_2$norms，LayerNorm。
- **Contrastive Pair Construction**. 















## 二者工作的结合

我们的ZeroShot NAS for contrastive of TS要将这个两个工作进行结合。其难点在于AutoCL的搜索空间架构并没有使用encoder的模块(如：self-Attention，CNN， LSTM等)的搜索。并且需要将AutoCTS的T-AHC模块迁移使用到这个搜索空间中，需要更改搜索空间的架构(但是AutoCL没有开源......)



考虑在AutoCL的搜索空间上加入一些可用的encoder模块。并将该搜索空间改为契合与AutoCTS的搜索策略。



## 目前的工作

现在把基本的框架想清楚了，接下来就是实现和测试了：

- arch架构：我的arch使用了[index, arch, param]的形式，将参数放入arch中进行搜索。这样也许可以在AHC中让模型也学习到参数细节。我对每个Cell也有其特定的arch。在最后交由AHC学习时组成一个完整的Arch
- Cell架构：我将每个backbone分成了多个Cell，并在net中对Cell进行堆叠，我手动架构其拓扑规则，但是cell内部的拓扑规则仍是AutoCTS++中的做法。需要注意的是，我在net中并没有对Contrastive Loss Cell 进行使用，而是在训练阶段使用。

对比学习的总体架构:

![Contrastive Learning](D:\blog_file\imgs\contrastive.png)

我其中的Encoder组件没有进行搜索，采用的固定的Transformer Encoder。将输入的形状[B, C, N, T]转化为[B, N, D]。

在对比对构建时，其中使用的Encoder也是固定的，CNN，LSTM等，其对于输入的形状并没有改变。（由于AutoCTS中有对于Pooling策略的搜索，我这里打算也进行效仿。考虑是否也需要单独设立一个Cell）

### 5/6 需更改部分

这两天将整体架构大致搭建起来了，接下来是修改细节部分。

1. 由于之前没注意数据流的形状，今天需要将operation的数据输出形状进行更改：

   - Cell中的forward我总感觉有问题，但就不知道怎么改......

   - Contrastive Pair Construction部分，注意输入输出形状。
   - Contrastive Loss部分。（AutoCTS对于对比损失中的相似性函数都有进行选择，现在不考虑这个选择，真加的话不知道arch如何架构了）

2. 将net中那些arch进行更改，改成各个Cell都有其独特的arch。我还在思考怎么更改

上述的工作都搞定后，目前先不对AutoCTS中Search Solution的所有operation进行实现。先测试这个NAS跑起来如何（最重要的是能不能跑起来，这bug还是相当难改的）。再将T-AHC对我的arch和模型进行适配。



## 5/7

目前好像理解了一些，AutoCTS的Solution Space中的hierarchical pooling好像是指搜索在对比对构建中的池化操作，而不是在对比对构建之后进行池化操作。

目前看来是比较好实现的，就是这个arch该怎么弄？我是否需要为其单独设立一个Cell?

- 第一个，修改operations中的Contrastive Pair Construction部分。注意InstanceContrastive的全局对比，TemporalContrast的时间步对比，CrossScaleContrast的跨区域对比。这样的话其中的Encoder也需要更改，需要适配每个对比对构建operation的方法。
- 其中池化层建立Cell，在每个对比对构建operation中调用。



全错了，之前对于对比对构造的想法都是错的。

正确的是需要使用相似函数构造出相似矩阵，相似矩阵中会有对比对的损失。所以对比对构建和对比损失其实是一体的。那对于我的搜索就特别难受了。





```
[[0, 1], [0, 3], [1, 3], [1, 1], [2, 2], [2, 3], [3, 1]
```
