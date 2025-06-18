# 空间架构

这只是帮助我进行架构思路的，多数是代码构建的想法。想了想好像这些文字在论文中用不上🫥

对比学习的总体架构图如下：

![contrastive](D:\blog_file\imgs\contrastive.png)

## 对比学习的基本概念：

-  $X$输入：形状为$X \in \mathcal{R}^{B\times C\times N\times T}$，经过不同的Data Augmentation，形成不同的视图
- $Encoder$组件：Encoder是在对比学习中最重要的部分，在我们的NAS架构中先不考虑Encoder的搜索(若时间允许可以加入Encoder的搜索)。数据$X_1, X_2$经过Encoder后变为$h_1, h_2 \in R^{B\times C\times T}$。这些数据作为输入进入到Contrastive Loss中，Loss在反向传播时优化Encoder参数。Encoder的输出就是对比学习的输出，可用于downstream tasks
- $Pair Construction \   and \  Contrastive Loss$: 在看过多数对比学习源码时，这其实是放在一起的(貌似也可以分开放置), 正负样本对的构建可以理解为一个构建相似矩阵的过程，这个过程需要使用到相似度矩阵，$PairConstruction$返回的是相似度矩阵和正负样本对索引。

- $Positive \ Negative \ Pair$： 同一个数据经过不同的数据增强就是一对正样本对，如$z_1[i]和z_2[i]$就是一对 正样本对，同一Batch中的其他样本就是当前索引下的负样本。

## 我的NAS如何适配对比学习

参考文章$AutoCLS$中加入了数据增强的方法的搜索，但是并没有对$Encoder$的搜索。

![Solution Space of AutoCLS](D:\blog_file\imgs\solution_space_of_AutoCL.png)

参考文章$AutoCTS$++中使用了$arch$数组来进行对Cell组件中DAG的构建。我们决定沿用此思想。

这里要提出我们工作中需要独立设计的部分：

### 模块化$Cell$

由于在我们的搜索空间中需要用到$Data\ Augmentation, Pair\ Contstruction$等组件。所以对于这些组件我们为其都设立一个独立的$Cell$。

### $Contrastive\ Pair\ Construction$的构建细节

$AutoCLS$在对比对构建时选取了三个方法：$Instance\ Contrast$, $Temporal\ Contrast$, $Cross-Scale\ Contrast$。在构建对比对时还加入了$\textbf{hierarchical pooling}$来进一步提取信息(是这个作用吗？),对比对构建和层级池化都有其需要选择的options。

#### 各个Contrast的功能

- $InstanceContrast$: 这个contrast组件对序列进行实例级别的对比，输入形式为$z_1, z_2 \in R^{B\times T\times C}$的数据经过构建后成为$z\in R^{2B\times T\times C}$的数据。这个数据返回到$ContrastLoss$中，形成形状为$R^{T\times 2B\times 2B}$的相似度矩阵。

- $TemporalContrast$:  这个Contrast组件对序列进行时间步级别的对比, 输入形式为$z_1, z_2 \in R^{B\times T\times C}$的数据经过构建后成为$z\in R^{B\times 2T\times C}$的数据。这个数据返回给$ContrastLoss$后，形成形状为$R^{B\times 2T\times 2T}$的相似度矩阵。这里给出$TemporalContrast$的简单代码演示：

  ```python
  def instance_contrastive(z1, z2):
      B, T = z1.size(0), z1.size(1)
      if B == 1:
          return z1.new_tensor(0.)
      z = torch.cat([z1, z2], dim=0)  # 2B x T x C
      z = z.transpose(0, 1)  # T x 2B x C
      return Z
  ```

  其中$z$就是我们需要返回给下面对比损失的数据。就是调用我定义好的

- $Hierarchical\ Pooling$: 对于长序列来说，单靠前两个Contras组件是可能不够的，所以这里有一个层次池化的选项。将$h_1, h_2 \in R^{B\times T\times C}$的数据分层池化成$s$ 个$h_1^{(s)}, h_2^{(s)} \in R^{B\times T_s\times C}$。其实就是将 $T$不断根据$kernel\_size$不断减小，直到为1，每二分一次就是一层，再通过要使用的Contrast组件来计算各个Contrast的Loss，最后加权求和还是区平均都随意了。

- $Cross-Scale\ Contrast$: 这个Contrast组件是在进行了层次池化后可进行的操作，若粗粒度特征由细粒度池化而来，则应把它们视为正样本，否则视为负样本

由于$Contrastive Pair Construction$需要的参数有三个：1.决定是否使用该Contrast方法。2.决定$Hierarchical\ Pooling$的池化方式。3.决定$Hierarchical\ Pooling$的kernel大小。

### $Contrast\ Loss$的构建细节

$AutoCLS$在构建$Contrast\ Loss$时有三个选项：1.使用何种Loss方法。2.使用何种相似度函数。3.使用温度参数的设定。

所以$Contrast\_arch$的构建我目前的构想是这样的$contrast\_arch[i]: [node\_index, Loss\_choice, [similarity\_Function, Temperature]]$,不同的地方就在第三个$param$的参数搜索。

由于$Contrast\ Loss$要配合$Contrastive\ Pair\ Construction$构建，我们需要提前构建好$Contrast\ Loss$，在$Contrastive\ Pair\ Construction$时调用。我们以$InfoNCE$为例子的伪代码：

```py
class InfoNCE(nn.Module):
    def __init__(self, ):
        
```



### $arch$数组的构建。

在$AutoCTS$++中，$arch$的含义为：$arch[0]$表示DAG中的节点索引, ,$arch[1]$表示该节点使用什么Cell，所以该arch就是构建一个$backbone$。我们需要沿用此思想，但是需要加入$arch[2]$表示该Cell中operator的option。由于每个Cell中的operator所需的参数都不相同,  我决定参考Cell设计的思路，为每个Cell设立其单独的arch如：$Aug\_arch, Emb\_arch$等。

这个在$Data\ Augmentation\ and \ Emb.\ Transformation$中是较好实现的，因为其$arch$中的$param$只有一个而且是数值。但在$Pair Construction \   and \  Contrastive Loss$中，由于我们还需要搜索$Similarity\ Functions$和$Loss\ types$等，这就不好进行arch中的$param$搜索了。

#### 我的工作与$AutoCTS$++的区别

- $AutoCTS$++：这个工作的搜索空间仅搜索Encoder模块，于是很多待选模块需要重复使用，以此找到最合适的架构。而且Encoder内部并没有待选参数，所以其架构为$arch[i]: [node\_index, op\_index]$
- 我们的工作：在我们的工作中，我们并不需要对一些组件进行多次操作(事实上只能使用一次),那我便需要考虑不使用一般的DAG架构。在$param$部分，由于各个组件的所需参数不同，我需要对每一个Cell构建其单独的arch。

于是我对于此工作arch的构想为：

1. 放弃$AutoCTS$++中的arch的网状结构而改用线性结构。
2. 对于较为复杂的$param$，我采用$param\ list$的方式，$e.g$: $ContrastLoss$中的$param: [Similarity\_function, Temprature]$,
3. 由于最终这些$arch$结构需要组合起来，并返回给 $T-AHC$组件进行比较。其中$param$的部分我打算映射如节点中，作为节点数据，让比较器不仅学习到架构特征，还能学习到参数特征。为此，我需要对复杂的$param\ list$，我需要采用一个Encoder将其映射为一个数值(直接传入向量信息是否也可取？)。



## 新的想法

我先前被局限在$AutoCTS$++ 的那种数组的arch架构，并认为这样好构建邻接矩阵。事实上，我们并不需要数组化arch，也不需要构建邻接矩阵。

- 字典化$arch$：我们将所有的arch组件和参数放置在一个字典中，有了这样的arch构建，在$operation$编写阶段，我就可以不被arch架构给束缚，可以写出高模块化的$operation$组件
- 字典编码后形成图结构：由于我们的对比学习是一个线性结构，所以我们并不需要构建DAG。加上我们的arch中也编码如何参数信息，所以$T-AHC$不仅能学习到结构信息，也能学习到参数信息(我猜这是有用的)。

于是我们的搜索空间定义可以长成这样：

```python
SEARCH_SPACE = {
    'AugCell': {
        'enabled_ops': ['resize','jitter','rescaling','point_masking','frequency_masking' ,'crop'],
        'discrete_values': DISCRETE_VALUES,
        'order_values': [1, 2, 3]
    },
    'EmbedCell': {
        'jitter_values': DISCRETE_VALUES,
        'mask_values': DISCRETE_VALUES,
        'norm_options': ['batchnorm', 'layernorm']
    },
    'PairCell': {
        'temporal': [True, False],
        'neighbor': [True, False],
        'multiscale': [True, False],
        'pool_type': ['avg', 'max'],
        'kernel_range': [3, 5, 7]
    },
    'LossCell': {
        'loss_options': ['InfoNCE', 'NTXent'],
        'sim_options': ['cosine', 'dot'],
        'temperature': DISCRETE_VALUES
    }
}
```

arch的搜索可以为：

```python
def sample_arch(search_space):
    # 离散值集合
    discrete_vals = search_space['AugCell']['discrete_values']

    # AugCell1
    enabled_ops1 = random.sample(search_space['AugCell']['enabled_ops'],
                                k=random.randint(1, len(search_space['AugCell']['enabled_ops'])))
    aug_params1 = {op: random.choice(discrete_vals) for op in enabled_ops1}

    #AugCell2
    enabled_ops2 = random.sample(search_space['AugCell']['enabled_ops'],
                                k=random.randint(1, len(search_space['AugCell']['enabled_ops'])))
    aug_params2 = {op: random.choice(discrete_vals) for op in enabled_ops2}

    # EmbedCell
    embed_jitter = random.choice(discrete_vals)
    embed_mask = random.choice(discrete_vals)
    embed_norm = random.choice(search_space['EmbedCell']['norm_options'])

    # PairCell
    pair_temporal = random.choice(search_space['PairCell']['temporal'])
    pair_neighbor = random.choice(search_space['PairCell']['neighbor'])
    pair_multiscale = random.choice(search_space['PairCell']['multiscale'])
    pool_type = random.choice(search_space['PairCell']['pool_type'])
    kernel = random.choice(search_space['PairCell']['kernel_range'])

    # LossCell
    loss_fn = random.choice(search_space['LossCell']['loss_options'])
    sim = random.choice(search_space['LossCell']['sim_options'])
    temperature = random.choice(discrete_vals)

    # assemble
    arch = {
        'AugCell1': {
            'enabled_ops': enabled_ops1,
            'params': aug_params1,
        },
        'AugCell2': {
            'enabled_ops': enabled_ops2,
            'params': aug_params2,
        },
        'EmbedCell': {
            'jitter': embed_jitter,
            'mask': embed_mask,
            'norm': embed_norm
        },
        'PairCell': {
            'temporal': pair_temporal,
            'neighbor': pair_neighbor,
            'multiscale': pair_multiscale,
            'pool_type': pool_type,
            'kernel': kernel
        },
        'LossCell': {
            'loss': loss_fn,
            'sim': sim,
            'temperature': temperature
        }
    }
    return arch
```

## 对于Encoder模块的搜索

在昨天和老师讲过之后，我们决定需要加入Encoder模块的搜索，这样的话就可以增加创新点。

我们应该是第一个提出这种想法的人(在对比学习方向)，所以我们注重的只是实现而不是细节。有些细节部分不是很完美不需要过分的在意。我们注重实现。

所以我将采用$AutoCTS$++的Encoder搜索方法。但是难点是如何对Encoder和我的Contrastive Learning结构进行编码。

### 将 $AutoCTS$++和$Contrastive Learning$方法架构进行编码(编码进入图像)

目前我的想法是：

- 将对比学习架构和Encoder架构分别搜索，其中对比学习的$Contrastive\ arch$还是一个字典形式，并在编码使仍采用$one-hot$方式编入图形空间。对于Encoder的$Encoder\ arch$，目前我决定采用数组形式，编为一个图状结构并入对比学习的DAG中(没错，前面的$Contrastive arch$也可以编码入图形空间)

目前对于$Contrastive\ arch$的编码来说，我已经将其架构编码好了。目前的工作就是将Encoder的搜索模块所构成的DAG并入对比学习架构的DAG中。最终传入$T-AHC$。注意：不要太过在意其中的细节，我们主要的工作就是实现，毕竟这是一个没人做过的工作。



从好实现的角度上看，我的建议是不在Encoder中加入需要嵌入邻接矩阵的模型，这过于繁琐，而且不好实现....





## 整理现状

- 根据$AutoCTS$++的代码， 我已经有一个对Encoder模块很好的arch架构，也有了将其转换为邻接矩阵的代码。
- 我的对比学习空间架构在大致上其实也已经架构好了。
- $T-AHC$的代码在$AutoCTS$++中有。

综上，我现在需要做的事是：

1. 构建好整个模型的架构。net
2. 构建好将模型的编入邻接矩阵空间中。geno_to_graph

我现在需要看看NAS的搜索是不是就是仅凭arch的搜索实现的。(继续阅读$AutoCTS$++的源码)







## T-AHC的功能和编码设计

通过阅读源码看来，T-AHC训练完成后，还是要进行Random_Sample来选取arch（默认20000次）。

T-AHC就是通过任务特征和arch特征学习到哪个arch对于这个任务的表现会更好，任务特征提取的对比学习模块$ts2Vec$就是为了这个task-aware的任务而设计的。

那么对于我的contrastive_arch来说，加入参数信息的邻接矩阵结构可能会更好？

现在专注于net的架构，如果加入参数的contrastive_arch好实现的话，就再加。不好实现就只是传入连接信息。



今天实现了net网络的基础功能，我在net的forward中直接返回了loss，这样在训练时调用即可？（其实我不确定这种架构设计能否在backwards时传到我的Encoder_cells中，但是在看过源码后我觉得这是可行的）。

接下来就是写出训练的代码，我在考虑是否直接放入Net中？也许这是比较好的，毕竟我这个就是一个对对比学习策略的搜索。

5/16



目前将net网络架构完成了，其中还有一些缺陷没有完成：

1. 在normalization的部分中，我没有给出Normal_shape，导致在LayerNorm时并不能进行。需要更改
2. 我目前并没有对Loss架构进行搜素，这是由于我的相似度函数还没有实现，于是目前就只是默认使用了InfoNCELoss，后期需要更改
3. 对于Encoder模块，我没有更改原文中的Encoder，这个先看效果如何，不行再对Encoder模块进行更改
4. 现在的contrastive的operator模块还是有些问题，有些时候的输入会变成nan。奇了怪了，还要改:( 这个错误的原因是我使用l2 normalization有问题，明天改

现在的Loss有点高.....让我很担心......

现在的工作：

将Contrastive_arch和encoder_arch编码进入到邻接矩阵去，并合理地输入进这个T-AHC模块。



我已经写好了一版的geno_arch_to_adj。但是现在只是进行了简单的拓扑架构的学习，并没有将参数一并学习进去。带有参数的adj写的有问题，还需要更改。

明天先根据这个已经写好的geno_arch_to_adj将arch传入进AHC中，若是能跑通，就接着写。

5/19



最难的地方莫过于将离散的dict形式的arch转化为adj并且生成对应的ops。由于这些数据需要放入GCN中进行学习，这些操作其实是必要的。



现在最新的geno_arch_adj已经能够整合拓扑结构和特征信息了。但是不知道这样的架构能否让GCN学习到。只能问问ai了:(



当ahc_engine的工作完成后，花一天的时间完善net的代码，尽可能减少后期的修改和优化。后面就可以做手训练的代码了。

终于要开始做实验了.....但愿实验效果不要辜负我吧。

5/20



现在将AHC模块已经完成了，就能实现基本的zero-shot了。

接下来需要修改的部分：

1. T-AHC的训练代码。下午弄弄
2. dataloader的代码。
3. 对比学习的contrastive_cell部分进行完善，数据增强，嵌入转换和对Loss的搜索。遗憾的是现在并没有对Loss的搜索，这个我认为其实是必要的。



今晚需要把net的训练代码写出来，就算写不出来也还是要将源代码弄明白





### 5/28

现在的任务就是弄清楚到底是哪里导致了Loss没有产生下降。现在我很难受.....



### 5/30

目前确定了需要将模型Loss下降收敛过慢的问题解决，我需要着重关注数据增强和对比损失函数部分。

- 数据增强部分：不采用AutoCLSS中极端的参数选择，改为选用较为适中和较小的数据增强。并且需要加入数据增强的顺序搜索。
- 对比损失部分：收敛速度过慢不应该是模型其他位置的问题，我认为和对比损失函数中的一些部分有关。我需要着重关注

在Loss收敛过慢的问题解决之后我才会着手下一步的实验问题。（但可能也会破罐子破摔？）



temperature=0.3时，一般的下降幅度大概为0.01？会不会是我太心急？



在将温度参数设置调小之后，下降幅度较为明显（相比于之前）。我在跑完20个epoch之后将对温度参数进一步调小。在我确定了Loss下降问题是由温度参数引起的后，我将完善对比损失部分的NAS。



当temperature=0.05时，下降的速度甚至不如temperature=1时。

当temperature-0.3时，目前来看Loss下降的速率最快。

但是我并没有消除变量，会搜索到一些增强一般的数据。



```
encoder_arch:{'AugCell1': {'enabled_ops': ['jittering', 'resize', 'frequency_masking'], 'params': {'jittering': 0.1, 'resize': 0.4, 'frequency_masking': 0.3}}, 'AugCell2': {'enabled_ops': ['frequency_masking'], 'params': {'frequency_masking': 0.1}}, 'EmbedCell': {'embedding_jittering': 0.1, 'embedding_masking': 0.1, 'normalization': 'l2'}, 'PairCell': {'temporal': False, 'pool_type': 'max', 'kernel': 3}, 'LossCell': {'loss': 'NTXent', 'sim': 'cosine', 'temperature': 0.2}}

encoder_arch: [[0, 2], [0, 0], [1, 0], [0, 1], [2, 1], [1, 2], [3, 3]]
```

这个contrastive_arch的架构目前表现是最好的。温度参数我这里设置的是0.3。loss的收敛速度还可以。最终验证Loss降到了0.8多，在训练Loss降到了0.7左右。epoch数为30

我认为之前收敛速度过慢还是因为我的contrastive_arch有时候搜索不到很好的架构。

现在确定了，我的NAS架构是有能力搜索到性能较好的对比学习模型的，根据经验来看，对于需要的epoch数量大概只需要20-30个左右。(搞半天原来是一个乌龙......)



在确定了我的NAS架构能搜索到合适的对比学习架构后，我的训练代码和dataloader代码都没有问题。可以确定接下来的任务了：

- 完善contrastive_operations和contrastive_cell部分的搜索。加入order和对于Loss的搜索。
- 将generate_seeds完成。
- 更改T-AHC的训练代码，完成zero-shot的所有NAS架构
- 赶快将T-AHC训练出来，看看T-AHC在处理zero-shot任务时花费的时间是多少。
- 搭建下游任务的环境（数据处理，模型选定等等），还有询问赖老师到底使用什么模型，实验设置应该如何。





### 6/3

在训练的generate_seeds代码完成之后，下一步的任务：

- 完成AHC模块的训练代码
- 完善Loss的搜索，现在我的Loss基本上就是没搜索
- 加入数据增强的order搜索

task_feature部分没什么问题了，现在就是上面两个问题导致我的Random_Search部分很难写出来。这就有点头疼了。

先该contrastive_operations吧

其实除了AAAI之外的任务，我还有很多学校的任务，也需要在这两天解决：

- 马原的PPT
- 希冀上的实验任务/
- 数据库上的Mooc考试/
- 大学生职业生涯规划任务





把AHC训练的代码和Random_Search的代码看明白了再说。

把课程上的任务也要跟进，明天把Contrastive_cell的搜索完善



现在的问题还是对于我们net训练的时候Loss下降缓慢，这样搜索出的大部分是很次的架构，我担心最终因为有用的样本太少导致T-AHC学习不到如何提取好的架构。这是最麻烦的。。。。。

我大概训练了20-30个架构，但这其中有用的只有1个，最终需要1000多个架构，这样的比例运气好也只有50个是效果好的，这样真的能学到吗？





我觉得Loss不下降的问题主要来自于contrastive_cell的不完善。所以我需要对Loss模块进行完全的更改，在data_augmentation部分加入order的搜索。这是必要的。。。



Loss的完善昨天已经完成了，但是现在发现在数据增强部分还是有很大的问题，有些输入的形状都发生了更改。现在更改Embed_Transformation和Augmentation部分代码，查看数据流的形状。



### 6/6

目前看来我的训练速度很慢的问题解决了，现在可以着重关注loss下降和Loss搜索等一系列问题了。目前一次epoch训练稳定只耗费2.5分钟了(其实还有增长空间)

若是将这些解决之后，后面还有一大堆问题等着我呢，真是头疼。连找个时间写随记的时间都没有了



- 数据库考试mooc上的，今晚就搞定吧，免得以后忘记
- 金仓的老师已经加上了，晚上弄弄kca/kcp报名的事情
- 还有大学生职业生涯规划的作业。。。明天就要交了。
- 晚上帮师兄写一个预测代码提交，我感觉分应该不会怎么高，我试一试我的模型，将我的模型改成multi_step的。





### 6/7

现在对于对比学习的任务基本告终了，有问题也以后再改吧，现在就直接写下游预测任务的代码了。我先写出预测任务的代码，把这个事给解决了再说。

现在看来，下游任务的代码是非写不可了。因为对于对比学习来说，只看loss的值并无意义，我不能将这个loss作为label的标准，而需要下游的预测或是分类任务上做出





### 6/9

目前我在网络中定义的encoder输入形状为[B,C,N,T]，于是在下游预测任务中也需要使用相同形状

预测模型使用：目前打算使用简单的svm或是线性探针来进行预测。

使用对比学习模型得到的结果作为下游预测的输入数据，原数据集中的标签作为训练/测试标签，以此训练测试预测模型。

所以现在在train_one_epoch的代码中，训练部分的代码不需要更改。更改验证的代码，并返回给验证的损失和评估函数什么的。



现在这个下游预测部分的逻辑基本理清楚了：

首先需要写一个正常的dataloader(带标签的)，随后将x通过encoder提取特征，作为训练输入，数据集标签正常使用。

上面将数据处理好后，就可以开始训练和评估模型。





### 6/10

今天的任务是把下游预测的代码写出来。写完后将我的DSTA-Net改为多步预测的代码改出来。

对于岭回归训练，允许的输入只能为二维输入。那么只能为[B, D] (样本数量，表征维度),其中表征维度[Timestamp x in_dim], 样本数量为[Batch_size x Nodes_nums]



### 6/11

现在下游预测任务的代码也写出来了，但是有个新的问题，就是对比学习下游预测任务模型都是在CPU上跑的。但是现在实验室的CPU占用很夸张，实验进度很慢，而且我现在的代码会出现一些警告之类的问题。让我很是难受，我担心预测效果很差，但是我最担心的是连个结果都没有，我都不能知道结果到底好不好。见鬼了。



现在速度慢主要是受限于sklearn库的一系列CPU操作，现在又有很多人在使用服务器，CPU已经被占满了:(



### 6/13

下游预测的结果我使用了P:12, Q:4的架构进行了预测，效果好像是特别好。这让我挺爽的，但是现在出现了个问题，nnd，我的代码在哪里都跑不了。

都是爆显存。

现在瞎改的把原本的代码改的都跑不动了，还好备份了。



### 6/14

现在确定了效果极好是因为过拟合了。我认为问题出在下游预测任务中。
