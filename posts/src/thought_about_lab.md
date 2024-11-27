# 对于近期两个项目的一些笔记

> 这两天跟着书做了两个project，一个是关于放假预测回归的模型，还有一个是MINST数据库的分类问题。这两个项目都是基础中的基础，这帮我增长了我的一些见识和能力，于是我觉得我有必要进行一些总结。

## California的房价中位数预测

这是书中第一个端到端的项目(虽然我并不理解端到端是什么意思)，这个项目作为初学认知很有帮助，跟我之前做的项目也有类似之处(我之前的项目全都是归一化后的数据，真的很难去做特征提取)。这里特征提取和衍生的思想和经验是完全值得学习的。



书中给出了一些机器学习项目的指导步骤清单，我将会用这个清单进行逐步作业。

- **Step_1:  框出问题看整体**

  - 确定业务目标：当然是预测放假嘛，其实这样没什么需要注意的。

  - 设计系统的前期思想：

    这当然是一个有监督学习的项目，给出了具体的房价以及其相关信息，我们需要做的就是构建模型进行回归预测。

  - 选择性能评价指标： 选择RMSE作为评价指标，这是一个很典型的性能评价指标

- **Step_2: 获取数据**

  - 要尽可能的自动化高效的获取数据：

    这里使用os，tarfile和urllib库进行操作，贴一段代码：

    ```py
    import os
    import tarfile
    import urllib.request
    
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    
    def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
        if not os.path.isdir(housing_path):
            os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
    ```

    这段代码会通过url获取tar.gz包，并在你的项目文件里创建目录放入。

  - 获取数据集地描述：

    使用head，describe等方法，让我们对于数据有个简单的第一印象。还可以使用plt地hist获取数据的直方图，更直观的获取数据的信息。

  - 这里要注意最好划分一个测试集放一边，再也不要看它。（无数据监听）

- **Step_3: 数据分析**

  这我认为是一个最重要的步骤，有句话我很喜欢'Garage In Garage Out'，数据的分析以及特征的提取固然是机器学习项目中最重要的步骤。书中也提到不要随意的进行超参数的调整。毕竟特征的创建决定了项目性能的上限，而模型的选择只是尽可能地去接近这个上限。

  - 可视化数据：这需要我们