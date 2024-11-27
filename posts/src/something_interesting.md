# 木卫三上的时间



## 星际穿越 - 时间膨胀

我们都看过应该都看过星际穿越吧，主角们进入了一个巨型行星，这颗星球被称为“米勒星球”。它靠近一个极大质量的黑洞——Gargantuna。极大的质量产生极大的引力，根据广义相对论，引力会引起时间膨胀。这就导致了他们在米勒星球仅仅过了一个小时，地球已经度过了几十年了。这段经历虽然经过了电影的夸张，但是确实是有科学背书的。



当时我看到这个真的是觉得太神奇了（毕竟我从小就对天文感兴趣），但是碍于我当时甚至都没听说过相对论的名号，对其原理的探究也就此作罢。但是今非昔比了，虽然我还是不懂相对论  :) 但是我知道了有时间膨胀理论这个事了，所以简单的算一算我想要的时间膨胀还是可以的。



## 那木星呢？

我们都知道，太阳系中质量最大的行星是木星，在木星有一个卫星叫木卫三 ——Ganymede，它的质量和地球差不多。处在太阳系中最大行星的旁边，我有理由相信木星的引力会引起木卫三上一定程度的时间膨胀。当然这样的时间膨胀可以忽略不记，毕竟木星的质量和太阳的质量不是一个量级的。

我想知道在盖米尼德（也就是木卫三）上的时间膨胀程度会是多少，因为地球和木卫三都差不多诞生于一个时期嘛，这么说起来的话，搞不好木卫三比地球还要年轻一些呢。有没有觉得很神奇？



## 相对论与时间膨胀公式

再次重申，我真的不懂相对论的原理，我只知道引力会影响时间的流逝。在强引力场中，时间的流逝会更慢，别的我一概不知。但是广义相对论中给出了时间膨胀的数学公式，这就能让我这个小白算一算我想要的数据了。
$$
\frac{t_0}{t} = \sqrt{1 - \frac{2GM}{rc^2}}
$$
其中：

- t<sub>0</sub> 是指远离引力源的时间流逝（这里指的就是地球的时间流逝了）
- t是指在引力场中的时间流逝（即木卫三上的时间流逝速度）
- G 是引力常数， 
- M 是木星的质量
- r 是木卫三与木星的平均距离
- c 是光速

有了这些公式和数据，我们就能简单的算出时间膨胀了.......吗？简单算算就知道这个太难算了，数字小的可怜。于是我决定扩大时间度量单位。我看看在地球度过了一年后，盖米尼德的时间会慢多少，并且使用计算机来帮助我计算

```py
import math

# Constants
G = 6.674 * 10**-11  # gravitational constant, m^3 kg^-1 s^-2
M_jupiter = 1.898 * 10**27  # mass of Jupiter, kg
r_ganymede = 1.07 * 10**9  # average distance between Ganymede and Jupiter, m
c = 3 * 10**8  # speed of light, m/s

# Calculate time dilation factor t_0 / t
time_dilation_factor = math.sqrt(1 - (2 * G * M_jupiter) / (r_ganymede * c**2))

# Calculate the time difference for a year (31,536,000 seconds in a year on Earth)
time_difference_per_year = 31_536_000 * (1 - time_dilation_factor)  # seconds

time_dilation_factor, time_difference_per_year
```



## 同样有趣的结果

最后我们能得到time_difference_per_year，根据地球度过一年的时间（31536000秒），木卫三的时间一年会比地球慢上0.0415秒。这世间确实足以忽略不计了。也可想而知在天文学中的那些足以引起明显时间膨胀的星体的质量会是个怎样惊人的数字。

地球与木卫三都生成与45亿年前，我们假设他们俩年龄相同。经过了45亿牛，以及木卫三的略微的时间膨胀，木卫三的实际年龄会比地球年轻多少呢？还是用计算机简单算算便知道了。

```py
# Total time difference per Earth year on Ganymede compared to Earth (previous calculation)
time_difference_per_year = 0.0415  # seconds per year

# Total years since formation (4.5 billion years)
years_since_formation = 4.5 * 10**9

# Calculate total time difference over 4.5 billion years
total_time_difference = time_difference_per_year * years_since_formation  # seconds

total_time_difference  # total seconds over 4.5 billion years
```

最后给出的结果是186750000.0秒，换算成年也就是大约5.93年。这么说来，要是地球和木卫三同年生，那么木卫三还要比地球年轻个6岁呢 :)