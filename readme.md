# 关于GAN的学习心得

**一些个人理解**

### 一些数学概念

#### 随机变量
变量的值无法预先确定，仅以概率取值的变量，包含连续和离散随机变量和

#### 概率分布
- 离散概率分布：概率函数f(x)给出了随机变量x取某个特定值的概率，可用列表来表示
![离散概率分布](https://pic1.zhimg.com/80/v2-c8a89985d152b0337511bf8dd43aec44_1440w.jpg)
- 连续概率分布：f(x)被称为概率密度函数，没有直接给出概率，只能求在耨个区间内取值的概率
![连续概率分布](https://pic1.zhimg.com/80/v2-2ffd2245ebc32c480eab316c2194afc0_1440w.jpg)

#### 数据的概率分布
一张8x8像素的灰阶图片，是64维度空间中的一个点，每一个维度的值都代表一个位置的像素值。  
例如狗图片的概率分布，首先需要把图片resize成相同分辨率，才具有可比性，比如8x8.那么每一只狗都是64维空间的一个点，那么所有的狗图片就构成了此空间的一个分布，在这个分布内，随机采样出来的点都是狗的样子。
![数据概率分布](https://pic3.zhimg.com/80/v2-9807e30da8f358096f43e4dfc223dee2_1440w.jpg)

#### 最大似然估计
模型参数未知，通过抽样计算出模型参数

### GAN原始论文笔记
#### 原始Loss
$$
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

minmax游戏，判别网络D希望V取最大，生成网络G希望V取最小，分别来看，对于判别网络：
$$\max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]$$

$V(D,G)$ 相当于-BCELoss，判别网络接收到真实数据和生成数据，并希望判别效果最好；
$$
\min _{G} V(D, G)=\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

生成网络希望-BCELoss尽可能小，代表让判别器不能分辨生成数据和真实数据的差别。上式在训练初期，容易产生梯度消失，因为判别网络更容易训练，返回生成网络的梯度很小，Ian提出了一个生成网络的替代loss：
$$
\max _{G} V(D, G)=\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log D(G(\boldsymbol{z}))]
$$
证明见：https://www.depthfirstlearning.com/assets/gan_gradient.pdf

两个网络动态平衡。这个minmax游戏其实是真是分布和生成数据分布的JS散度的近似（证明见 https://jonathan-hui.medium.com/proof-gan-optimal-point-658116a236fb），

#### WGAN
用Wasserstein距离作为新的loss，wloss，并且需要lipschitz-1约束，梯度的绝对值至多为1
### L1约束的做法
- weight clipping：稳定性比较差
- spectral normalization：模型效率较低
- gradient penalty：效果最好
### WGAN-GP
软性的L1约束，
