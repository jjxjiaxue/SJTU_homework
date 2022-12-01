# 对抗攻击大作业

## 整体实验设置
- 使用 class Warp_Resnet 处理mean与std
- 使用 argparser 让代码简洁
- 使用 assert 保证符合范数规范
- 忽略模型错分的个别测试数据，ResNet-50 ACC 96.6%, vit 97.1%, 在正确分类的数据里面计算 untargeted 攻击成功率
```python
# Drop the misclassified sample
label = torch.tensor([int(item[1])]).cuda()
if pred.item() != label: continue
```

## Linf-based whitebox attack

- 使用 PGD 实现对 ResNet-50 和 vit 的白盒攻击
- Perturbation Budget: Linf-norm = 1 / 255
- 攻击轮数：40
- 每轮强度：0.1 / 255

- ResNet-50 攻击成功率：99.69%
- vit 攻击成功率：95.58%
攻击成功率已经很高，不再进行调参优化


## L2-based whitebox attack

- 使用 PGD 实现对 ResNet-50 和 vit 的白盒攻击
- Perturbation Budget: L2-norm = 0.3
- 尝试的创新点：增加迭代次数，并使用类似（learning rate decay）的思想来降低每一个step的系数，从而拿到更好的优化结果。

### 实验结果：

#### PGD attack on ViT

每次迭代 epsilon_iter = 0.04，如果使用decay，在后一半迭代次数中，epsilon_iter /= 2

| 迭代次数 | 80 | 40 |
| :-----| ----: | ----: |
| 使用decay | 64.1% | 60.6% |
| 不适用decay | 63.3% | 60.5% |

结论: 最好成功率64.1%，增大迭代次数会很有帮助，但会显著增加攻击时间，使用decay可以帮助一点点。

#### PGD attack on resnet50

迭代次数：80，使用decay，epsilon_iter = 0.04，成功率：85.6%

## Patch-based whitebox attack

- 使用 mask 保证范数规范
```python
adv = original_image * (1 - mask) + x * mask
```
- 实现思路：首先选取patch，其次对patch内像素使用 PGD-Linf 进行攻击，根据之前经验，增大攻击轮数从而提升攻击强度，并使用decay技巧
- 攻击轮数：80

探究创新点：如何选取patch

### On ResNet-50

此报告中探究四种patch位置选取方式，选取前50个数据为验证集，括号中为攻击成功率，epsilon_iter：0.05 / 256

- Patch选择左上角（32%）：听起来就是最差的，事实也是
- Pathc选择中间（54%）：中性策略？事实是最好的
- 使用16x16平均卷积去计算梯度绝对值之和，选择梯度信息最大的patch（46%）：效果不是最好的原因可能因为patch-based attack中所有pixel变化幅度巨大，梯度信息只能决定“微小”变化内影响最终logits大小，因此不准。 
- 每隔4个pixel，丢掉16x16的patch，观察丢掉次patch是否能让logits减小（40%），以此来衡量patch的重要性：实验中发现总是倾向于丢掉左上角附近的patch，而不是网络中间，并且logits减小程度也不大，因此效果不是最好。

探索结果：pathc放在最中间，效果最好，纵使可以通过不同位置多次攻击选择最好，但过于浪费攻击时间。报告尝试去用极短的时间去找合适的位置，但遗憾并没有找到。

最终实验结果：epsilon_iter = 0.1 / 255,  选择[105, 105]（上述方案2）为adversarial patch起始位置，攻击成功率62.2%

### On vit

思考：如何选择patch，是否需要考虑vit 16x16 patch embedding结构？

根据vit 源代码，每一个patch embedding成768的feature是通过卷积实现，此卷积将每一个patch 16x16个像素，通过 256 x 768 线性层，映射到特征空间。

```python
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```
- 选择把一个patch内所有像素改变，可以直接实现完全操控此patch的embedding feature，但不能改变其他的patch embedding
- 选择四个vit patch的交汇区域，可以实现同时影响四个相邻patch的feature，但影响程度有限

具体的效力取决于影响四个相邻patch更有用，还是完全操控一个patch feature更有用，此处难以下定论。

最终实验结果：epsilon_iter = 0.1 / 255, 选择[105, 105]为adversarial patch起始位置，攻击成功率73.8%

### 思考：CNN, ViT, adversarial patch分布区域有什么规律
- 经过查找CNN与ViT的介绍，由于 CNN 更容易受到texture影响，而ViT更多关注全局相关性，因此猜想：CNN的adversarial更倾向于与texture密集区域，而ViT的patch更容易与content相关。

## 迁移攻击

- 实验设置：同时忽略被两个模型都分错的样本，报告攻击准确率，使用Linf对抗样本
- 用攻击 resnet-50 的对抗样本攻击 vit，成功率 0.63%
- 用攻击 resnet-50 的对抗样本攻击 vit，成功率 0.95%

## 黑盒L2攻击

- 实现方法: Simple Black Box Attack (SimBA)
- L2 范数: 5
- 最大迭代次数: 10000
- 每次迭代增加noise的L2范数 eps_iter = 0.004
- ViT 攻击成功率 48.6% 平均query数量2080次查询
- ResNet 攻击成功率 61.1% 平均query 数量 1750次查询
- 由于运算力有限, 没有进行调参优化, 应该还有空间.

对创新点的思考:

- 通过阅读LeBA论文, 尝试复现使用替身网络gradient map来指导增加noise这一思路, 在gradient map大的地方多进行随机noise搜索, 听上去就是一个work的方法. 此项目中, 每隔100轮对 adv_image 进行 TIMI 攻击, 然后计算高斯平滑后的adv_noise作为引导区域. 并根据引导区域对随机产生的noise进行normalization, 引导区域大的地方多进行随机noise搜索.

对比实验结果: 

- 关于在ViT上降低搜索空间: 由于ViT的每一个patch的像素会使用同一个(3x16x16 -> 768, stride = kernelsize 的卷积)ensemble成一个单独的feature vector, 数学上等价于不同patch都会经过一个 3x16x16 -> 768的全连接层进行处理, 因此不同patch的像素不会互相影响, 实际搜索adv_noise可以分patch进行, 从而降低搜索空间, 但算力限制没有做出来.

(ViT将patch embed到特征空间的 1x1 卷积层)
```python
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
```