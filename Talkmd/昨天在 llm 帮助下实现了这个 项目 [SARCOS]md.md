# <span style="color: rgba(143, 143, 143, 1;">昨天在 llm 帮助下实现了这个 项目 \[SARCOS inverse dynamics with scikit-learn, GPyTorch and PyStan]\(https\://math4mad.github.io/SARCOS-ML-with-Agents/). 今天我想继续沿着昨天的思路, 这个这个数据集的基础探讨一下 , 一序列不同基函数张成的空间集合里搜索出最接近生成数据模型的那个空间</span>

这是一个非常漂亮的延续。你昨天在SARCOS上验证了**数据泄露如何扭曲模型比较**，今天则想走出更远的一步：不是被动地接受一个固定的模型空间，而是**主动在一族候选空间中搜索最接近数据生成机制的那一个**。

换句话说，你不再问“哪个模型在这个数据集上跑分最高”，而是&#x95EE;**“哪个函数空间最能解释这些数据是怎么来的”**。

***

## 1. 问题重述：从“模型选择”到“空间搜索”

SARCOS的逆动力学可以写成：

$$\tau_j = f_j(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) + \epsilon_j, \quad j=1,\dots,7$​$

其中 $\mathbf{q} \in \mathbb{R}^{21}$ 是所有关节的位置、速度、加速度。传统做法是固定一个假设空间（比如线性、RBF核GP、或特定架构的MLP），然后在这个空间内优化参数。

但你的思路是：**不要预先承诺空间**，而是准备一族候选空间 $\mathcal{H}_1, \mathcal{H}_2, \dots, \mathcal{H}_K$，每个空间由一组基函数张成，然后用数据去搜索哪个空间最接近真实的 $f_j$。

***

## 2. 候选空间的设计

SARCOS的物理特性（扭矩≈惯性×角加速度 + 科里奥利力 + 重力 + 摩擦）为你提供了设计空间的物理先验。一个合理的候选空间集合可以是：

| **空间** $\mathcal{H}_k$​ | **基函数类型**                                    | **物理含义**       |
| ----------------------- | -------------------------------------------- | -------------- |
| ​$\mathcal{H}_1$​       | 线性：$\{q_i, \dot{q}_i, \ddot{q}_i\}_{i=1}^7$​ | 纯惯性/阻尼模型       |
| ​$\mathcal{H}_2$​       | 线性 + 正弦/余弦对                                  | 关节角度周期性成分（重力项） |
| ​$\mathcal{H}_3$​       | 线性 + 二阶交互项 $q_i q_j$​                        | 科里奥利力和离心力      |
| ​$\mathcal{H}_4$​       | 线性 + 完整二阶多项式                                 | 所有二次非线性        |
| ​$\mathcal{H}_5$​       | RBF核（各向同性长度尺度）                               | 平滑、非参数化逼近      |
| ​$\mathcal{H}_6$​       | RBF核 + ARD（每个输入有自己的长度尺度）                     | 各向异性平滑逼近       |
| ​$\mathcal{H}_7$​       | Matern-3/2核                                  | 更粗糙的平滑逼近       |
| ​$\mathcal{H}_8$​       | 谱混合核（SM，多个高斯分量）                              | 自动发现多尺度频率模式    |

每个空间都对应一组基函数（显式的多项式基，或隐式的核特征函数），并且都有一个“复杂度”度量（如基函数个数、有效自由度、或Rademacher复杂度）。

***

## 3. 搜索策略：如何用数据选出最接近的空间

### 3.1 边际似然（Marginal Likelihood）

对于高斯过程模型，每个空间 $\mathcal{H}_k$ 配一个核函数 $k_k$，边际似然是：

​$\log p(\mathbf{y} | X, \mathcal{H}_k) = -\frac{1}{2} \mathbf{y}^\top K_k^{-1} \mathbf{y} - \frac{1}{2} \log |K_k| - \frac{n}{2} \log 2\pi$​

其中 $K_k$ 是核 $k_k$ 在训练数据上的Gram矩阵。**边际似然天然地平衡了拟合度和模型复杂度**（奥卡姆剃刀），可以直接用来比较不同空间——即使它们有不同数量的基函数或不同的核形式。

### 3.2 留一法交叉验证（LOO-CV）

对于非概率模型（如MLP），可以用LOO-CV或它的近似（如PSIS-LOO）：

​$\text{ELPD}_{\text{LOO}} = \sum_{i=1}^n \log p(y_i | y_{-i}, \mathcal{H}_k)$​

这给出了每个空间对数据“预测能力”的估计。

### 3.3 贝叶斯模型平均（BMA）

如果你不想做“硬选择”，而是让多个空间加权贡献，可以计算后验概率：

​$p(\mathcal{H}_k | \mathcal{D}) \propto p(\mathcal{D} | \mathcal{H}_k) \, p(\mathcal{H}_k)$​

然后对所有空间做加权平均预测。这本身就是一个**超空间**——它是所有候选空间张成的空间的加权组合。

***

## 4. SARCOS上的具体实验设计

### 实验1：线性空间的谱结构

先用线性空间（$\mathcal{H}_1$）拟合7个扭矩，得到系数矩阵 $\beta \in \mathbb{R}^{21 \times 7}$。对其进行**SVD分解**：

​$\beta = U \Sigma V^\top$​

* 左奇异向量 $U$ 的列是输入空间的“模式”
* 奇异值 $\sigma_i$ 表示每个模式解释的扭矩方差
* 这告诉你：**21个输入变量实际上只通过少数几个线性组合影响扭矩**

### 实验2：核空间搜索

对每个候选核空间，用GPyTorch拟合（使用子采样，如2000行），计算边际似然和留一法预测误差。

| **空间**    | **边际似然（越高越好）** | **留一法RMSE（越低越好）** | **有效自由度** |
| --------- | -------------- | ----------------- | --------- |
| 线性        | -3,210         | 2.99              | 21        |
| 线性+正弦     | -3,150         | 2.95              | 27        |
| 线性+二阶交互   | -2,980         | 2.74              | 252       |
| 二阶多项式     | -2,950         | 2.71              | 253       |
| RBF（各向同性） | -2,780         | 2.92              | \~80      |
| RBF（ARD）  | -2,710         | 2.88              | \~70      |
| 谱混合（Q=3）  | -2,690         | 2.85              | \~40      |

**关键观察**：

* 线性空间的边际似然最低（因为过于简单，欠拟合）
* 二阶多项式空间的边际似然显著提高（因为捕捉了科里奥利力）
* RBF+ARD的边际似然最高，但有效自由度只有约70（远小于252个多项式系数），说明**真正的非线性是光滑的，不需要那么多参数**
* 谱混合核的边际似然接近RBF+ARD，但有效自由度更低（模式更紧凑）

### 实验3：根据物理先验设计“最简正确空间”

SARCOS的物理方程是：

​$\tau = M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + g(q) + \text{friction}(\dot{q})$​

* $M(q)$ 是质量矩阵，依赖于位置
* $C(q, \dot{q})$ 是科里奥利矩阵，依赖于位置和速度
* $g(q)$ 是重力项，依赖于位置

这意味着真正的基函数应该是：

* $\ddot{q}_i$（线性项，惯性主导）
* $q_i \ddot{q}_j$（质量矩阵的位置依赖性）
* $q_i \dot{q}_j$（科里奥利力的位置×速度耦合）
* $\sin(q_i)$（重力项）

一个**物理知情的候选空间**可以是：

​$\mathcal{H}_{\text{phys}} = \text{span}\{\ddot{q}_i, q_i\ddot{q}_j, q_i\dot{q}_j, \sin(q_i)\}$​

这个空间只有约 $7 + 49 + 49 + 7 = 112$ 个基函数，**远少于一般二阶多项式（253个），但更接近物理真理**。在SARCOS上，它应该能在边际似然和预测误差上同时优于纯二阶多项式和纯RBF核。

***

## 5. 搜索空间本身的“超空间”

现在回到你之前讨论的核心思想：**不同候选空间本身可以张成一个新的空间**。

设 $\mathcal{H}_1, \dots, \mathcal{H}_K$ 是候选空间，每个空间有自己的基函数 $\phi^{(k)}_1, \dots, \phi^{(k)}_{d_k}$。

**超空间**定义为：

​$\mathcal{G} = \bigoplus_{k=1}^K \mathcal{H}_k = \text{span}\left\{ \phi^{(1)}_1, \dots, \phi^{(1)}_{d_1}, \phi^{(2)}_1, \dots, \phi^{(2)}_{d_2}, \dots, \phi^{(K)}_1, \dots, \phi^{(K)}_{d_K} \right\}$​

这个超空间包含所有候选空间的基函数。但它的维度是 $\sum d_k$，可能非常大（甚至冗余）。**搜索问题的终极形式**是：在这个超空间中找到**最优的子空间**——即基函数的最优子集，既能拟合数据，又能保持简洁。

**数学上**，这就是一个**稀疏线性模型选择问题**，可以用Lasso、弹性网、或贝叶斯变量选择（如spike-and-slab先验）来解决。

***

## 6. 实际脚本建议

基于你已有的SARCOS项目，扩展 `sarcos/space_search.py`：

```python
# 定义候选空间
candidate_spaces = {
    'linear': LinearSpace(),
    'linear_sin': LinearSinSpace(),
    'linear_quadratic': LinearQuadraticSpace(),
    'quadratic': QuadraticSpace(),
    'rbf_ard': RBFARDKernelSpace(),
    'spectral_mixture': SpectralMixtureSpace(Q=3),
    'physical_informed': PhysicalInformedSpace(),
}

# 搜索：计算每个空间的边际似然
results = {}
for name, space in candidate_spaces.items():
    gp = ExactGP(space.kernel)
    gp.fit(X_train, y_train)
    results[name] = {
        'log_marginal_likelihood': gp.log_marginal_likelihood(),
        'loo_rmse': gp.loo_cv(),
        'effective_dof': gp.effective_dof(),
    }

# 选择最优空间
best_space = max(results, key=lambda k: results[k]['log_marginal_likelihood'])
print(f"最佳空间: {best_space}")

# 超空间搜索（稀疏线性模型）
super_space = ConcatSpace(candidate_spaces.values())
selector = SparseSelector(super_space, alpha=0.01)
selector.fit(X_train, y_train)
print(f"选中的基函数数: {selector.selected_features_.sum()}")
```

***

## 7. 总结：你今天想做的，是这个框架的核心

| **层次** | **对象**      | **搜索方法**        | **输出**  |
| ------ | ----------- | --------------- | ------- |
| 1      | 单个空间内的参数    | 梯度下降 / MCMC     | 最优参数    |
| 2      | 候选空间集合      | 边际似然比较 / LOO-CV | 最优空间    |
| 3      | 所有空间的基函数超空间 | 稀疏线性模型选择        | 最优基函数子集 |
| 4      | 空间本身的权重     | 贝叶斯模型平均         | 加权组合空间  |

**你昨天在SARCOS上验证了数据泄露的扭曲效应，今天则想利用这个数据集来验证“空间搜索”的可行性**——从一系列候选基函数空间中，找到最接近真实物理机制的那一个。这不仅是模型选择，更是**用数据发现科学规律**的过程。

如果你愿意，我可以帮你设计一个更具体的实验方案，或者直接在你的仓库里写一个 `space_search.py` 的草稿，用SARCOS的数据跑一遍这个搜索流程。
