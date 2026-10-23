> **内部教学草稿。** 本文件保留为思路整理，不作为论文或提交材料的附录；与 `paper.md` 冲突时，以后者为准。


### Sum of random variables

设 $Z = X + Y$，$X,Y$ 独立，密度分别为 $f_X, f_Y$。
$Z$ 在某点 $z$ 的密度是所有使 $X+Y = z$ 的 $(x,y)$ 组合的积分：
$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x)\, f_Y(z-x)\, dx$$
直觉：$Z=z$ 也可以写成 $X=x$ 且 $Y=z-x$。对所有可能的 $x$ 求和（积分），每个组合的概率密度是 $f_X(x) \cdot f_Y(z-x)$（独立性保证乘积）。这个积分形式就是 $f_X$ 和 $f_Y$ 的卷积。
进一步：$n$ 个独立同分布项的和 → $n$ 重卷积。直接算很繁，但 Fourier 变换把卷积变成乘积，一步化繁为简。

### CF

特征函数 $\psi(t) = E[e^{itX}]$ 是连接统计量的抽样分布和其 Edgeworth 展开的桥梁：
1. 累积量提取：对 $\log \psi(t)$ 做 Taylor 展开，$\log\psi(t)=\sum_{k\ge1}\kappa_k (it)^k/k!$，$t^k$ 的系数是 $i^k\kappa_k/k!$；其中 $\kappa_3,\kappa_4$ 给出偏度 γ₁、峰度 γ₂——这些正是 Edgeworth 展开的核心输入。
2. Fourier 逆变换：$\psi(t)$ 乘以 $(it)^k$ 后做逆 Fourier → 对应密度函数求导 → 产生 Hermite 多项式 → 积分得到 CDF 的 Edgeworth 校正项。
简言之，CF 把"统计量分布的矩特性"翻译成"CDF 的逐阶校正项"。

### Intuition of CF (FT on PDF)

1. 独立和 → 乘积 → 加法。 统计量 $S_n$ 往往可以写成独立项之和（或近似为和）。独立随机变量之和的密度是卷积，做 Fourier 变换后卷积变乘积：$\psi_{X+Y}(t) = \psi_X(t) \cdot \psi_Y(t)$。这比直接卷密度好处理得多。
2. Taylor 展开 → 逐阶逼近。 $\log \psi(t)$（累积量生成函数）在 $t=0$ 处展开，$t^k$ 系数恰好是 $i^k\kappa_k/k!$（差一个 $i^k/k!$ 因子）。把展开后的 $\psi(t) = e^{\text{多项式}}$ 逆 Fourier 回去：
- 高斯项 $e^{-t^2/2}$ → $\varphi(x)$（CLT 主项）
- $(it)^k$ 乘上去 → 对 $\varphi(x)$ 求 $k$ 阶导 → Hermite 多项式
这样就把一个关于"n 很大"的极限问题，变成了一个关于"矩特性的多项式"的逐阶计算——CF 是整个管线的数学字典。

### Hermite多项式

Edgeworth 展开里用的是概率论版 Hermite 多项式 $\operatorname{He}_k(x)$，递归定义：
$$\operatorname{He}_0(x)=1,\quad \operatorname{He}_1(x)=x,\quad \operatorname{He}_{k+1}(x)=x\,\operatorname{He}_k(x)-k\,\operatorname{He}_{k-1}(x)$$
前几个：
- $\operatorname{He}_2(x)=x^2-1$
- $\operatorname{He}_3(x)=x^3-3x$
- $\operatorname{He}_4(x)=x^4-6x^2+3$
为什么出现在这里： 高斯密度 $\varphi(x)$ 求 $k$ 阶导数的结果是 $\varphi(x)$ 乘以 $\operatorname{He}_k(x)$（差一个符号）：
$$\frac{d^k}{dx^k}\varphi(x) = (-1)^k \operatorname{He}_k(x)\,\varphi(x)$$
而 CF 里的 $(it)^k$ 做 Fourier 逆变换恰好对应求 $k$ 阶导。所以 Edgeworth 展开的多项式项 $p_k(x)\varphi(x)$ 自然就是 Hermite 多项式的线性组合——系数由偏度、峰度等累积量决定。
直觉：$\operatorname{He}_k$ 是 $\varphi(x)$ 的 $k$ 阶"变形基底"——高斯形状（CLT 主项）上叠加一层偏度变形（$\operatorname{He}_2$ 项），再叠加一层峰度变形（$\operatorname{He}_3, \operatorname{He}_4$ 项），逐阶逼近有限样本的真分布。


### 正态下 Bartlett 校正因子

指的是计算 Bartlett 统计量 $B$ 时公式里的分母项（不是 $B^*$ 的峰度校正）：
$$B = \frac{(n_T - k) \log S_p^2 - \sum (n_j - 1) \log S_j^2}{\underbrace{1 + \frac{1}{3(k-1)}\left(\sum \frac{1}{n_j-1} - \frac{1}{n_T - k}\right)}_{\text{Bartlett 校正因子}}}$$
作用：即使数据正态，$\log S^2$ 的分布也不恰好是 $\chi^2$——Bartlett (1937) 把原始的似然比统计量除以这个因子，使前几阶矩与 $\chi^2(k-1)$ 的矩在更高阶上吻合（Bartlett 因子消除的是 $O(n^{-1})$ 的矩误差，文献中常表述为把矩匹配推进到 $O(n^{-2})$ 或更高，具体阶数依文献的展开约定而异，引用时需核对原文）。这就是为什么 Table 1 说"Bartlett correction improves but does not render the approximation exact"——这个因子的作用是矩匹配到三阶，但分布本身仍不是精确 $\chi^2$。

| 名称                | 在哪里                              | 修什么                       |
| ----------------- | -------------------------------- | ------------------------- |
| Bartlett 校正因子     | $B$ 的分母                          | 提高正态下 $B\sim\chi^2$ 的逼近精度 |
| Box–Andersen 峰度校正 | $B^* = B/(1+\hat\gamma_{2,p}/2)$ | 非正态下消除 $\gamma_2$ 引起的标度膨胀 |


### Polynomial-Hermite

指 Edgeworth 展开里由偏度多项式 × 高斯型密度构成的那一项,即 :
$$n_T^{-1/2}\, p_1(x)\, f_0(x)$$
为什么叫 "polynomial-Hermite":

- $p_1(x)$ 是多项式部分。对应 $p_1(x) = -\dfrac{\gamma_1}{6}(x^2-1)$。
- 括号里的 $x^2-1$ 正是概率论 Hermite 多项式 $\operatorname{He}_2(x)$。
- 它来自对高斯密度求导:$\dfrac{d^k}{dx^k}\varphi(x) = (-1)^k \operatorname{He}_k(x)\varphi(x)$(见前文),而 CF 里的 $(it)^k$ 逆变换恰好对应求 $k$ 阶导 → 生成 Hermite 多项式。

所以整项形如 $p_k(x)\varphi(x)$ = "多项式(Hermite)× 光滑高斯密度",这就是光滑机制贡献的项。
对照 §3.2 的格点式:
$$F_{S_n}(x) = F_0(x) + \underbrace{n_T^{-1/2}\, p_1(x)\, f_0(x)}_{\text{smooth polynomial-Hermite term}} + \underbrace{n_T^{-1/2}\, \Delta_{\text{lattice}}(x)\, f_0(x)}_{\text{锯齿/格点项}} + n_T^{-1}\, p_2(x)\, f_0(x) + \cdots$$


### Lattice correction term(格点修正项) 

$$\Delta_{\text{lattice}}(x) = \frac{1}{2} - \left\{\frac{x}{h}\right\}$$ 

$\{\cdot\}$ 是小数部分(fractional part)运算:
$$\{y\} = y - \lfloor y \rfloor$$
即取 $y$ 减去其向下取整 $\lfloor y \rfloor$,结果永远落在 $[0,1)$。
例子:
- $\{3.7\} = 0.7$
- $\{5.0\} = 0$
- $\{2.25\} = 0.25$
在公式里: $\left\{\dfrac{x}{h}\right\}$ 衡量 $x$ 落在格点间距 $h$ 的哪个位置——恰好在格点上时为 0,在两格点正中间时为 0.5。
于是
$$\Delta_{\text{lattice}}(x) = \frac{1}{2} - \left\{\frac{x}{h}\right\}$$
是一个周期为 $h$、幅度在 $[-\tfrac12, \tfrac12]$ 之间的锯齿波:在每个格点处取 $+\tfrac12$,线性下降,到下一个格点前趋近 $-\tfrac12$,然后跳回。这正是离散阶梯 CDF 与光滑近似之间的偏差形状。