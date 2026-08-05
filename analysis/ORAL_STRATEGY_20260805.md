# ICLR Oral 主线升级决策（2026-08-05）

## 结论

项目目标不降级，但旧中心命题必须替换：不再把 IVS 或任一 activation-density score 宣称为一般 causal-validity certificate。新的中心主线是：

> **Restoration is zero-order evidence. A mechanistic claim requires agreement of the local interventional response field under an explicitly specified target intervention distribution.**

中文表述：行为恢复只匹配一个函数值，是零阶证据；机制恢复至少需要在明确的目标干预分布下，局部干预响应场也一致。

暂定题目：

> **When Restoration Lies: Behavioral Recovery Is Only Zero-Order Evidence of Mechanism**

或更理论化的版本：

> **Restoration Is Not Identification: Interventional Response Signatures for Mechanistic Interpretability**

这不是把原论文收缩成 cautionary result，而是把原来的启发式 validity classifier 升级为一个“不可识别性结果 + 正面识别条件 + 可执行审计方法”的完整理论—实验闭环。

## 为什么旧主线不能原样保留

7 月审计后的实验已经确定：

1. 旧百万级 reconstruction z-gap 来自低多样性 support、训练内标定和 scale floor；cross-fit 后最大绝对 reconstruction z 仅 4.94。
2. 旧低-overlap 标签依赖 reference；corrupt reference 下的 9 个低-overlap site 在 clean、mixture、matched-counterfactual reference 下全部翻转。
3. 旧 pooled NMH AUROC 被 layer、position 和时间可达性混杂。
4. 新的同位置、时间可达实验仍稳定证明 L4--L8 存在高行为恢复、低 NMH recovery，但 overlap 约为 0.49--0.53，并不低。
5. 公平 baseline 下 IVS 没有优于 activation MLP；已有标签也只是 distribution shift，不是 mechanism ground truth。
6. 已知真值合成任务支持 cross-fit overlap 在目标 support 明确时识别 off-support donor，但不支持一般 causal validity。

所以最强的保留事实不是“低 IVS 导致机制绕过”，而是：

> **高恢复可以同时发生在 off-support 与 on-support 区域；support compatibility 和 mechanism agreement 是正交证据。**

## 截至 2026-08 的新颖性边界

必须主动避开的重叠是：

1. Grant et al., ICLR 2026 已研究干预产生的 divergent representations，并区分 harmless 与 pernicious divergence。
2. Sutter et al., NeurIPS 2025 已证明不受限的 causal-abstraction alignment 可以让任意网络匹配任意算法，高 interchange accuracy 也可能空洞。
3. Vaidyanathan et al., arXiv 2026-06 已证明 transformer activation patching 的 NIE 混入 mediator--bypass interaction，并在 IOI 上解释排名和 faithfulness 异常。
4. Guo et al., arXiv 2026-06 已使用 matched donor、mis-specification bound 和 off-manifold diagnostic 分析 interchange intervention。
5. Lin and Liu, arXiv 2026-05 已提出 mechanistic-interpretability causal claim 必须披露 identification assumptions 的 position argument。

因此以下版本不足以冲 oral：

- “我们发现 restoration 不等于 mechanism”；
- “我们提出另一个 off-manifold/OOD score”；
- “我们要求作者说明 reference 或 identification assumptions”；
- “我们把 IVS 换成 conformal p-value”。

conformal calibration 可以是重要模块，但不能单独充当论文主创新。

## 新的理论对象：Interventional Response Signature

设 patch site 后的下游映射为 \(f(h)\)，clean/counterfactual target state 为 \(h_c\)，patched state 为 \(h_p\)。

### 零阶证据

行为恢复只检查：

\[
f(h_p) \approx f(h_c).
\]

这只说明两个状态落在近似相同的输出 level set 上，不限制它们附近的计算。

### 一阶/二阶机制证据

对一组目标分布允许的局部 probe \(\delta\)，比较：

\[
f(h_p+\delta)-f(h_p)
\quad\text{与}\quad
f(h_c+\delta)-f(h_c).
\]

随机方向有限差分构成 **Interventional Response Signature (IRS)**。一阶 IRS 是下游 Jacobian 的随机 sketch；加入对称或成对 probe 后可估计曲率/interaction 信息。

这给出一个任务通用的 mechanism witness：它不依赖预先知道 IOI 的 NMH，也不把单一输出恢复当成机制证据。

## 计划中的核心理论结果

### T1. Zero-order non-identifiability

即使 \(f(h_p)=f(h_c)\)，局部响应 \(J_f(h_p)\) 与 \(J_f(h_c)\) 可以任意不同。因此 behavioral restoration 对局部功能机制不识别。

目标不是只给一个反例，而是给一族构造和无上界结果：在固定 restoration error 下，response-field discrepancy 可以任意大。

### T2. Local response sufficiency bound

若 \(f\) 在两个邻域内二阶平滑，且一阶 response discrepancy 有界，则对半径 \(r\) 内的允许 probe，干预效应差满足：

\[
\left|[f(h_p+\delta)-f(h_p)]-[f(h_c+\delta)-f(h_c)]\right|
\leq \epsilon_J r + C_H r^2.
\]

这把“mechanism agreement”变成可证伪、可估计的局部功能性主张。二阶扩展与 mediator interaction 直接连接，但不重复已有的 NIE/PIE 分解。

### T3. Reference-relative admissibility

IRS 的 probe 分布必须写成显式目标分布 \(Q(\delta\mid l,p,c)\)。support/admissibility 是 \((h,Q)\) 的属性，不是 activation 点自身的内禀真假标签。

### T4. Finite-sample conformal admissibility

在独立 fit/calibration split 与 exchangeability 条件下，对复合 nonconformity score 做一次整体 split-conformal calibration，得到 target-reference 条件下的有限样本错误率控制。

注意：当前三个 component ECDF 的几何平均不是自动有效的 conformal p-value。新实现必须先在 calibration samples 上计算同一复合 nonconformity，再对 query 做最终 rank calibration。

### T5. Evidence hierarchy / identification region

论文不再输出单一“valid/invalid”裁决，而报告三个互不替代的对象：

1. \(R\)：零阶 behavioral restoration；
2. \(S_Q\)：相对于明确目标分布 \(Q\) 的 conformal admissibility；
3. \(M_Q\)：允许 probe 下的 interventional response agreement。

机制结论的强度由可排除的 mechanism equivalence class 决定，而不是由三个数相乘得到一个总分。

## 实验主线

### E1. 已知真值 synthetic：先建立正反识别

在现有 dormant/gated-path synthetic 基础上构造四个象限：

- 高 R、高 S、高 M：真正机制恢复；
- 高 R、高 S、低 M：on-support restoration lie；
- 高 R、低 S、低 M：off-support shortcut；
- 低 R、高 S：自然但无效的 intervention。

必须跨独立模型 seed、独立数据 seed 和 probe seed。这里是 theorem-to-ground-truth 的核心验证，不是 smoke test。

### E2. IOI：把新实验从反例升级为主证据

保留 L4--L8、固定 IO position、所有 NMH 均在下游的 protocol。检验：

- restoration 在 L4--L8 仍高；
- NMH recovery 低；
- support overlap 正常；
- IRS 是否稳定识别 response-field mismatch；
- IRS 与 NMH 作为独立机制 witness 是否一致，而不被 layer/position 解释。

成功标准：在 exact site / prompt-matched 控制下，IRS 对 NMH mismatch 有稳定效应，并优于单一 overlap、activation norm 和未经配对的梯度基线。

### E3. 非 IOI 已知机制任务

优先 Greater-Than 或可完全验证的 compiled/synthetic transformer；但考虑最新工作指出 piecewise-affine compiled model 可能压低 interaction，最终必须同时包含一个真实 pretrained task。

目标是证明 IRS 不是 IOI/NMH 的同义改写。

### E4. 真实跨任务与跨模型

至少覆盖：

- IOI；
- Greater-Than 或 factual recall；
- 2--3 个模型家族，而不是只扩展参数规模；
- donor/reference/probe-distribution factorial；
- noising 与 denoising 的区分；
- 与最新 interaction、divergence、matched-donor 方法的公平对照。

### E5. 可证伪的失败条件

以下任一情况发生，当前 IRS 版本不能作为 oral 主方法：

1. IRS 主要由 activation distance 或 layer index解释；
2. 在 high-R/high-S/low-A 条件下，IRS 不能稳定识别 mismatch；
3. probe 轻微变化就翻转结论；
4. synthetic 正面条件下无法获得预期 coverage/power；
5. 只在 IOI 成立；
6. 理论只能重述 Taylor expansion，无法导出新的审计决策或可检验预测。

## 执行顺序

### P0：3 天内完成理论最小闭环

1. 写出 T1/T2 的正式 statement、assumptions 和 proof。
2. 定义 IRS estimator、probe distribution 和误差分解。
3. 实现真正的 composite split-conformal calibration。
4. 在现有 synthetic 和 GPT-2 L4--L8 数据上做最小可行 IRS 实验。

### P1：只有 P0 通过才扩 GPU

1. 3--5 个 synthetic model seeds；
2. GPT-2 / GPT-2 Medium；
3. 非 IOI task；
4. probe/reference factorial 与强 baseline。

### P2：主文重构

在 P0/P1 数字冻结前，不继续修补旧 abstract。新主文结构应是：

1. restoration 是零阶证据；
2. 不可识别性 theorem；
3. IRS + conformal admissibility；
4. known-ground-truth identification；
5. real-model restoration lies；
6. 与 divergence、interaction、causal abstraction 的边界。

## 是否需要 GPT Pro

目前不需要让 GPT Pro 替我们发明主线；检索和现有数据已经给出一条足够明确、可证伪且比旧主线更高的路线。

更合适的使用时点是：P0 theorem 和最小实验完成后，把完整 formal statement、反例、最新 prior-art matrix 一次性交给 GPT Pro 做 adversarial novelty review。它应扮演红队，不应在证据冻结前充当方向生成器。

## Oral 门槛

这条路线仍不能保证 oral，但它具备 oral 所需的结构：

- 一个领域级问题：何时 intervention effect 能支持 mechanism claim；
- 一个清晰的不可识别性结果；
- 一个正面的、带保证的识别/审计对象；
- 一个跨 synthetic ground truth 与真实 transformer 的惊讶性结果；
- 对 2025--2026 最新工作的直接统一和超越，而非换名复述。

最终判断标准不是“论文显得宏大”，而是：审稿人能否从理论和实验中得到一条过去没有、以后会实际改变 activation-patching 实践的规则。
