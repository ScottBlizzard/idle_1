# ICLR 2027 Idea Guide（大改版）：When Restoration Lies — 因果干预的"效果 ⊥ 有效性"两轴框架

**目标会议**：ICLR 2027
**定位**：机制解释可靠性 / causal intervention 方法论 + 一个反直觉的实证攻击
**资源假设**：2× A40 (48GB) + 8× 4090 (24GB)。算力不再是约束；它的作用是**多 seed 统计严谨度 + 全消融网格并行 + 解锁 7B/DAS scale 叙事**。
**与 v1 蓝图的根本区别**：v1 的灵魂是"我们提出 IVS 这个诊断分数"（增量、约 6 分）。本版把灵魂换成一个**反直觉的方法论攻击**——*behavioral restoration 不是机制正确的证据*；IVS 退居为"抓住这个攻击的仪器"。

---

## 0. 一句话版本

机制解释大量依赖 activation patching / interchange intervention 等因果干预，而领域默认：**只要干预能恢复目标行为（high behavioral effect / logit-diff restoration），它定位的 circuit 就可信**。我们要证明这是错的，而且是**系统性地、可预测地**错：

> **一次干预的"行为效果"和它的"有效性（latent 是否是模型自然可能产生的真实反事实状态）"是两根正交的轴。领域把它们当成一根。而恰恰是高效果、低有效性的干预，最容易让标准方法论自信地把因果归到错误的组件上。**

我们构造出这个"gotcha"（在已知真相的合成模型里，标准 patching 满分恢复行为却指错 circuit），用一个 intervention validity 仪器把它抓出来，并给出"在效果和有效性之间报告 trade-off"的修复/过滤协议。

推荐题目：

> **When Restoration Lies: Behavioral Effect Is Not Evidence of Mechanism in Causal Interpretability**

备用（偏方法）：

> **Validity-Aware Causal Interventions: Diagnosing When Restoration Implies the Wrong Mechanism**

---

## 1. 核心反直觉主张（论文的灵魂，放在最前面）

这一节是全篇的弹头，对应作者过往两篇中稿（OrthKD / OV-OrthKD）共享的"house style"。

### 1.1 我们沿用的反直觉模板

作者已验证两次的成功模板是：

> 把领域里一个**被当成单一指标**的东西，拆成两根**正交的轴**；证明领域的默认做法是把两轴合并；而这种合并**有害**——一个看似"好/强"的信号放错位置反而把结果带坏。

- OrthKD：拆 `teacher 质量` ⊥ `信号该作用的层级`。弱 teacher 的 logits 有毒、feature 有用。
- OV-OrthKD：`weak audio should help represent, not decide`；decision-level 转移反而更差。

**本论文是这套模板在 mechanistic interpretability 的第三次转写：**

| 过往套路 | 本论文对应物 |
|---|---|
| 被当成单一指标的东西 | **behavioral restoration / 干预效果** |
| 领域默认 = 强信号就该信 | "干预能恢复行为 → 它定位的 circuit 就对" |
| 拆出的第二根正交轴 | **intervention validity（latent 是否真实 / on-manifold）** |
| 反直觉弹头 | **效果最强的干预，可能恰恰最误导**（靠 invalid / dormant path 满分恢复 → 自信指错） |
| "in-domain 静默、OOD 才决定" | **只看 restoration 永远发现不了，只有 ground-truth / 查 latent 才暴露** |

### 1.2 一句话 claim

> Behavioral effect and intervention validity are distinct, separable axes that the field conflates. We show that high-effect, low-validity interventions systematically and predictably induce confident-but-wrong mechanistic conclusions, give a per-intervention validity instrument that catches them, and propose validity-aware reporting and repair that trade effect against validity rather than maximizing restoration blindly.

### 1.3 第二层反直觉（让它更难被解释掉）

validity 问题**恰好藏在大家平时只看的地方之外**：

- 你只看 restoration / logit-diff（headline 指标）→ 一切正常，发现不了。
- 只有当你（a）在合成里掌握真相、或（b）去查 latent 的 locus / 后续层轨迹 → 问题才暴露。

这和 OrthKD 的"orthogonality in-domain 静默、OOD 才决定性"是同构的。两层反直觉叠加，是这条线能冲高分的结构性来源。

---

## 2. 与已有工作的边界（确认 novelty，不是堆文献）

目的：说清楚我们和最近工作的差异**正好落在"第二根轴"上**。

### 2.1 必须正面对标

**Addressing Divergent Representations from Causal Interventions（ICLR 2026 Oral, openreview cZrTMqYVL6）**
- 已有：证明干预会产生 OOD / divergent latent；区分 harmless / pernicious divergence；用 CL-loss 变体把表示拉回自然分布。
- 我们的差异（关键）：
  - 它的贡献是"发现问题 + 一个训练期缓解"。我们的贡献是**把"效果 ⊥ 有效性"做成一个可证伪的实证攻击**：在已知真相的设定下，证明 low-validity 干预**导致错误的因果结论**，而不只是"表示发散"。
  - 它停在"divergence 存在且可缓解"；我们推进到"**divergence 何时把机制结论带偏、能否被 per-intervention 抓住、要不要修**"。
  - CL-loss 是训练 objective；我们的 validity 是 **per-intervention、training-free 的报告/审计维度**，可套在任何已有干预上。把 CL-loss 纳入 repair baseline。

**Towards Best Practices of Activation Patching（ICLR 2024, arXiv 2309.16042）**
- 已有：patching 的 metric / corruption / 超参如何影响 localization。
- 差异：他们关注"patching 方案如何影响结论"，**仍把 restoration 当 ground truth 代理**。我们恰恰攻击这个代理本身：restoration 高也可能是 invalid path 造成的假阳性。

**DAS / Finding Alignments（arXiv 2303.02536）+ Boundless DAS / Interpretability at Scale（NeurIPS 2023）**
- 已有：causal abstraction 框架，把高层因果变量对齐到分布式表示。
- 差异：我们不再找 alignment，而问 **alignment 搜出来的 interchange intervention 是否 valid**。DAS 既是高价值 baseline，也是 7B scale 实验的载体。

**Have Faith in Faithfulness / EAP-IG（COLM 2024, arXiv 2403.17806）**
- 已有：circuit 层面的 faithfulness（去掉 circuit 外是否保持行为）。
- 差异：他们的 faithfulness 是 circuit 级；我们的 validity 是 **intervention 级**（某次反事实 latent 是否仍在自然条件流形附近）。论文要明确：**validity 是 intervention-based faithfulness claim 的前置风险信号，但不等于 faithfulness。**

**Towards Automated Circuit Discovery / ACDC（NeurIPS 2023 Spotlight, openreview 89ia77nZ8u）**
- 差异：我们不发现新 circuit，而**审计 ACDC/EAP/EAP-IG 的每步 patching 是否产生 invalid 反事实**，把 validity 当质量控制。

### 2.2 支撑背景（不作主 baseline）

- ROME / causal tracing（NeurIPS 2022, openreview -h6WAS6eE4）：干预在 factual recall 的代表；小规模 sanity。
- Causal Mediation for Gender Bias（NeurIPS 2020）：mediation 背景。
- IOI（arXiv 2211.00593）、Greater-Than（NeurIPS 2023, p4PckNQR8k）：真实语言任务主战场。
- Causal Scrubbing（chanlawrence.me）：思想背景，不必完整复现。

---

## 3. 主线框架：effect ⊥ validity

### 3.1 形式化

第 `l` 层自然激活分布：

\[
H_l = \{h_l(x_i)\}_{i=1}^n
\]

一次干预产生的反事实 latent：

\[
\tilde{h}_l = I(h_l(x_{target}), h_l(x_{source}), c)
\]

两个轴：

- **Effect**：\(E(\tilde{h}_l)\)，干预对目标行为的恢复/改变（logit-diff restoration、interchange accuracy 等）。这是领域已经在用的。
- **Validity**：\(V(\tilde{h}_l)\)，\(\tilde{h}_l\) 是否落在**满足同一因果变量 / 任务条件 `c`** 的自然表示流形 \(\mathcal{M}_l(c)\) 附近，且后续层轨迹自然演化。

核心主张：\(E\) 和 \(V\) 近似正交，且 **高 \(E\) 低 \(V\) 区是危险区**——标准方法只看 \(E\)，于是在这个区里系统性出错。

### 3.2 为什么这不是"又一个 OOD 分数"

- 不是看 \(\tilde{h}_l\) 像不像**任意**自然激活，而是像不像**满足 `c` 的条件自然激活**（conditional manifold）。
- 它和**解释错误 / false causal claim 直接绑定**（第 5 节用 ground truth 证明），而普通 OOD 分数从不做这个绑定。
- 我们会用消融证明：朴素 Mahalanobis / energy / KDE 预测解释错误的能力，**显著弱于** conditional validity。

---

## 4. 贡献设计（按重要性重排）

> 顺序变了：先 gotcha（弹头），再仪器，再修复，最后 suite。v1 把仪器排第一是定位错误。

### 4.1 贡献一（核心）：The Restoration-Lies Phenomenon

在**已知真相**的合成模型里，干净地构造并证明：

> 存在高 behavioral restoration 的干预，它恢复行为靠的是一条 invalid / dormant 路径，因此标准 patching 会把因果**自信地归给错误组件**（false positive causal claim），而真实机制在别处。

这是全篇唯一靠"想"、AI 替不了的部分，也是分数天花板所在（见第 5 节构造）。

### 4.2 贡献二（仪器）：Intervention Validity Score（IVS）

一个 **per-intervention、training-free** 的诊断分数，用来在没有真相的真实任务里**预警**贡献一里那种危险干预。

| 组件 | 作用 | 必须 |
|---|---|---|
| Local kNN density | 干预点附近有无自然邻居 | 必须 |
| Local PCA reconstruction error | 是否落在局部线性流形 | 必须 |
| Mahalanobis / shrinkage covariance | 稳定层级 OOD 分数 | 必须 |
| Conditional causal-variable consistency | 是否落在目标因果变量条件分布 | 必须 |
| Path consistency | 后续层轨迹是否自然演化 | 强烈建议 |
| Dormant-pathway activation score | 是否点亮了本不该亮的隐藏路径 | 强烈建议 |
| Behavioral side-effect score | 是否影响非目标行为 | 强烈建议 |

\[
\text{IVS}(\tilde{h}_l) = \sigma\!\left(-\alpha D_{density} - \beta D_{manifold} - \gamma D_{path} - \delta D_{side}\right)
\]

定位：**IVS 是 risk indicator，不是真理判定器。** 它的价值由"能否预测贡献一的错误"来背书，而不是由分数好看来背书。

### 4.3 贡献三（修复，次要）：Natural-Manifold Constrained Repair + Validity-Aware Reporting

诊断之后给两条路，**但主张是 trade-off，不是"总能修好"**：

- **Repair**：把干预投影回局部 tangent manifold，或做约束优化
  \[
  \min_z\; L_{effect}(z, y_{target}) + \lambda D_{manifold}(z, \mathcal{M}_l(c)) + \mu D_{path}(z)
  \]
- **Filtering / Reporting**：不修，只拒绝或标注低 validity 的 causal claim（往往是更稳、更诚实的版本）。

| 方法 | 角色 |
|---|---|
| kNN natural projection | baseline 必做 |
| Local PCA tangent repair | 推荐主修复方法 |
| Constrained optimization | 增强版 |
| CL-loss extension（对标 Oral） | baseline / 对比，非主方法 |
| Layer-wise DAE | 可选增强 |
| Conditional normalizing flow | 仅可选 |

### 4.4 贡献四：Intervention Validity Diagnostic Suite + Reporting Checklist

小型诊断套件（不走 benchmark 赛道）+ 一份"以后因果干预论文应报告哪些 validity 诊断"的 checklist。后者是潜在的领域影响力来源。

---

## 5. 核心实验 E2：The Restoration-Lies Table（全篇引擎）

> 这一节单独拎出来，因为它就是论文的"那张表"，形状直接对标作者 OV-OrthKD 的 Table 3（role-swap 式反直觉消融）。E2 立得住，论文从 6 分变成有机会冲高；E2 立不住，整条线该换设定。

### 5.1 合成 gotcha 的构造（需 Phase-1 验证，但目标明确）

构造一个小 transformer（2–4 层），训练在一个**真相已知**的任务上，使其内部存在两条都能产出正确输出的路径：

- **Primary path `P_nat`**：模型在自然分布上**实际使用**的机制 = ground-truth circuit。
- **Dormant path `P_dorm`**：一条冗余/捷径路径，在自然 forward 中被某个 context gate 压制、几乎不激活；但**可以被一个 off-manifold 的 latent 状态触发**。

设计要点：

1. 任务标签主要由特征 A 决定，模型学到用 `P_nat`（读 A）。
2. 加一个冗余特征 B + 一个 gate token，使得 B→输出的通路 `P_dorm` 在自然数据上被 gate 关闭。
3. 构造一次 activation patching（例如从 gate 打开的 source 注入、或中层 patch），把 hidden state 推到 `P_dorm` 的吸引域 → **logit-diff 满分恢复**。
4. 但被 credit 的组件属于 `P_dorm`，**与真实机制 `P_nat` 无关** → false positive causal claim。

因为是合成的，你**确切知道** `P_nat` 是真相、`P_dorm` 是 artifact，于是能算 false-positive rate；并验证 **IVS 把这次 patch 标为 low-validity**，从而预测"这个结论不可信"。

### 5.2 "那张表"的形状（核心交付物）

| 干预 | behavioral restoration | IVS（validity） | 定位到的 circuit | 对/错 |
|---|---:|---:|---|:---:|
| Patch @ site-X（触发 dormant） | **高（~满分）** | **低** | `P_dorm` 组件 | **错（false positive）** |
| Patch @ site-Y（沿自然路径） | 中-高 | 高 | `P_nat` 组件 | 对 |
| 标准协议（只看 restoration）→ 选了第一行 | — | — | — | **被骗** |
| Validity-aware（看两轴）→ 选了第二行 | — | — | — | 救回 |

这张表 = 论文的"脊背发凉"时刻。再配 role-swap / corruption / 多 seed 误差棒做成 hard-to-vary。

### 5.3 指标

- false causal claim rate（核心）。
- circuit overlap / faithfulness / completeness / minimality vs ground truth。
- IVS-error correlation（带多 seed 置信区间）。

**强结论**：IVS 不是好看的 OOD 指标，而是**能预测机制解释是否出错**的风险信号。

---

## 6. 其余实验

### E1：Intervention validity landscape（先导）
不同方法（activation patching / mean / resample ablation / DAS / path patching）× 层 × 任务的 validity 地图。
图：layer×method heatmap、restoration-vs-IVS scatter。
结论：**高效果不保证高有效性**，先把两轴的正交性看出来。

### E3：Repair 改善 validity-effect frontier
对 invalid 干预做 kNN / Local PCA / 约束优化修复，对比 CL-loss、no-repair。
指标：IVS、causal effect preservation、downstream side effect、**validity-effect Pareto AUC**。
结论：不是盲目降 divergence，而是在效果与有效性间建立更好的 Pareto。

### E4：真实语言 circuit case study（IOI / Greater-Than / induction）
原 patching 结论在 high-IVS vs low-IVS 子集里是否稳定？哪些 head/layer 的结论最依赖 invalid 干预？repair 后定位是否更稳？
（真实任务定位为 case study，不把全部 claim 压上去——见 9.4 容错。）

### E5：与自动 circuit discovery 兼容（ACDC/EAP/EAP-IG）
对每步 patching/attribution 算 IVS，过滤低 IVS 操作。
结论：low-IVS 操作产生 less faithful circuit；validity-aware filtering 能减少明显错误。

### E6：IVS 鲁棒性与消融（防"arbitrary score"质疑）
去 density / 去 path consistency / 去 conditional label；kNN 数、PCA rank、层数、激活数据量；**并与朴素 Mahalanobis/energy/KDE 对比预测解释错误的能力**——证明 conditional validity 的必要性。

---

## 7. 资源与并行实验编排（2× A40 + 8× 4090）

**心态先摆正**：算力不是瓶颈，GPT-2 small 上 patching 是推理级开销。这套卡的真实价值是三件事。

### 7.1 8× 4090：统计严谨度 + 全消融网格
- E2 / E6 是 embarrassingly parallel 的笛卡尔积（method × layer × seed × task × kNN × PCA rank）。一晚扫满，**给所有相关性结果配多 seed 误差棒和显著性检验**——这是 Oral 评审最吃的，单卡串行你会偷懒砍掉一半。
- surprise 常藏在大 sweep 的尾部/离群点。并行 = 多买"撞见异常"的彩票。

### 7.2 2× A40 (48GB)：解锁 scale 叙事
- 稳跑 7B（Alpaca/Llama 级）推理 + activation 缓存，把 v1 里仅 10% 权重的 **DAS / Boundless DAS validity 审计**升级成正式实验：**"validity 在 7B DAS 场景下是否依然预测错误？"** 直接回应 Oral 那篇的 setting，差异化大增。

### 7.3 不要做的事
- **不要因为有卡就上大模型铺摊子。** 模型越大、摊子越宽，越稀释主 claim。主线锁在"合成 gotcha + GPT-2 small"，7B 只作 scale 验证的一条腿。

### 7.4 证据权重（重排）

| 实验 | 权重 | 目的 |
|---|---:|---|
| E2 合成 restoration-lies | **35%** | 弹头：证明 low-validity → false causal claim |
| E4 GPT-2 IOI/Greater-Than/induction | 25% | 真实 relevance |
| E3 repair / Pareto | 15% | 不只诊断，能行动 |
| E5 ACDC/EAP 兼容 | 10% | 工具价值 |
| 7B DAS scale（A40） | 10% | scale 差异化、对接 Oral |
| E1 landscape + E6 ablation | 5% | 必要支撑 |

---

## 8. Baseline

**必做**：vanilla activation patching；mean / zero / resample ablation；path patching；DAS / interchange；**CL-loss / Divergent 原文修复（必须正面对比）**；kNN natural projection；no-repair validity filter。

**选择性**：ACDC（做 circuit discovery 时）；EAP/EAP-IG（做 edge attribution 时）；causal scrubbing（思想对比）；OOD 检测器（energy/Mahalanobis/KDE，作 IVS 组件消融，**非主贡献**）。

---

## 9. 容错空间（每条都已 de-risk，灵魂不塌）

### 9.1 如果合成 gotcha 造不干净（最大风险）
- 这是 Phase-1 的 Go/No-Go 核心。先用最小构造验证"invalid 干预能否满分恢复又指错"。
- 若难造：退到 **failure taxonomy** —— 给出 invalid 干预导致解释失败的几类典型模式 + 真实任务里的 instability 证据，主 claim 改为"restoration 不充分 + 给出风险分类"，仍成立。

### 9.2 如果 repair 洗掉 causal effect
- 主张本就是 **Pareto / filtering**，不是"总能修"。改成"报告 trade-off + 在不可修区做诊断贡献"。

### 9.3 如果 IVS-error 相关性不够强
- 拆子指标，找真正预测错误的项（往往是 path consistency / dormant-pathway / side-effect）。
- 区分 harmless vs pernicious invalidity（与 Oral 一致）。
- claim 降为"risk indicator, not deterministic predictor"。

### 9.4 如果真实任务不如合成明显
- 真实任务作 case study；用 known-circuit 的 **stability** 而非 absolute correctness；多 prompt template / corruption 展示 validity 对超参选择的解释力。

### 9.5 如果 reviewer 说 "naturalness ≠ faithfulness"（必问）
- 承认 naturalness 不充分；强调 low validity 是 unfaithful claim 的 **风险因子**。
- 标准表述：*Intervention validity is a necessary diagnostic dimension for causal interpretability, not a replacement for behavioral faithfulness tests.*

### 9.6 如果 CL-loss baseline 很强
- 不硬拼单点。强调我们 **training-free / per-intervention / 通用**；CL-loss 是训练 objective，IVS 是评估/报告标准，且能**解释 CL-loss 何时有用**。

---

## 10. 完成清单

### 10.1 最小强版本
1. effect ⊥ validity 的形式化 + IVS 子指标。
2. **合成 restoration-lies gotcha + "那张表"（E2）。**
3. 证明 IVS 与 false causal claim 显著相关（多 seed）。
4. Local PCA / tangent repair。
5. GPT-2 small IOI + Greater-Than（E4）。
6. 对比 vanilla patching / DAS / CL-loss / kNN repair。
7. validity-effect Pareto plots。
8. E6 消融（含"conditional vs 朴素 OOD"对比）。

### 10.2 Oral 潜力版本（额外）
1. ACDC/EAP/EAP-IG 兼容（E5）。
2. **7B DAS scale 验证（A40）。**
3. path consistency / dormant-pathway 深入分析。
4. 理论 proposition：在某些 hidden-pathway 构造下，behavioral restoration 必可由 invalid 干预触发 → 单看 restoration 必产生 false causal claim。
5. 发布 diagnostic suite + reporting checklist。

---

## 11. 写作框架

### 11.1 摘要骨架（reframed）
> Causal interventions such as activation patching and interchange interventions are central to mechanistic interpretability, where an intervention that restores the target behavior is routinely taken as evidence that the patched component implements the mechanism. We show this inference is systematically unsafe: behavioral effect and intervention validity—whether the intervened latent state is a realistic counterfactual on the model's natural conditional manifold—are distinct, separable axes. In controlled circuits with known ground truth, we construct high-restoration interventions that recover behavior through an invalid, dormant pathway and thereby drive standard methodology to confidently localize the wrong circuit. We introduce a per-intervention validity instrument that predicts these false causal claims, show the effect persists in GPT-2 language-model circuits and DAS at scale, and propose validity-aware reporting and natural-manifold repair that trade effect against validity rather than maximizing restoration blindly. We argue intervention validity should be reported alongside behavioral effect.

### 11.2 Introduction 结构
1. 机制解释依赖因果干预，并默认 restoration ⇒ 正确机制。
2. 这个默认把两根轴合并了。
3. 我们拆开：effect ⊥ validity。
4. 反直觉弹头：高效果 + 低有效性 = 最危险（restoration lies）。
5. 仪器（IVS）预测错误；真实任务 + 7B 验证；repair/reporting。
6. 贡献列表（gotcha 第一，分数第二）。

### 11.3 Related Work
causal interventions / activation patching best practices / causal abstraction-DAS / automated circuit discovery / faithfulness & causal scrubbing / OOD & manifold（强调我们不是普通 OOD 论文，差异在"绑定解释错误的 conditional validity"）。

### 11.4 Method
problem setup → effect/validity 两轴 → IVS estimator → manifold repair → validity-effect frontier。

### 11.5 Reviewer 质疑与回应
| 质疑 | 回应 |
|---|---|
| naturalness ≠ faithfulness | 不声称充分，是风险诊断维度 |
| IVS 就是普通 OOD | conditional validity 且与 false causal claim 绑定，消融证明优于朴素 OOD |
| repair 是 heuristic | 主张是 Pareto + 多 estimator 消融 |
| 合成太多 | GPT-2 IOI/Greater-Than + 7B DAS 撑真实/scale |
| baseline 不够 | vanilla/DAS/CL-loss 必做，ACDC/EAP 可选 |
| 只是复现 Oral 的 divergence | 我们证明 divergence **导致错误结论** 且 per-intervention 可抓，CL-loss 仅作 baseline |

---

## 12. 实施路线

### Phase 1（两周 feasibility，决定生死）
- 实现 IVS 基础三件套：kNN density、Local PCA error、Mahalanobis。
- **核心：造出最小 restoration-lies gotcha**——确认"invalid 干预满分恢复 + 指错 circuit"现象真实可构造。
- 跑 GPT-2 small IOI 的 vanilla patching，出第一版 restoration-vs-IVS scatter。
- **Go/No-Go**：合成上能否干净分出 valid/invalid 且 invalid 真的指错？能 → 全速；勉强 → 调 hidden-pathway 构造再判；不能 → 退 9.1 的 taxonomy 版本。

### Phase 2（主实验）
- Local PCA repair + CL-loss / kNN baseline；IOI + Greater-Than；Pareto frontier；多 seed 全消融（8×4090 并行）。

### Phase 3（强版本增强）
- ACDC/EAP 兼容；7B DAS scale（A40）；path consistency；diagnostic suite + checklist 开源。

---

## 13. 最终判断

这条线的价值**不在"提出一个分数"，在于一个反直觉的方法论攻击**：*restoration 会说谎*。它符合作者已验证两次的 house style（拆正交两轴 + 高分信号放错位置反而有害），与 ICLR 2026 Oral 自然衔接但不是 follow-up（我们证明 divergence 会**导致错误结论**），且有望给领域立一个 reporting standard。

- 算力已非约束；卡用于多 seed 严谨度、全消融、7B scale。
- 成败 90% 压在 **Phase-1 的合成 gotcha 能否干净构造**上。
- 措辞红线：不说 validity = faithfulness；不说 repair 总能恢复正确解释；只说 validity 是被忽略的必要诊断轴。

**推荐：作为主线全速推进，第一步就攻 Phase-1 的 restoration-lies 构造。**

---

## 14. 参考文献清单

1. Grant et al. 2026. Addressing Divergent Representations from Causal Interventions on Neural Networks. ICLR 2026 Oral. https://openreview.net/forum?id=cZrTMqYVL6
2. Zhang & Nanda. 2024. Towards Best Practices of Activation Patching in Language Models. ICLR 2024. https://arxiv.org/abs/2309.16042
3. Geiger et al. 2024. Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. https://arxiv.org/abs/2303.02536
4. Wu et al. 2023. Interpretability at Scale: Identifying Causal Mechanisms in Alpaca. NeurIPS 2023. https://papers.nips.cc/paper_files/paper/2023/hash/f6a8b109d4d4fd64c75e94aaf85d9697-Abstract-Conference.html
5. Hanna et al. 2024. Have Faith in Faithfulness. COLM 2024. https://arxiv.org/abs/2403.17806
6. Conmy et al. 2023. Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS 2023 Spotlight. https://openreview.net/forum?id=89ia77nZ8u
7. Meng et al. 2022. Locating and Editing Factual Associations in GPT. NeurIPS 2022. https://openreview.net/forum?id=-h6WAS6eE4
8. Vig et al. 2020. Investigating Gender Bias in Language Models Using Causal Mediation Analysis. NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html
9. Wang et al. 2022. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. https://arxiv.org/abs/2211.00593
10. Hanna et al. 2023. How does GPT-2 compute greater-than? NeurIPS 2023. https://openreview.net/forum?id=p4PckNQR8k
11. Chan et al. 2022. Causal Scrubbing. https://chanlawrence.me/publication/chan-2022-causal/
