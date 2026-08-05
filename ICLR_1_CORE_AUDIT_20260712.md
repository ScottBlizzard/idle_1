# ICLR_1 核心技术审计与整改门槛

> 审计日期：2026-07-12  
> 审计对象：`D:\ICLR_1` 当前工作树中的论文、代码、实验记录与关键输出  
> 当前论文：`iclr2026/paper.tex` / `iclr2026/paper.pdf`  
> 用途：投稿前技术审计、实验重设计与后续复审  
> 性质：审稿人式风险评估，不是编辑决定，也不等于已经证明论文结论为假

## 1. 执行摘要

ICLR_1 仍然是当前五个项目中**研究上限最高**的项目。问题重要，`restoration / validity / mechanism alignment`（R/V/A）三轴框架具有明显的概念吸引力；如果关键验证通过，它仍有机会恢复为高分 ICLR 稿件，甚至具有 oral 讨论空间。

但当前版本存在三项会直接影响中心论证的高优先级风险：

1. 当前 IVS 直接测量的是 clean-source activation 相对于 corrupt observational reference 的条件重叠；这一对象尚未被证明等于论文所称的 counterfactual intervention validity。
2. 核心 IOI reference 每模板最多只有约 20 个唯一 corrupt prompts，但设置了 `N_REF=150` 与 `PROJ_RANK=32`；低秩 reference、训练内 reconstruction calibration 和 `1e-6` scale floor 可能显著放大 headline 的极端 gap。
3. NMH “独立验证”与 layer/position/time ordering 混杂：许多晚层 patch 发生在相关 NMH attention 已经计算之后，因此高 restoration、低 NMH recovery 不足以单独证明 IVS-invalidity 导致 mechanism bypass。

因此必须区分两种排序：

- **研究上限 / 潜在影响力：ICLR_1 可排第一。**
- **当前版本的投稿稳健度：需要 major redesign，暂不宜按 oral-ready 稿件推进。**

当前最合理的姿态是：**Weak Reject / Major Redesign before submission**。这不是 idea-level rejection，而是要求先建立测量对象、排除低秩数值退化，并完成真正独立的机制验证。

---

## 2. 审计范围与边界

### 2.1 已检查材料

- `iclr2026/paper.tex` 与 18 页 `paper.pdf`
- `theory_proofs.md`
- `experiment_report.md`
- `analysis/phase12_claim_evidence_map.md`
- `analysis/phase12_adversarial_review.md`
- `analysis/local_theory_audit.md`
- 关键 IVS、IOI、NMH、Pythia、baseline 和 figure-generation 实现
- 关键 JSON 输出与 Git/复现状态
- PDF 全页渲染与视觉检查

### 2.2 未完成事项

- 未重新运行长时间多模型 GPU 实验。
- 未进行全网 prior-art 检索；新颖性边界以本地论文和文献材料为限。
- 未在独立环境中完成第三方端到端复现。

### 2.3 结论使用规则

本文把发现分为：

- **Supported**：当前材料直接支持。
- **Weak / confounded**：存在信号，但替代解释尚未排除。
- **Not established**：当前证据不足以支撑论文中的强表述。
- **Not assessable**：缺少必要实验或独立材料。

---

## 3. 当前仍然成立、应当保留的结果

即使最严格地解释本次审计，以下内容仍具有明确价值。

### 3.1 晚层行为恢复不等于早期机制恢复

论文报告，GPT-2 的晚层/last-token patches 可以恢复约 92% 的行为输出，但 NMH attention recovery 仅约 4.1%。这一描述性结果支持：

> 输出恢复本身不足以证明目标内部机制已经恢复。

这是论文最稳固、最值得保留的 cautionary result。当前审计质疑的是将其进一步归因于 IVS-defined invalidity，而不是质疑晚层 answer injection / answer smuggling 现象本身。

证据位置：

- `iclr2026/paper.tex:131-146`
- `iclr2026/figures/fig4_answer_smuggling.pdf`

### 3.2 Reference distribution 的选择会决定诊断结果

同 site/context reference、mixed-site reference 和 wrong-position reference 会产生完全不同甚至反转的结果。这充分支持：

> 任何 activation validity / overlap 诊断都必须明确 reference distribution。

它尚不能证明 corrupt-condition reference 就是规范正确的 causal-validity reference，但已经证明“reference choice 不是实现细节”。

证据位置：

- `iclr2026/paper.tex:575-618`
- `analysis/conditionality_failure_cases.md`

### 3.3 R、分布重叠指标和机制 readout 可以经验性分离

kNN projection 可以提高 IVS，而 NMH recovery 基本不变。这说明不同指标并不等价。当前安全表述应是：

> Behavioral restoration、target-reference overlap 和所选 mechanism readout 是不同证据轴。

在 counterfactual validity 的目标分布建立前，不宜把第二轴直接等同于一般性的 causal intervention validity。

证据位置：

- `iclr2026/paper.tex:334-340`
- `analysis/repair_rva_upgrade.md`
- `outputs/exp_phase11_repair_fiber_gpt2.json`

### 3.4 Pythia-6.9B 的旧 chunking 问题已被识别

项目正确追踪并修复了旧 chunked patching 的对齐错误；新的 full-batch runs 将 Pythia-6.9B 作为 clean-audit / protocol-boundary case，而不是继续选择性报告旧的高 lie rate。这体现了较好的负结果与实现审计纪律。

证据位置：

- `src/exp_pythia69b_gap_verify.py:204-210`
- `analysis/local_completion_status.md:109-114`

---

## 4. 核心问题一：IVS 的 estimand / construct validity 尚未建立

### 4.1 论文与代码实际测量什么

论文将 IVS reference 定义为 corruption condition 下同 layer、position 的自然 activations：

- `iclr2026/paper.tex:256-265`

实际代码路径是：

1. 使用 corrupt activations 建 reference：`src/gpt2m_ioi.py:79`
2. 缓存 clean activations：`src/gpt2m_ioi.py:80,89`
3. 把 clean activation 注入 corrupt run：`src/gpt2m_ioi.py:100-101`
4. 用 corrupt reference 给 clean activation 打分：`src/gpt2m_ioi.py:105`

因此直接估计的对象更接近：

\[
\operatorname{Overlap}\left(H_{\mathrm{clean}},
P(H\mid X_{\mathrm{corrupt}})\right).
\]

### 4.2 为什么它不自动等于 counterfactual validity

标准 activation patching 本来就在执行近似的 `do(H := H_clean)`。如果 clean/corrupt 改变了目标语义变量，一个合法 source-counterfactual activation 不必属于 observational `P(H | X_corrupt)`。

更接近论文强主张的目标可能是：

\[
P\bigl(H\mid do(C=C_{\mathrm{clean}}),
\text{nuisance context fixed}\bigr),
\]

或经过语义匹配的 counterfactual reference distribution。当前项目没有证明 corrupt observational reference 与这些对象等价。

### 4.3 项目自身暴露出的边界

本地实验记录显示，Greater-Than、Gendered Pronoun、ROME last-position 等会因 clean/corrupt 改变 token 或语义条件而被整体判为 off-manifold。这既可能说明这些 intervention 不可信，也可能说明当前 reference 把预期的 counterfactual shift 当作 invalidity。

证据位置：

- `experiment_report.md:234-261`
- `experiment_report.md:1442-1449`

这些失败任务应作为主文中的 construct-validity boundary，而不应只留在实验账本中。

### 4.4 对论文主张的影响

当前可保留：

- `IVS measures conditional target-reference overlap.`
- `Reference choice materially changes the diagnostic.`
- `Low target-overlap patches may warrant independent mechanism checks.`

当前尚未建立：

- `IVS directly measures general causal intervention validity.`
- `Low IVS is itself sufficient evidence that a patch is causally invalid.`
- `Positivity violation under P(H|corrupt) is equivalent to invalid counterfactual intervention.`

### 4.5 必须补做的验证

同一批 patch 至少比较四种 reference：

1. corrupt observational reference
2. clean/source reference
3. clean-corrupt mixture reference
4. matched semantic-counterfactual / intervention-conditioned reference

必须在查看结论前冻结：reference 构造、conditioning variables、阈值、评估 site 和机制 ground truth。

#### 建议决策门槛

- 如果只有 corrupt reference 能产生 lie separation，而 matched counterfactual reference 将这些 patches 判为自然，则论文必须把 `validity` 降级为 `target-context overlap`。
- 如果多种合理 reference 均能识别同一批、并由独立机制证据确认的 patches，则 causal-validity 解释得到显著加强。

---

## 5. 核心问题二：低多样性 reference 与 PCA/scale 退化

### 5.1 具体实现事实

核心 multi-template IOI 脚本设置：

- `N_REF = 150`
- `PROJ_RANK = 32`
- 候选名字仅 20 个

证据：`src/exp_multi_template.py:25-37`

clean prompt 使用不同的 `IO` 与 `S`，而 corrupt prompt 构造为 `S,S`：

- 抽取 `IO,S`：`src/exp_multi_template.py:67-68`
- corrupt string 只依赖 `S`：`src/exp_multi_template.py:69-70`

因此每个模板最多只有约 20 个唯一 corrupt prompts。重复采样到 150 条并不会增加 reference support；中心化 activation matrix 的秩上限约为 19，却请求 PCA rank 32。

### 5.2 标定如何放大差异

`FastSiteReference` 使用 reference 自身完成 reconstruction-error calibration，并将 scale 下限设为 `1e-6`：

- `src/validity_fast.py:78-93`

reference 上的 reconstruction error 因低秩/训练内重构接近零，而 clean query 在未覆盖方向上的较小残差可以除以极小 scale，形成百万量级 z-score。

现有输出与该机制一致：

- valid `recon_z` 约为普通数量级
- invalid `recon_z` 可达约 `1e6`
- kNN / Mahalanobis 未表现出同等数量级的分离

证据：`outputs/exp_raw_zscore_fresh_gpt2.json`

### 5.3 对 headline 的影响

在完成重新标定前，以下主张属于 Not established：

- `10^12` gap 是自然 activation manifold 的真实几何性质。
- PCA95 得到的约 17 维是独立估计的 intrinsic dimension。
- 极端 gap 证明了接近完美的普适 invalidity boundary。

尤其需要注意：约 17 的 PCA95 维数接近 20 个唯一 corrupt prompts 的秩上限 19，可能主要反映 prompt-support 设计，而非自然流形的独立几何性质。

### 5.4 必须补做的验证

1. 每模板生成数百至数千个真正唯一、语义多样的 prompts。
2. 明确报告唯一 prompt 数、activation matrix rank、有效奇异值谱和 condition number。
3. 强制 `PCA rank <= effective training rank`，并截断近零奇异值。
4. 使用独立 split 拟合 PCA/reference normalization 和评估 query，即 cross-fitted calibration。
5. 同时报告 raw kNN、raw reconstruction、raw Mahalanobis、标准化分量和 composite IVS。
6. 对 scale floor 做预注册敏感性分析，避免由一个数值下限决定 headline gap。
7. 用 held-out same-distribution queries 校准假阳性率，而不是只在 reference self-score 上校准。

#### 建议决策门槛

- 如果扩大唯一 support、cross-fit 与截断零奇异值后，极端 gap 消失但机制 bypass 仍存在，则删除数量级和高维几何 headline，保留机制 cautionary result。
- 如果定性分类、raw component separation 与独立机制证据均稳定，则可恢复较强的 validity-diagnostic 主张。

---

## 6. 核心问题三：NMH 独立验证存在时间和位置混杂

### 6.1 当前验证设计

论文以 Name Mover Head attention recovery 作为独立机制标签。实现中使用固定 NMH heads：

- `src/exp_nmh_ground_truth.py:42-44`

在任意 patch site 运行 patched forward pass 后读取这些 heads：

- `src/exp_nmh_ground_truth.py:274-278`

### 6.2 混杂来源

许多低 IVS、高 restoration patches 集中在晚层/last-token。对于发生在相关 NMH attention 之后的 residual patch，它在时间上无法回溯恢复已经计算完的 attention pattern。

项目自身 timing scan 也显示：

- 中前层可以有较高 NMH alignment、较低 restoration
- L10/L11 可以有高 restoration、近零 NMH alignment

证据：`experiment_report.md:1324-1335`

因此 AUROC 可能部分反映：

> IVS 能识别 late/last sites，而 late patch 按计算顺序不可能恢复 earlier-head attention。

这不足以单独证明：

> invalidity 导致了 mechanism bypass。

### 6.3 当前安全结论

Supported：

- 晚层 patch 能在不恢复较早 NMH attention 的情况下恢复输出。
- Restoration-only ranking 会包含 late answer-injection sites。

Weak / confounded：

- IVS-invalidity 是 NMH bypass 的原因。
- IVS 对 NMH label 的 AUROC 是完全独立于 layer/position 的机制验证。

### 6.4 必须补做的验证

1. 在同一 layer、同一 position、相近 restoration 范围内比较高/低 IVS patches。
2. 使用 patch 之后仍可能受影响的 downstream mechanism readout。
3. 分层报告 early/mid/late 与 IO/last-position 内的 AUROC，而不是只报告 pooled AUROC。
4. 对 layer、position、restoration 做匹配、条件回归或分层置换检验。
5. 恢复或重建 known-ground-truth synthetic circuit / dormant-path experiment，使正确机制完全可知。

#### 建议决策门槛

- 如果 within-layer/position 对照中 IVS 不再预测 mechanism bypass，则删除“独立验证 IVS-invalidity”的强表述，保留 late-injection caution。
- 如果匹配后仍稳定关联，并在 known-ground-truth circuit 上识别错误机制结论，则中心因果证据显著增强。

---

## 7. 其他 Major Issues

### 7.1 Baseline 描述与实际实现不一致

论文将 MLP 失败解释为 `N=71, d=768` 的高维小样本问题，并给出错误的 `d/N=54.9`；实际 `768/71≈10.8`。更重要的是，主 baseline 实现并非输入 768 维 activation，而是输入三个 IVS component z-scores：

- `src/exp_complex_baselines.py:361-372`

这三个特征未充分标准化，而 `recon_z` 可达到百万量级。MLP AUROC 极低可能主要是优化/缩放失败，不能支持“训练式 detector 在高维小样本下失效”的解释。

整改要求：

- 更正文中的维度与比值。
- 分别实现 activation-space baseline 和 3-component baseline。
- 使用统一 train/validation/test split、标准化、调参预算和重复种子。
- 不再用当前 MLP 结果支持“training-free 必要性”。

### 7.2 理论命题的假设不足

当前几何命题仍偏直觉性：

- 仅假设函数 Lipschitz，却使用需要正则值、光滑性与非退化梯度的 implicit-function / level-set 论证。
- `random point on the level set` 没有定义概率测度；level set 可能非紧或具有复杂几何。
- 仅靠 manifold diameter 不足以得到所述 tube-probability bound，常数还依赖 reach、曲率、体积和横截性。
- PCA95 不能直接等同于 intrinsic/topological dimension。
- positivity 类比需要区分“超出 target observational support 的外推风险”和“无法从观测数据识别 treatment effect”。

整改要求：

- 将主文理论定位为 geometric intuition / sufficient-condition sketch，或补齐正则值、紧致性、测度和几何常数假设。
- 不用 toy scaling 的极小数量级作为实证结论。
- 让理论预测对应可以被直接证伪的实验，而不是只解释已有 gap。

### 7.3 统计单位与不确定性表述偏乐观

需要明确：

- 所谓多个 `seed` 多数是固定 pretrained model 上重新抽取 prompts，不是独立模型训练种子。
- 五个 templates 结构相近。
- 每个 site 的统计量已对 prompts 聚合。
- reference 实际唯一 prompt 数远小于标称 `N_REF`。
- 现有极窄 bootstrap CI 主要描述固定 prompt universe 下 positional pattern 的稳定性，不代表跨任务、跨语义分布或模型训练的不确定性。

整改要求：

- 把 model、template、prompt、site 的层级写清楚。
- 使用与目标外推层级一致的 hierarchical bootstrap 或 cluster-robust 分析。
- 避免把 prompt resampling 称为模型级 replication。

### 7.4 跨任务证据存在选择风险

IOI 跨模型实验共享相同 corruption/reference construction，因此不能排除共同设计混杂。ROME 主要仍依赖 IVS-defined labels；induction 只有较小 negative control。Greater-Than、Gendered Pronoun 等 distribution-shifting tasks 的失败没有在主文充分呈现。

整改要求：

- 把失败任务作为 reference-estimand 边界公开报告。
- 至少选择一个非 IOI 任务完成独立机制 ground truth，而不仅是再次计算 IVS。
- 区分“跨模型复制同一协议”和“跨任务验证 construct validity”。

---

## 8. 可复现性与版本一致性问题

### 8.1 当前发现

- 缺少统一运行入口、环境锁文件和项目级自动测试。
- Git 仅有较早的 blueprint commit，论文、`src/`、`outputs/` 和分析大多未形成可复现快照。
- 主生成脚本仍保留 `IVS_THRESHOLD=0.3`，而论文和部分手工更新后的结果使用 0.01：`src/exp_multi_template.py:25-31`。
- Figure 5 的部分数值由代码硬编码，而不是从 canonical JSON 自动生成：`src/_gen_paper_figures.py:339-346,378-381`。
- `_knn_dist` 对 calibration 和外部 query 都跳过最小距离；对 reference self-score 合理，但对 held-out query 可能错误丢弃真实最近邻：`src/validity_fast.py:52-63`。
- 主表混用了 canonical site count 与 bootstrap-config rate，部分百分比不能由同一分母直接复算。

### 8.2 投稿前最低要求

1. 建立单一 canonical config，冻结 IVS 阈值、reference、PCA、scale、prompt 和 seed。
2. 所有图表从 canonical JSON/CSV 自动生成，禁止人工硬编码 headline 数字。
3. 增加 unit tests：unique-prompt count、effective rank、cross-fit split、kNN self/query 行为、threshold consistency、table reproduction。
4. 提供环境锁文件和从原始输出重建全部表图的一键脚本。
5. 将核心代码、配置、原始/聚合结果和论文纳入同一 Git snapshot。

---

## 9. 论文表述与版面问题

### 9.1 优点

- 标题有吸引力，R/V/A 主线容易记忆。
- 主图总体清晰，没有裁切、字体缺失、重叠或未解析引用。
- 问题的重要性和实践影响表达充分。

### 9.2 需要修复

- 摘要信息过密，并把 `IVS-invalid` 与一般性的 `causally invalid` 交替使用。
- `hyperref` 未隐藏链接边框，PDF 中引用和交叉引用存在明显彩色框。
- 第 9 页与部分附录页较稀疏。
- 当前目录名和样式仍沿用旧 `iclr2026` 模板；正式投稿前需核对并统一目标会议模板。
- 必须在主文显著位置区分：behavioral effect、target-reference overlap、counterfactual validity、mechanism alignment。

---

## 10. Claim status matrix

| 当前主张 | 审计状态 | 当前安全表述 |
|---|---|---|
| Behavioral restoration 不足以证明目标机制恢复 | Supported within tested IOI setting | 晚层 patch 可恢复输出而不恢复所选早期机制 readout |
| Reference distribution 必须条件化并明确报告 | Supported | 不同 reference 会反转或消除诊断结果 |
| IVS 直接测量 causal intervention validity | Not established | IVS 测量相对于指定 target reference 的条件重叠 |
| 22–42% 的高恢复 patches 是 causally invalid | Weak / self-defined | 22–42% 在当前 corrupt-reference IVS 下低重叠 |
| `10^12` gap 是自然流形的真实几何边界 | Not established | 当前实现产生极端 composite separation，需排除低秩和 scale 退化 |
| IVS independently predicts mechanism bypass | Confounded | pooled IVS 与 NMH readout 相关，但 layer/position/time ordering 尚未排除 |
| kNN repair 证明 V 不蕴含 A | Weak / alternative explanation | projection 提高 target-overlap score，但未恢复 NMH readout |
| Zero ablation 在 GPT-2 IOI 中 invalid | Supported only relative to current reference | zero vector 不属于当前 GPT-2 IOI corrupt-condition reference support |
| 存在 universal layer-wise phase transition | Not established as universal | 多个 lie-bearing models 在相同协议下呈现 depth concentration |
| Pythia-6.9B full-batch 是 clean audit | Supported for tested protocol | 两个 reference sizes 下未检测到低-IVS高恢复 sites |

---

## 11. 优先整改计划

### P0：决定论文是否成立的实验

1. **Reference estimand comparison**：corrupt / clean / mixture / semantic-counterfactual。
2. **Unique-support rerun**：数百至数千唯一 prompts，effective-rank audit，PCA cross-fit。
3. **Independent mechanism validation**：within-layer/position matching，加 known-ground-truth circuit。
4. **Fair baseline rerun**：正确维度、标准化、相同 split 和相同调参预算。

P0 完成前，不建议继续扩展更多模型。

### P1：强 ICLR 稿件需要的升级

1. 在至少一个非 IOI 任务上验证 construct，而不是仅复用 IVS label。
2. 重写或降级几何理论，确保假设与结论一致。
3. 使用与外推目标一致的分层统计单位。
4. 在主文公开失败任务和 reference boundary。

### P2：投稿封装

1. 统一代码、阈值、JSON、表格和图。
2. 增加测试、环境锁和一键复现。
3. 更新模板、隐藏链接框并优化空白页。
4. 冻结 Git snapshot 和 claim-evidence map。

---

## 12. Go / Pivot / Stop 决策门槛

### Go：恢复高上限 ICLR 主线

同时满足：

- 合理 counterfactual references 下结论稳定；
- unique-support + cross-fit 后定性 separation 仍存在；
- within-site/layer 独立机制验证仍支持 bypass；
- 非 IOI 任务至少有一项独立验证；
- 表图能够从冻结输出自动复算。

此时可恢复强主张：R/V/A 是三个独立证据轴，并将 IVS 定位为经过验证的 conditional-validity diagnostic。该分支仍具有五个项目中最高的研究上限。

### Pivot：保留有价值的机制 cautionary paper

如果 IVS 的 causal-validity 解释不稳定，但 late answer injection 与 mechanism bypass 仍稳固，则改写为：

> Behavioral restoration can be achieved by late-layer answer injection without restoring the mechanism that an investigator intends to test; reference-aware overlap diagnostics can flag interventions requiring independent mechanism validation.

该分支应删除 `10^12`、universal validity、positivity equivalence 和强 causal-fiber 结论，但仍可能形成一篇有价值的 mechanistic-interpretability methodology paper。

### Stop：当前主线不再成立

如果 unique-support / cross-fit 后 gap 消失，matched counterfactual reference 将主要 lie patches 判为自然，且 within-layer 独立机制验证不再支持 IVS 与 bypass 的关联，则停止以 IVS validity 为中心的当前主线。可保留晚层 patching caution 和 protocol audit 作为新项目基础。

---

## 13. 三种审稿侧重

### Reviewer 1：技术正确性

- 总体评价：核心现象有价值，但 authors' case 尚未在 construct、数值标定和独立机制验证三个环节闭合。
- 最大优点：主动追踪 Pythia 实现问题；晚层 answer smuggling 现象明确。
- 最大问题：当前 `V` 的目标分布未建立，headline gap 可能由 reference 退化放大。
- 当前姿态：Major Revision / Weak Reject。

### Reviewer 2：原创性与意义

- 总体评价：R/V/A 框架、restoration lie 和 reference-aware causal evidence 都有较高潜在影响。
- 关注读者：mechanistic interpretability、causal representation、model debugging、AI evaluation。
- 最大问题：新颖性和意义依赖于 IVS 确实测到 causal validity；若只能测 target-overlap，贡献需要重新定位。
- 当前姿态：高上限，但需要决定性验证后才能判断是否达到强 ICLR 水平。

### Reviewer 3：跨领域可读性与完成度

- 总体评价：论文包装成熟、标题和图表有吸引力，但术语把 overlap、validity 和 alignment 压缩得过近。
- 最大优点：主线容易传播，实践建议明确。
- 最大问题：摘要过密、边界条件不醒目、代码与论文阈值/图表链尚未冻结。
- 当前姿态：写作接近成稿，科学论证与复现封装尚未达到同等成熟度。

### Cross-review synthesis

三种审稿侧重均同意：

- 该项目不是低价值项目，而是高上限、高方差项目。
- 晚层行为恢复不等于机制恢复，是当前最稳固的科学发现。
- 当前最大的风险不是模型数量不足，而是 `V` 的测量对象、reference support 与机制验证设计。
- 继续堆模型不能解决上述问题；P0 实验决定是否保留 oral-level 上限。

---

## 14. Risk / unsupported claims

在完成 P0 验证前，不应写成既定事实的主张包括：

- IVS 已经被证明是一般性的 causal intervention validity score。
- 当前 `10^12` gap 代表自然 activation manifold 的真实普适几何边界。
- pooled NMH AUROC 已经排除了 layer、position 和 time-ordering 混杂。
- `V \nRightarrow A` 已被当前 kNN repair 作为一般因果命题证明。
- depth phase transition 在 transformer architectures 中具有普适性。
- 当前跨模型结果等价于跨任务、跨数据分布或跨训练种子的独立复制。

这些表述目前属于“可能成立但证据不足”，而不是已经被本次审计证明为错误。

---

## 15. 最终主理人判断

ICLR_1 的正确管理方式不是放弃，也不是按原稿直接提交，而是进行一次**短、硬、可证伪的 P0 审计周期**：

1. 先确认测量对象；
2. 再排除 reference/PCA 数值退化；
3. 最后完成不受时序混杂的机制验证。

如果三关通过，应立即恢复其五个项目中最高优先级和最高上限定位；如果不能通过，则应快速 pivot，避免用更多模型规模掩盖 construct-level 问题。
