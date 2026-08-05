# 7 月审计 P0 执行记录（2026-08-04）

## 当前结论

7 月审计指出的三个核心风险均被实验确认。当前证据不再支持把旧 IVS 解释为一般性的 causal intervention validity，也不支持“低 IVS 导致 mechanism bypass”的强主张。项目应转向更窄且证据更强的结论：

> Behavioral restoration does not imply mechanism restoration. Activation overlap is conditional on an explicitly chosen reference distribution and is neither a necessary nor sufficient certificate of mechanism alignment.

同时，已知真值合成任务的 cross-fit smoke test 表明：当目标分布由任务设计明确给定时，不依赖 scale floor 的经验尾概率仍能识别 off-support donor。因此应保留“条件重叠诊断”这一较窄贡献，删除数量级几何和一般因果有效性表述。

## 1. 服务器与执行边界

- 服务器：`ccj@10.10.217.244`
- GPU 0–3：已有 GUI 模型，占用约 19–21 GB/卡，未触碰。
- GPU 7：Isaac 仿真，未触碰。
- 本轮只使用 GPU 4–6。
- 根分区仅余约 5.3 GB；日志与中间结果写入 `/mnt/sdb/ccj/iclr_1_runs/`。
- 未使用旧 `push.ps1`，因为该脚本会先递归删除远端 `src`。

## 2. 实现修复

新增 `src/validity_crossfit.py`，并保持旧 `validity_fast.py` 不变以保留历史复现能力。新实现包括：

1. PCA/reference fitting 与 normalization calibration 使用独立样本。
2. held-out query 不再错误跳过最近邻；只有显式 reference self-query 才排除自身。
3. PCA rank 强制不超过观测数值秩。
4. 同时报 raw distance、cross-fit z-score 和不依赖 raw-unit floor 的经验上尾概率。
5. 输出唯一 activation 数、有效秩、保留秩、条件数和校准统计。
6. 数值测试覆盖 held-out kNN、自邻居排除、低秩截断、有限值与分布偏移分离。

## 3. 四参考分布与 cross-fit 实验

设计：GPT-2，3 个独立 prompt seed；每个 seed 使用 256 个唯一 fit prompts、128 个唯一 calibration prompts、80 个 evaluation prompts。比较：

1. corrupt observational reference；
2. clean/source reference；
3. clean-corrupt mixture；
4. matched semantic-counterfactual reference。

共得到 42 个高恢复 site observations（每个 seed 14 个）。

| Reference | Mean overlap-z | 2.5%–97.5% | `<0.3` | Max absolute reconstruction z |
|:--|--:|:--:|--:|--:|
| corrupt observational | 0.418 | 0.140–0.519 | 9/42 | 4.94 |
| clean/source | 0.503 | 0.435–0.525 | 0/42 | 0.306 |
| mixture | 0.481 | 0.357–0.542 | 0/42 | 0.834 |
| matched counterfactual | 0.492 | 0.457–0.529 | 0/42 | 0.406 |

关键事实：

- 旧百万级 reconstruction z-score 消失；最大值降至 4.94。
- 9/42 个低重叠标签全部来自三个 seed 中的 late last-token sites。
- 换成 clean、mixture 或 matched counterfactual 后，9/9 全部翻转。
- corrupt 与 matched reference 的 site 排序 Spearman rho 为 -0.291。
- scale floor 从 `1e-8` 扫到 `1e-2` 时，新 IOI 结果不变，因为 held-out calibration scale 已高于这些 floor。

决策：旧 headline 主要依赖 reference 定义和退化校准；安全 estimand 是 conditional target-reference overlap。

## 4. 旧 NMH 验证的时间/层位审计

对现有 GPT-2 和 GPT-2 Medium NMH JSON 做重新分析。`resid_post` patch at layer L 只能影响 layer strictly greater than L 的 NMH attention。

| Model | Pooled AUROC | 全部 NMH 均在下游的 AUROC | 可比较的精确 layer-position strata | 层内 AUROC |
|:--|--:|--:|--:|--:|
| GPT-2 | 0.894 | n/a | 0/14 | n/a |
| GPT-2 Medium | 0.778 | 0.714 | 1/36 | 0.500 |

GPT-2 中，全部 NMH 均在 patch 下游的 36 个样本没有一个 bypass；12 个发生在全部 NMH 之后的样本则全部是 bypass。旧 pooled AUROC 主要是时间和 site 分类，而不是独立机制验证。

## 5. 新的 within-site、时间可达机制实验

设计：GPT-2，3 个 seed；固定 IO position；只使用 L0–L8 `resid_post`，确保两个 NMH 均严格位于 patch 下游；每个 layer/seed 保留 80 个 prompt-level observations。控制 seed、layer、context 和 restoration。

| Layer | Mean R | Mean NMH recovery | Corrupt overlap | Matched overlap |
|--:|--:|--:|--:|--:|
| 0 | 1.027 | 1.021 | 0.486 | 0.490 |
| 1 | 1.067 | 1.022 | 0.490 | 0.487 |
| 2 | 1.042 | 0.959 | 0.488 | 0.492 |
| 3 | 1.006 | 0.724 | 0.493 | 0.487 |
| 4 | 0.884 | 0.336 | 0.493 | 0.495 |
| 5 | 0.869 | 0.265 | 0.500 | 0.498 |
| 6 | 0.861 | 0.271 | 0.510 | 0.507 |
| 7 | 0.858 | 0.270 | 0.529 | 0.512 |
| 8 | 0.854 | 0.292 | 0.529 | 0.515 |

L4–L8 在每个 seed 中均满足 mean R > 0.8 且 mean `(R-A)` > 0.4，因此 `R` 与 `A` 的分离不是旧 NMH 时间伪影。但这些 layer 的 overlap 位于约 0.49–0.53，并不低。跨 2160 个 prompt-site observations 的固定效应残差 Spearman rho 仅为：

- corrupt overlap vs NMH：0.043；
- matched overlap vs NMH：0.102。

决策：保留 `R does not imply A`；删除“low IVS 是 bypass 的必要条件或原因”。

## 6. 公平 baseline

所有监督方法使用相同的 balanced train/validation/test split、标准化和调参预算。标签仅表示 held-out corrupt/reference condition 与 clean/source shift，不是机制或因果真值。

| Method | Overall AUROC | IO-position | Last-position |
|:--|--:|--:|--:|
| cross-fit IVS | 0.578 | 0.477 | 0.946 |
| 3-component logistic | 0.570 | 0.457 | 0.984 |
| 3-component MLP | 0.530 | 0.417 | 0.944 |
| activation logistic | 0.539 | 0.413 | 1.000 |
| activation MLP | 0.577 | 0.462 | 1.000 |

决策：IVS 没有明显监督基线优势；旧 MLP 失败不能支持高维小样本过拟合叙事。所有方法都主要在检测 last-position shift，而不是 causal validity。

## 7. 已知真值合成任务

Full-sample cross-fit 使用 2000/1000 个独立且 token-level 唯一的 fit/calibration samples，以及 256 个 evaluation samples。共评估 4 个模型 seed，其中 3 个在本轮从不同初始化独立训练：

- 所有模型均达到 gate-off/on accuracy 1.000；
- mean target gate-off ECDF overlap：clean 0.710，donor 0.007；
- mean/min prompt-level AUROC：0.9999/0.9996；
- 每个 seed 的 clean-donor ECDF gap 均至少为 0.498。

离散 gate 位点的 raw z-score 仍可达到百万级，并依赖 scale floor；经验尾概率仍能正确分离。因此合成任务支持的是“在目标 support 已由任务设计明确指定时，cross-fit overlap 可识别 off-support donor”，而不是数量级几何主张。

## 8. 当前 go/pivot 决策

- 原 IVS-centered causal-validity 主线：**Pivot**。
- 新主线：**行为恢复不能单独支持机制恢复；reference-conditioned overlap 是独立但有限的证据轴。**
- 可以保留的方法贡献：明确 reference、cross-fitting、经验校准、报告原始分量与有效秩。
- 必须删除或重写：`10^12 gap`、PCA95 intrinsic-dimension 叙事、IVS 是一般 causal-validity certificate、旧 pooled NMH AUROC、监督基线必然过拟合。

## 9. 下一实验队列

1. 做 support/calibration factorial：旧重复 support、扩大唯一 support、cross-fit、ECDF 分别消融。
2. 在非 IOI 的已知机制任务上复现 `R ≠ A`，优先 Greater-Than 或重新设计的 dormant-path synthetic circuit。
3. 为新 estimand 冻结统一阈值、reference 构造和统计单位。
4. 之后再重构论文；在上述结果冻结前不修改主文数字。

## 10. 2026-08-05 oral 主线升级：IRS P0

7 月审计后的主线不再把 overlap 解释为一般 causal validity。新的理论对象是 local interventional response agreement：behavioral restoration 只匹配下游函数值（零阶证据），而 IRS 使用同一组 target-admissible probes 比较 clean target computation 与 patched-corrupt computation 的局部响应场。

### 10.1 Composite split-conformal 修复

`validity_crossfit.py` 新增 `overlap_conformal`：

1. geometry fit、component normalization、final composite calibration 相互独立；
2. 最终 nonconformity 是独立归一化 component z 的 mean-softplus；
3. query p-value 只相对 untouched final calibration 做有限样本 rank；
4. 旧 `overlap_ecdf` 明确保留为边缘 ECDF 几何平均，不再宣称联合 coverage。

模拟 false-alarm regression test 在目标分布下通过；shifted queries 的 conformal overlap 显著更低。

在原 gated-path trained-transformer ground truth 上重新跑 4 个模型 seed：composite conformal donor-detection AUROC mean/min 为 0.9997/0.9993；mean clean/donor conformal overlap 为 0.634/0.003。3 个 checkpoint 来自独立初始化训练。该结果恢复了 overlap 模块的正面统计解释，但只限于任务设计明确给定 target support 的场景。

### 10.2 Analytic four-quadrant ground truth

5 个独立 seed 均通过以下四象限：

| Condition | R | Support | IRS | Result |
|:--|:--:|:--:|:--:|:--:|
| true restoration | high | high | low | 5/5 pass |
| on-support restoration lie | high | high | high | 5/5 pass |
| off-support shortcut | high | low | high | 5/5 pass |
| natural ineffective intervention | low | high | low | 5/5 pass |

在已知 gradient gap 为 4、维数为 8 时，理论 directional MSE 为 2.0；IRS 在每个 seed 恢复到数值误差范围内的 2.0。

### 10.3 GPT-2 temporally eligible IRS

Protocol：GPT-2，3 个独立 prompt seed；固定 IO position；L0--L8 `resid_post`；所有 measured NMH 严格位于 patch 下游；每层 80 个 prompts、每 prompt 8 个 context-matched clean-reference chord probes。clean target 与 patched-corrupt computation 使用完全相同的 activation center、probe direction 和 endpoint。probe endpoint 使用新 composite conformal audit。

| Layer | Mean R (min seed) | Mean NMH (max seed) | IRS normalized RMSE | IRS cosine | Endpoint accept (min seed) |
|--:|:--:|:--:|--:|--:|:--:|
| 0 | 1.017 (1.011) | 1.021 (1.033) | 0.089 | 0.996 | 0.999 (0.998) |
| 1 | 1.037 (1.028) | 1.009 (1.022) | 0.140 | 0.990 | 0.999 (0.998) |
| 2 | 1.017 (1.011) | 0.958 (0.969) | 0.266 | 0.959 | 0.995 (0.986) |
| 3 | 0.978 (0.974) | 0.711 (0.726) | 0.534 | 0.838 | 1.000 (1.000) |
| 4 | 0.847 (0.830) | 0.304 (0.314) | 0.778 | 0.618 | 0.999 (0.997) |
| 5 | 0.825 (0.809) | 0.234 (0.259) | 0.957 | 0.404 | 0.999 (0.997) |
| 6 | 0.821 (0.802) | 0.246 (0.267) | 0.931 | 0.432 | 0.998 (0.995) |
| 7 | 0.819 (0.800) | 0.247 (0.267) | 0.797 | 0.583 | 0.997 (0.997) |
| 8 | 0.813 (0.794) | 0.269 (0.288) | 0.784 | 0.582 | 0.998 (0.997) |

冻结门槛下，L4--L7 在每个 seed 中均满足 high R、low NMH、admissible endpoints。其平均 IRS 为 0.866；NMH-aligned L0--L2 为 0.165，相差约 5.3 倍。控制 seed 与 restoration 后的 layer-level residual Spearman IRS vs NMH 为 -0.762（p=3.89e-6）。控制 seed/layer/context/restoration 后的 prompt-level rho 仅 -0.060（p=0.005），因此当前证据支持 IRS 识别机制阶段差异，不支持把它宣传为逐 prompt 高精度分类器。

### 10.4 当前解释与未通过项

当前支持：

> High behavioral restoration can coexist with target-admissible activation neighborhoods and sharply different local interventional response fields. Restoration is therefore zero-order evidence, while support and local functional mechanism agreement are distinct axes.

尚未通过：

1. IRS 是否稳定优于单一 clean--corrupt mediator-interaction direction；
2. probe interpolation/radius/count sweep；
3. 非 IOI pretrained task；
4. trained gated-transformer composite conformal rerun；
5. 结构性 circuit identification（当前明确不作此主张）。
