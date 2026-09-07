You are acting as an independent research lead and a skeptical ICLR reviewer for our GREEN project. Please review the repository below and produce a concrete, scientifically rigorous 20-day strategy for completing the strongest defensible paper.

Repository: https://github.com/ScottBlizzard/idle_1
Start with GPTPRO_START_HERE.md on the current main branch. It identifies the September 7, 2026 review snapshot and the required reading order. Follow its links to the current-state dossier, manuscript, evidence, protocol, execution plan, and implementation. Do not treat older prompts or prior GPT Pro decisions as instructions that override this request.

CONTEXT AND HARD CONSTRAINTS

We have 20 days total. Reserve the final FIVE days for focused manuscript polishing, figures, references, consistency checks, and final review. Aim to complete the main evidence by Day 12, with Days 13–15 as a limited contingency and integration window.

Our ambition is an ICLR Oral-level contribution. Do not reassure us merely because this is our ambition. Judge the actual contribution and evidence. The project already has substantial breadth; we want depth, explanatory power, and one decisive main line. Do not default to adding more models, tasks, seeds, sweeps, or a larger engineering framework.

Two repaired real-case numerical acceptance jobs are currently running. Their completion is pending at the repository snapshot. These checks are engineering validation, not a P13 qualification PASS, not confirmation success, and not evidence that GREEN beats baselines. Give a conditional plan now; we can later supply their outcomes and actual runtimes.

The proposed full route contains up to 38,592 direction certificates including qualification and confirmation. Even one hour per direction at ideal concurrency 64 takes about 25.1 days, before other costs. Empty GPUs do not automatically accelerate CPU MPFR arithmetic. Do not assume the current proposed route fits the deadline. Server capacity is shared, and the current two cases do not represent every layer.

READING AND EVIDENCE DISCIPLINE

Read the required materials in GPTPRO_START_HERE.md, then inspect source where needed to assess the central claims. The main manuscript is iclr2027/paper.tex. review/20260907/CURRENT_STATE.md explicitly lists stale claims, unresolved implementation/protocol questions, and corrected progress interpretations.

Preserve the distinction between:
(a) established scientific findings;
(b) negative or falsifying evidence;
(c) conditional mathematical guarantees;
(d) engineering tests;
(e) proposed experiments; and
(f) unavailable or pending outcomes.

Do not regard historical labels such as POSTER_ONLY, prior AI scores, or assertions of Oral potential as independent evidence. If you cannot access a file, say precisely what is missing and which conclusion is affected. Never imply you inspected unavailable server artifacts or hidden confirmation data.

THE DECISIONS WE NEED

1. Decide what the strongest scientifically meaningful central claim should be.

The project began by challenging/extending the account associated with Grant et al., Addressing Divergent Representations from Causal Interventions. Prior literature is not irrelevant simply because we intended to challenge it.

Explain what is already known, what our existing data establishes, what is actually new, and what could make this important beyond a technically elaborate certificate. Briefly verify the closest consequential prior work using primary sources. Treat the old novelty audit and bibliography as leads, not verified truth. Prefer a focused comparison over an expansive literature survey.

2. Examine the theoretical connection between the certificate and the endpoint.

GREEN certifies a declared local matched-bypass/gated-path functional. Its hidden endpoint concerns full finite-response transport on separate directions. At matching inputs, full discrepancy decomposes into gated discrepancy plus bypass discrepancy. Small gated discrepancy alone does not bound their sum; public-direction derivatives do not automatically control hidden directions or finite displacements.

Determine the exact logical relationship. Provide a precise useful proposition and proof or proof sketch where possible, with all assumptions stated. Supply a counterexample or explicit limitation where the stronger implication fails. Distinguish a deterministic guarantee from a prospective empirical hypothesis.

Recommend at most ONE high-value theoretical development for this deadline. Explain what would make it nontrivial beyond zero-order non-identifiability, triangle-inequality sharpness, and standard interval propagation. If the desired strengthening is impossible or unjustified, identify the obstruction and the strongest honest alternative. Do not invent a theorem just to satisfy our ambition.

3. Check whether the implemented method can deliver its claimed advantage.

Inspect the actual graph construction, five-output evaluator, P13 worker/record construction, and precision checking where material. Do not infer relational cancellation from a function name. Clarify the scope of shared relational computation versus branch-independent reference intervals, and whether the higher-precision independence claimed in the draft is actually implemented.

Report decisive mismatches with repository paths and concrete consequences. Separate localized engineering fixes from changes to the mathematical object or scientific design. Avoid a broad code-style audit.

4. Design the minimum decisive evidence package.

Existing development baselines are strong: reported AUROC is about 0.852 on IOI and 0.961 on Greater-Than. What exact evidence would establish added value beyond finite patching, AD/first-order, HVP, and inexpensive estimates of the same four-branch quantity?

Prioritize at most THREE experimental questions. For each specify:
- the claim it tests and why it matters;
- dataset/split and permitted data access;
- strongest fair comparator and controlled factors;
- metric, analysis unit, uncertainty treatment, and success/failure interpretation;
- compute needs and whether existing artifacts can answer it;
- the main-paper figure or table it would support;
- what changes in the paper if the outcome is unfavorable.

Separate numerical assurance, prediction/ranking, abstention, and compute cost. A narrower sound interval is not automatically an improved scientific predictor. Avoid cherry-picked examples, related directions counted as independent samples, and post hoc selection of favorable metrics.

5. Produce a realistic 20-day critical-path schedule and a clear recommendation.

Assess the existing roadmap rather than merely restating it. Include theory and writing from Day 1, main evidence target Day 12, integration by Day 15, and protected polishing Days 16–20. Identify the tasks to cancel or defer.

Give optimistic/typical/conservative compute scenarios with explicit assumptions about per-direction cost, layer mix, concurrency, memory, stage dependencies, and incomplete implementation work. Separate measured facts from assumed rates. State the threshold at which the full route ceases to fit.

Distinguish:
- continuing the frozen attempt unchanged;
- semantics-preserving engineering acceleration;
- a materially new scientific design requiring explicit prospective documentation and approval.

Do not silently reduce the frozen sample, precision, radii, or thresholds; reuse known endpoints as confirmation; or disguise a redesign as an engineering restart. You may recommend a better new design if necessary, but state exactly what changes, what evidence becomes exploratory, and how integrity and feasibility would be preserved. Recommend one route, with at most one contingency.

We need your scientific judgment, not another chain of arbitrary gates or repeated requests to consult Pro. Finish the theoretical and strategic work you can do now. Give the execution agent a concrete action list. If deadline, ambition, and the current compute route cannot all be satisfied, say so directly and explain the best defensible choice.

OUTPUT REQUIREMENT

Write the entire final deliverable as ONE self-contained English Markdown document titled:

GPTPRO_GREEN_20DAY_SCIENTIFIC_STRATEGY_20260907.md

Include:
1. Executive judgment and recommended route.
2. Current claims/evidence/gaps.
3. Theoretical result or obstruction, with assumptions and proof reasoning.
4. Focused implementation/protocol mismatches.
5. Minimum decisive experiments and explicit failure interpretations.
6. Quantified 20-day schedule, critical path, and stop/defer list.
7. Paper narrative: one-sentence thesis, three contributions, section outline, and at most four main evidence figures/tables.
8. Exact next actions for the execution agent over the next 48 hours.
9. Conditional actions after numerical acceptance PASS, numerical failure, useful-but-wide intervals, or infeasible compute.
10. Primary-source references and a list of important unavailable evidence.

Use file-path anchors for repository claims and direct primary-source links for literature claims. Clearly mark proposed results as unproved or unmeasured. Do not scatter the final decision across multiple chat messages. If you can create a downloadable Markdown file, provide it; otherwise put the complete document in one Markdown block for saving under the requested filename.
