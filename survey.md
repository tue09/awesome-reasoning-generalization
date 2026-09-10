
# Beyond the Training Distribution: A Survey of Reasoning Generalization in Large Language Models

## 1. Introduction

Reasoning benchmarks are useful only to the extent that success transfers beyond the examples, templates, domains, and difficulty levels used to build a model. This requirement is stricter than held-out accuracy. A model may answer unseen items from a familiar template while failing after a symbol renaming, a change in wording, an extra inference step, a new language, or a new tool interface. Recent controlled studies expose this gap in arithmetic, formal languages, graph planning, scientific composition, analogy, and multimodal reasoning. For example, schema-preserving number changes can sharply reduce accuracy on GSM8K ([Numeric-Remapping Attacks, 2026](https://arxiv.org/abs/2606.03606)); performance can collapse as scientific domains are composed ([XDomainBench, 2026](https://arxiv.org/abs/2605.14754)); and models that solve familiar automata constructions still fail on unseen constructions requiring global consistency ([Beyond Memorization, 2026](https://arxiv.org/abs/2601.13392)). These failures motivate a survey centered on transfer, not benchmark score alone.

We use the standard statistical formulation in Section 2: generalization concerns test risk relative to a stated training distribution. We then apply four scope criteria to reasoning studies. Each core paper must identify the reasoning task, the reference training exposure, the test shift, and evidence of transfer or failure. These are inclusion criteria for this survey, not a new formal definition. They prevent a test item that is merely absent from supervised fine-tuning, but familiar from pre-training, from being described without qualification as beyond the training distribution.

The literature does not support one universal recipe. Training methods can favor reusable rules, but they can also narrow exploration or overfit the surface form of demonstrations. Inference methods can expose latent competence, but added compute often cannot repair a missing algorithm. Architectural changes can align computation with iterative or symbolic structure, but many results are obtained on small controlled models. Analysis papers show that success depends on data coverage, representation alignment, optimization dynamics, and the form of the shift. The central claim of this survey is thus:

> Reasoning generalization is produced by an alignment among the training signal, the inference procedure, the model's computational structure, and the test shift. Improving only one part rarely transfers across every shift.

This survey organizes work around four intervention and evidence questions:

1. **Training for Generalization:** How can learning produce reasoning skills that transfer to unfamiliar problems?
2. **Inference for Generalization:** How can a fixed model solve unfamiliar problems through changes at inference?
3. **Architecture for Generalization:** Which structural properties support transferable reasoning?
4. **Analysis of Generalization:** When does reasoning generalize, and what explains its successes and failures?

The fourth pillar separates behavioral observations from explanations. Explanations are then divided into empirical mechanistic evidence and theoretical results under explicit assumptions. This separation matters. A change in accuracy establishes a behavioral fact, an activation probe gives correlational internal evidence, a causal intervention supports a mechanism, and a theorem applies only within its formal model.

**Review protocol.** The literature search was updated on 8 September 2026. Searches covered reasoning generalization, out-of-distribution reasoning, systematic and compositional generalization, length generalization, reasoning transfer, and test-time generalization. A targeted abstract-field audit additionally combined post-training, SFT, distillation, RLVR, and GRPO terms with generalization, transfer, unseen-domain, and OOD terms. Capped queries were split by submission period to prevent older 2026 and 2025 records from falling below arXiv's result limit. After deduplication, we screened 6,007 arXiv records and checked relevant ACL Anthology and OpenReview entries. A second pass read every retained abstract and inspected the full paper whenever the abstract did not identify the evaluation split. Every core study required four identifiable elements: reasoning task, reference training exposure, test shift, and evidence of transfer or failure. We excluded ordinary reasoning-score improvements, broad robustness studies in which reasoning was only one task type, model-to-model transfer without an unfamiliar-problem shift, and expressivity results with no generalization claim. The final evidence corpus contains 124 core studies: 86 first posted to arXiv in 2026, 29 in 2025, and 9 earlier foundations. Four nearby surveys and two contextual comparisons are not counted as core evidence. The 130-source machine-readable manifest includes titles, authors, abstracts, taxonomy labels, source URLs, and local PDF paths in [`paper_manifest.tsv`](paper_manifest.tsv); audit exclusions and reasons are recorded in [`excluded_papers.tsv`](excluded_papers.tsv).

**Relation to nearby surveys.** Existing reviews cover broader NLP generalization ([Hupkes et al., 2023](https://arxiv.org/abs/2210.03050)), inductive reasoning ([Survey of Inductive Reasoning, 2026](https://arxiv.org/abs/2510.10182)), length extrapolation in algorithmic tasks, reinforced reasoning, or generalizability of LLM agents ([Agent Generalizability Survey, 2026](https://arxiv.org/abs/2509.16330)). This survey differs by treating training, inference, architecture, behavior, mechanisms, and theory as parts of one question: whether a reasoning procedure survives a stated shift. It also gives priority to 2026 evidence while retaining only the older papers needed to define the main methods and limits.


## 2. Preliminaries

**Definitions.** We begin by formalizing **reasoning generalization** through the standard learning-theoretic notion of performance on previously unseen problems. Let \(x\) denote an input problem, \(y\) its reference answer, and $f_\theta(x)$ the prediction of a language model under a specified inference procedure. Given training and test distributions $P_{\mathrm{tr}}(x,y)$ and $P_{\mathrm{te}}(x,y)$, generalization concerns achieving low expected test loss,

$$
R_{\mathrm{te}}(\theta)
=
\mathbb{E}_{(x,y)\sim P_{\mathrm{te}}}
\left[\ell\!\left(f_\theta(x),y\right)\right],
$$

where $\ell$ measures task-specific prediction error. **In-distribution generalization** evaluates unseen examples drawn from the same underlying distribution, $P_{\mathrm{te}}=P_{\mathrm{tr}}$. **Out-of-distribution generalization** evaluates performance when $P_{\mathrm{te}}\neq P_{\mathrm{tr}}$, requiring the model to remain effective under distribution shift. These distinctions follow the standard formulation of generalization under changing data distributions ([Liu et al., 2023](https://arxiv.org/abs/2108.13624)).

For reasoning tasks, distribution shifts can concern the structure and complexity of the problems themselves. **Compositional generalization** concerns combining familiar components or operations in combinations absent from training ([Lake and Baroni, 2018](https://arxiv.org/abs/1711.00350)). **Length generalization** concerns extrapolating from shorter training instances to longer problem instances ([Anil et al., 2022](https://arxiv.org/abs/2207.04901)). These settings assess whether reasoning performance extends beyond the configurations encountered during learning. For models developed through multiple training stages, the reference distribution must be specified: a problem may be outside the distribution used for supervised fine-tuning or reinforcement learning while still resembling examples encountered during pretraining. Consequently, generalization beyond post-training data must be distinguished from generalization beyond pretraining exposure ([Ni et al., 2025](https://arxiv.org/abs/2510.15990)).

### 2.1 What can shift?

Reasoning generalization studies vary along several independent axes:

| Shift axis | Training side | Test side | What it probes |
| --- | --- | --- | --- |
| Instance | Seen samples from a fixed generator | New samples from the same generator | Ordinary statistical generalization |
| Surface or format | Familiar wording, symbols, or serialization | Paraphrases, renamed symbols, new notation, or reordered facts | Invariance to irrelevant form |
| Composition | Seen primitives and combinations | New combinations of familiar primitives | Reuse and recombination of rules |
| Length or depth | Short sequences or shallow proofs | Longer sequences, more hops, or deeper nesting | Extrapolation of iterative computation |
| Difficulty | A restricted success or complexity range | Easier, harder, or differently difficult items | Transfer across capability regimes |
| Task or domain | One task family or subject | A new task family or subject | Portability of reasoning procedures |
| Language or modality | One language or representation | Another language, text format, image, or audio | Separation of reasoning from the interface |
| Environment or tool | Familiar APIs, rules, or feedback | New interfaces, tools, dynamics, or reward structure | Agentic and procedural transfer |
| Knowledge | Facts used during learning | New facts governed by familiar rules | Use of learned rules on new content |

These axes can co-occur. A larger visual planning map changes both input length and spatial complexity. Translating a math problem can change language, tokenization, and surface cues. Strong studies isolate one axis or use factorial controls to estimate interactions among them.

### 2.2 Model, procedure, and system generalization

It is useful to distinguish three objects. **Model generalization** concerns what the parameterized model can compute with a fixed decoding rule. **Procedure generalization** concerns the pair of model and inference algorithm, such as decomposition, search, verification, or tool use. **System generalization** also includes retrievers, verifiers, memories, and environments. A tool-augmented model may solve longer tasks without the base model learning a length-general algorithm. This is a valid system result, but it should not be reported as a change in the model's internal capability.

We write an inference procedure as $a \in \mathcal{A}$ and its prediction as $f_{\theta,a}(x)$. A fair comparison reports both risk and inference cost:

$$
\mathcal{G}(a; P_{\mathrm{tr}},P_{\mathrm{te}})
=
R_{\mathrm{te}}(\theta,a)-R_{\mathrm{tr}}(\theta,a),
\qquad
C(a)=\mathbb{E}[\text{tokens, calls, time, or FLOPs}].
$$

The gap $\mathcal{G}$ is meaningful only when the training and test risks use comparable tasks. For cross-domain evaluation, absolute target performance, the change from the base model, and the change relative to an in-domain control should all be reported.

### 2.3 Evidence standard

This survey uses four evidence levels:

1. **Behavioral:** controlled performance under a named shift.
2. **Representational:** probes or similarity measures associate internal states with transfer.
3. **Causal-mechanistic:** patching, ablation, or a training intervention changes both the proposed mechanism and generalization.
4. **Theoretical:** a proof establishes a result for an explicit architecture, precision model, data process, or optimization assumption.

No level replaces another. Theorems can identify possible or impossible computations but may not predict optimization in large models. Causal results on small models can identify a mechanism without proving that frontier models use it. Behavioral evaluations remain necessary because the final object of interest is transfer under real shifts.

### 2.4 Taxonomy assignment rule

The pillars are defined by the primary scientific question, not by vocabulary in a paper's title. A paper is assigned to **Training** when its main intervention changes learned parameters through data or feedback; to **Inference** when base-model weights remain fixed and the intervention changes computation at deployment; to **Architecture** when the main claim concerns the computational structure or inductive bias of the model class; and to **Analysis** when the main contribution is a behavioral pattern, causal explanation, or theorem. Hybrid papers are discussed across sections but receive one primary manifest label. This rule keeps architecture separate from training: recurrence or relative position can be studied as a structural property even when a particular training recipe is used to fit it.

**Overview.** Section 3 studies how pre-training, mid-training, supervised fine-tuning, distillation, reinforcement learning, and self-improvement shape transferable reasoning. Section 4 covers prompting, decomposition, search, verification, tools, and memory for fixed models. Section 5 reviews recurrent depth, positional structure, modular and neuro-symbolic systems, and alternative computational representations. Section 6 separates behavioral patterns, empirical mechanisms, and theoretical accounts, then derives a common evaluation standard and open problems.

## 3. Training for generalization

Training for generalization asks which learning signals make a reasoning procedure reusable after the surface form, content, composition, or complexity changes. The key issue is not whether a model fits hard examples. It is which solution among many training-compatible solutions the optimizer selects. A surface heuristic and an abstract algorithm can have similar training loss but different behavior under shift.

### 3.1 Pre-training exposure and capability boundaries

Pre-training claims require an explicit exposure boundary. [On the Interplay of Training Stages](https://arxiv.org/abs/2512.07783) provides controlled evidence by varying pre-training, mid-training, and RL data and measuring whether later training expands large-pass@$k$ support on held-out reasoning problems. It finds expansion only when pre-training leaves headroom and later data lie near the model's capability boundary.

[Nexus](https://arxiv.org/abs/2604.09258) is useful background on pre-training optimization: at similar aggregate pre-training loss, its optimizer lowers generic OOD language-modeling loss and improves downstream benchmarks, including GSM8K. The paper does not isolate a reasoning-specific training-to-test shift, so this survey does not count it as core evidence for reasoning generalization.

### 3.2 Mid-training and continued pre-training

Mid-training sits between generic pre-training and task-specific post-training. It includes continued pre-training on reasoning-rich corpora, long-context adaptation, domain mixtures, and curricula that alter the model's usable interface without specializing it to one benchmark.

Position exposure is a clear example. [Randomized YaRN](https://arxiv.org/abs/2606.23687) trains on short contexts with positions sampled from a larger range, then evaluates the context-length shift from 16K to 128K on BABILong, MRCR, and LongBench v2. This is evidence for long-context positional transfer, not by itself for deeper reasoning. [Random Float Sampling](https://arxiv.org/abs/2602.14050) makes position indices continuous and randomized, reducing the train-test mismatch for unseen lengths. These results treat position as part of the training distribution, not only as an architectural detail.

Length transfer can also be learned across tasks. [Meta-RFFT](https://arxiv.org/abs/2502.11525) post-trains on rule-following tasks with controlled lengths and tests new tasks at longer lengths, combining broad multi-task exposure with minimal target adaptation. This separates transfer of a length-handling procedure from extrapolation on one familiar operation.

Verified synthetic data offers another route. [Selective Left-Shift](https://arxiv.org/abs/2607.07748) moves iterative compiler and test feedback from deployment into offline data generation, then combines SFT with RLVR for low-resource programming languages. [Learning from Synthetic Data Improves Multi-hop Reasoning](https://arxiv.org/abs/2603.02091) finds that rule-generated fictional knowledge can improve real multi-hop question answering. [Fundamental Reasoning Paradigms](https://arxiv.org/abs/2602.08658) trains deduction, induction, and abduction on symbolic trajectories, then tests transfer to natural-language tasks with real-world knowledge. The common mechanism is procedural supervision: the data teaches how to combine information without requiring the facts at test time to have appeared during training.

Mid-training can still overfit. [Rethinking Easy-to-Hard](https://arxiv.org/abs/2603.27226) finds no robust benefit from difficulty ordering over random sampling across SFT and RL on controlled deduction. The lesson is that curriculum value depends on which invariance or subroutine is exposed, not on an easy-to-hard label alone.

### 3.3 Post-training for generalization

Post-training adapts a pre-trained model through supervised traces, preference or reward signals, and hybrid self-improvement loops. These approaches share the same stage boundary but differ in what selects the transferable procedure: demonstrations, rewards, or an interaction between generated experience and parameter updates.

#### 3.3.1 Supervised adaptation: SFT and distillation

Supervised fine-tuning can either expose a transferable procedure or compress a narrow answer style. [Learning to Adapt SFT Data](https://arxiv.org/abs/2605.26924) treats a fixed expert dataset as distributionally mismatched supervision and learns transformations better suited to the target model. [Memorize Theorems, Not Instances](https://arxiv.org/abs/2605.09270) changes the target of supervision from problem-solution surface patterns to explicit theorem application. [Learning from Mistakes](https://arxiv.org/abs/2601.04992) shows that negative trajectories may retain valid intermediate steps, slow overfitting, and preserve policy entropy. These papers agree that filtering only for correct final answers can remove useful variation.

SFT generalization is also checkpoint-, data-, and capability-dependent. [Rethinking Generalization in Reasoning SFT](https://arxiv.org/abs/2604.06628) reports a dip-and-recovery pattern in cross-domain performance, stronger transfer from verified long traces, and a divide between stronger models that acquire procedures and weaker models that imitate surface form. [Why Does RL Generalize Better Than SFT?](https://arxiv.org/abs/2602.10815) attributes part of the VLM gap to implicit difficulty filtering by RL and shows that difficulty-curated SFT can surpass the compared RL models. In a controlled stage-wise study of biological reasoners, [How Post-Training Shapes Biological Reasoning Models](https://arxiv.org/abs/2606.16517) likewise finds that OOD performance can peak early and decline while ID accuracy continues to rise. [From Meta-Thought to Execution](https://arxiv.org/abs/2601.21909) instead separates abstract strategy supervision from instance-specific execution and reports gains on both ID and OOD evaluations. Final-checkpoint comparisons can therefore confuse SFT with a fixed algorithmic effect when the actual variables include optimization time, data difficulty, and the granularity of the supervision target.

Distillation is most transferable when it respects the student's state and the reasoning decisions that matter. [Making Expert Reasoning Learnable](https://arxiv.org/abs/2602.02405) adapts human expert solutions whose omitted steps make them distributionally mismatched to a student. [Geometric Self-Distillation](https://arxiv.org/abs/2607.06855) limits predictive drift when a privileged teacher view is more confident than the student can justify. [Invariant Gradient Alignment](https://arxiv.org/abs/2606.05025) aligns updates across logically isomorphic examples from different semantic domains. [RP-OPSD](https://arxiv.org/abs/2608.06347) concentrates multilingual distillation around reasoning pivots rather than surface realization. [RELAY](https://arxiv.org/abs/2502.08482) aligns loop iterations with CoT steps and transfers long reasoning traces from a length-generalizing looped model to an autoregressive model.

Teacher behavior can transfer too broadly or lose useful uncertainty. [Every Coin Has Two Sides](https://arxiv.org/abs/2608.16647) finds that on-policy distillation transfers across domains, languages, and reasoning horizons mainly when teacher and student share a model origin, while multiple teachers can interfere. [Why Does Self-Distillation Degrade Reasoning?](https://arxiv.org/abs/2603.24472) links OOD degradation to suppressed epistemic verbalization when a richly conditioned teacher produces overly certain traces. These results make teacher-student compatibility and uncertainty expression part of the transfer target, not implementation details.

Post-training objectives also differ in their response to nuisance correlations. [Causality-Aware Post-Training](https://arxiv.org/abs/2506.09433) decomposes biased prediction into event estimation and intervention, improving OOD performance on formal causal and logical reasoning tasks. This result supports explicit counterfactual controls rather than assuming that an optimizer is inherently robust.

The combined evidence favors four properties of supervised reasoning data:

- **Rule exposure:** traces show which operation applies and why.
- **View diversity:** equivalent problems vary in names, wording, language, and format.
- **Error diversity:** incorrect trajectories reveal failure patterns and preserve alternatives.
- **Model compatibility:** supervision is reachable from the student's current policy rather than copied from an arbitrarily stronger teacher.

#### 3.3.2 Reinforcement learning: capability selection and reward design

Outcome-only reinforcement learning can increase benchmark accuracy without identifying which part of a trajectory should transfer. Recent methods use rewards that target invariance, process quality, or environment composition.

The contrast between SFT and RL is conditional rather than absolute. [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161) finds stronger RL transfer to unseen text-rule and visual variants in controlled environments. [Generalization of RLVR Using Causal Reasoning](https://arxiv.org/abs/2512.20760), however, finds an RL advantage over SFT only for particular model sizes and training query levels, with benefits depending on initial competence. [RL Fine-Tuning Heals OOD Forgetting](https://arxiv.org/abs/2509.12235) further shows that RL often recovers an early SFT OOD peak rather than exceeding it. [When RL Fails after SFT](https://arxiv.org/abs/2606.09932) attributes a related failure to reduced plasticity after excessive SFT. The relevant comparison is therefore a checkpoint-by-checkpoint trajectory with matched data and compute, not one terminal SFT model against one terminal RL model.

[GRAIN](https://arxiv.org/abs/2608.27142) rewards structural invariance under changes to node names and task narratives. [Group Causal Counterfactual Policy Optimization](https://arxiv.org/abs/2602.06475) scores reasoning steps by robustness and effectiveness under counterfactual candidate comparisons. [Rubric-Grounded RL](https://arxiv.org/abs/2605.08061) decomposes a judge reward into grounded criteria and reports transfer to benchmarks outside the source corpus. [RACES](https://arxiv.org/abs/2606.12373) recursively composes verifiable environments to create new reasoning patterns. [STRATAGEM](https://arxiv.org/abs/2604.17696) uses self-play rewards intended to favor abstract, domain-independent trajectories. [DYPO](https://arxiv.org/abs/2604.08926) combines SFT and RL signals through an adaptive exploitation-exploration gate. [Can LLMs Learn to Reason Robustly under Noisy Supervision?](https://arxiv.org/abs/2604.03993) distinguishes active from inactive label noise and uses online label refinement to improve both ID and OOD reasoning. [Sharpness-Guided GRPO](https://arxiv.org/abs/2511.00066) instead reweights tokens using model confidence to reduce a gradient-norm proxy for sharp updates. It reports gains over GRPO across math, logic, and agentic QA, including out-domain QA datasets; its formal bound concerns finite-sample generalization, so the direct evidence for distributional transfer is empirical.

Direct cross-domain results range from broad transfer to near-total failure. [CEDAR-GRPO](https://arxiv.org/abs/2608.14791) trains four models on domain-neutral abduction and improves all of them on eleven unseen tasks spanning investigation, clinical reasoning, and code debugging. [X-Reasoner](https://arxiv.org/abs/2505.03981) transfers text-only SFT and RLVR to multimodal and medical tasks, while [SUPERNOVA](https://arxiv.org/abs/2604.08477) obtains gains on unseen reasoning benchmarks from natural-instruction RLVR data. [GraphDancer](https://arxiv.org/abs/2602.02518) trains graph exploration in one domain and tests unseen domains and question types. In contrast, [Breaking Barriers](https://arxiv.org/abs/2506.19733) finds through observational and interventional studies that gains can vanish when target domains require different reasoning patterns. [Paying Less Generalization Tax](https://arxiv.org/abs/2601.18217) finds for RL-trained agents that state-information richness and planning complexity predict transfer better than text similarity or realism. [Transfer-Aware Curriculum](https://arxiv.org/abs/2606.25178) responds to this asymmetry by sampling domains according to whether their gradients help other domains.

Whether RL expands capability or merely changes sampling depends on the boundary being tested. [Does RL Really Incentivize Reasoning Capacity?](https://arxiv.org/abs/2504.13837) finds across model families that RL gains at small sampling budgets often disappear at large pass@$k$, indicating distributional reweighting rather than new support. A controlled pre-training, mid-training, and RL study reaches a narrower conclusion: [On the Interplay of Training Stages](https://arxiv.org/abs/2512.07783) finds large-pass@$k$ gains when RL data target the edge of competence and pre-training leaves headroom. In controlled settings with known data, [RL Grokking Recipe](https://arxiv.org/abs/2509.21016) observes acquisition and transfer of algorithmic coding strategies after a delayed transition, and [RL Post-Training Builds Compositional Reasoning Strategies](https://arxiv.org/abs/2607.07646) finds that RL consolidates primitive rewrite procedures into reusable sequential and parallel compositions. [From Reasoning Traces to Reusable Modules](https://arxiv.org/abs/2606.18089) formalizes a complementary account in which SFT supplies candidate modules and RL identifies and recombines them. [How Does RL Post-training Induce Skill Composition?](https://arxiv.org/abs/2512.01775) adds a structural limit: transfer depends on expression-tree shape, not only depth. These results are compatible if RL can recombine available procedural ingredients under some pre-training organizations without generally creating unconstrained new support.

Domain mixtures also introduce interference and diversity loss. [Can One Domain Help Others?](https://arxiv.org/abs/2507.17512) and [When Domains Interact](https://arxiv.org/abs/2602.01365) find asymmetric and order-sensitive transfer. [Does Math Reasoning Improve General LLM Capabilities?](https://arxiv.org/abs/2507.00432) finds that many reasoning-tuned models fail to transfer beyond math and links SFT failures to larger representation and output drift. [One-Shot RLVR](https://arxiv.org/abs/2504.20571) reports cross-category and post-saturation transfer from a single training example, with exploration playing a central role. [Decomposing Elements of Problem Solving](https://arxiv.org/abs/2505.22756) finds that GRPO mainly improves execution robustness while planning encounters a coverage wall. [When RL Suppresses Its Own Vocabulary](https://arxiv.org/abs/2605.29190) traces puzzle-to-math transfer to reasoning primitives, but also finds that vanilla RL suppresses exploration and backtracking primitives. [Exploration-Driven Optimization](https://arxiv.org/abs/2605.09853) likewise shows that preserving distributional diversity can improve the value of later self-consistency. The general principle is that RL should optimize not only answer reward, but also the policy support and reusable operations needed on shifted problems.

#### 3.3.3 Hybrid and self-improvement training

Self-improvement changes model parameters or training data using prior solutions and failures. [STEPS](https://arxiv.org/abs/2601.03676) constructs a skill taxonomy, synthesizes underrepresented skill combinations, and evaluates compositional transfer in downstream agent tasks. [CALM](https://arxiv.org/abs/2607.23771) trains language models with several inference controllers and tests held-out controller compositions. Both studies define what is learned during training and what combination changes at test time.

Persistent memory that changes future inference without updating base-model weights belongs to Section 4.4, not to Training. This distinction keeps parameter learning separate from system-level reuse of prior experience.

### 3.4 Training synthesis

| Training branch | Transfer mechanism | Strongest current evidence | Main risk |
| --- | --- | --- | --- |
| Pre-training exposure | Establish reusable capabilities before adaptation | Controlled training-stage and large-pass@$k$ comparisons | Opaque exposure makes the boundary hard to identify |
| Mid-training and curricula | Expose positions, procedures, or domains absent from base training | Long-context and verified-synthesis transfer | More data may repeat the same shortcut |
| SFT and distillation | Teach explicit operations and reachable traces | Checkpoint sweeps, cross-model transfer, and logically isomorphic OOD tests | Teacher forcing, late-stage overfitting, and suppressed uncertainty narrow support |
| RL and RLVR | Select and sometimes compose effective procedures with outcome or process feedback | Held-out domains, capability-controlled pass@$k$, and composed environments | Reward hacking, entropy collapse, domain interference, and dependence on base competence |
| Self-improvement and controller-aware training | Learn reusable skills or controller cooperation | Unseen skill combinations and held-out controller compositions | Training may specialize to the observed combination grammar |

Training results are strongest when they include a base-model control, an in-domain control, one or more isolated shifts, and a transfer matrix. A gain on a different benchmark is suggestive but not sufficient if the benchmark may share templates, sources, or answer styles with training.

## 4. Inference for generalization

Inference methods keep the base model fixed and change how a solution is elicited, searched, checked, or executed. They can improve **elicitation generalization** when a model contains a useful capability that ordinary decoding fails to access. They cannot be assumed to create a missing algorithm. Controlled shortest-path experiments find that inference-time scaling improves performance within the learned capability region but does not rescue recursive length failure ([Shortest Path Generalization, 2026](https://arxiv.org/abs/2604.15306)).

### 4.1 Prompt elicitation and problem decomposition

[Least-to-Most prompting](https://arxiv.org/abs/2205.10625) provides direct easy-to-hard evidence: demonstrations use easy subproblems, while evaluation includes harder compositions that ordinary chain-of-thought prompting does not solve reliably. It therefore belongs in this survey for its transfer design, not simply because it is a prompting method.

Recent work makes prompt intervention more targeted. [Constraint-First Reasoning](https://arxiv.org/abs/2608.05254) first extracts answer-space requirements and then checks the solution against them. Its cross-benchmark audit disables AIME-specific conventions and tests heterogeneous OlympiadBench formats, which separates a reusable constraint procedure from prompt fit to one answer convention. [Test-Time Hinting](https://arxiv.org/abs/2605.16410) routes examples to short hints derived from recurring VLM failures and evaluates transfer to unseen benchmarks and models.

Prompting changes the conditional distribution seen by the model. It should therefore be evaluated for robustness to paraphrase, demonstration order, label choice, and prompt length. A method that succeeds only with one instruction string improves prompt fit, not general reasoning transfer.

### 4.2 Sampling, search, and aggregation

Sampling and search exploit variation in a fixed model's solution distribution, but higher accuracy from more samples is not by itself evidence of generalization. The qualifying evidence must show that coverage, selection, or repair continues to work after a named shift. [Test-Time Scaling via Error Localization](https://arxiv.org/abs/2607.21453) supplies such evidence: a localizer learned for code repair retains valid prefixes and transfers to mathematics benchmarks without task-specific retraining. [Exploration-Driven Optimization](https://arxiv.org/abs/2605.09853) supplies a complementary training-side result by preserving policy diversity and then testing self-consistency on five OOD reasoning tasks.

These results distinguish three sources of gain:

- **Coverage:** more samples increase the chance that a correct path is generated.
- **Selection:** a verifier, vote, or reconciler identifies the better path.
- **Repair:** feedback localizes a defect and preserves valid work.

Reporting only final accuracy hides which source is active. A strong evaluation includes pass@$k$, selection accuracy conditional on a correct candidate, tokens or calls, and performance under the target shift.

### 4.3 Verification and transferred inference control

Verification and control qualify as generalization evidence only when the diagnostic or policy transfers beyond the setting used to construct it. [UPAIR](https://arxiv.org/abs/2607.17188) is training-free and evaluates a shared uncertainty-progress rule across three reasoning models and five cross-domain benchmarks. [Reasoning Errors Have a Region and a Direction](https://arxiv.org/abs/2608.05660) trains a residual-stream validity detector on reasoning benchmarks and reports answer-selection gains on unseen reasoning benchmarks. This is transfer of the detector, not evidence that the generator itself generalizes. Its factuality results are outside this survey's core claim. [On the Ability of Transformers to Verify Plans](https://arxiv.org/abs/2603.19954) gives theoretical and empirical conditions under which plan verification length-generalizes.

For adaptive computation, the required result is an accuracy-cost frontier on a shifted test distribution. Token savings or better stopping on the calibration benchmark alone establish efficiency, not reasoning generalization.

### 4.4 Tools, external memory, and test-time learning

Tools externalize operations that the model performs unreliably. Code execution, calculators, symbolic solvers, theorem provers, search, and environment feedback can turn an extrapolation problem into a sequence of locally checked steps. [To Infinity and Beyond](https://arxiv.org/abs/2510.14826) shows that tool use can unlock length generalization in state-space models. This is system-level generalization: the model learns when and how to delegate, while the tool supplies exact computation.

Memory methods qualify when stored experience is reused under a stated shift. [ReasoningBank](https://arxiv.org/abs/2509.25140) evaluates cross-task, cross-website, and cross-domain transfer on Mind2Web and a multi-website WebArena subset. [MILES](https://arxiv.org/abs/2607.06974) learns modular subgoal-instruction selection from confident samples and applies it to later uncertain reasoning states, including transfer across frozen model backbones. Its evidence concerns state and model transfer within mathematical reasoning, not cross-domain problem transfer. Both studies require controls for near-duplicate retrieval because reuse of a close solution is weaker evidence than recombination on an unfamiliar problem.

Some methods adapt during inference through gradients or repeated interaction. This blurs the phrase "fixed model." We reserve **fixed model** for unchanged base parameters, but allow an external controller, prompt, memory, or temporary adapter to change. Every study should state which state persists across examples, because persistent test-time learning can leak test information and change the evaluation distribution.

### 4.5 Inference synthesis

| Inference branch | Best suited for | Failure condition | Required control |
| --- | --- | --- | --- |
| Prompting and decomposition | Misaligned task representation or missing intermediate structure | Decomposition is wrong or prompt-sensitive | Matched prompts and paraphrase tests |
| Sampling and search | Correct paths exist with low but nonzero probability | No useful candidate is generated | Pass@$k$ and oracle-selection upper bound |
| Verification and repair | Errors are easier to detect than avoid | Verifier shares the generator's shortcut | Independent or execution-based checks |
| Transferred inference control | Difficulty varies across shifted examples | A monitor is calibrated only on its source benchmark | Source-target split and accuracy-cost frontier by shift |
| Tools and memory | Exact operations or reusable skills can be externalized | Tool routing or retrieval fails | Tool-free and retrieval-distance controls |

Inference is most effective when it changes the computational path in a way matched to the failure. Repeating the same biased decode, adding generic reflection, or increasing the token budget need not improve transfer.

## 5. Architecture for generalization

Architecture determines which computations are easy to represent, learn, and extend. Expressivity alone is not enough. A standard transformer may represent an algorithm in principle but converge to a shorter surface heuristic. Architectural work therefore aims to make reusable iteration, relative structure, modular composition, or exact state tracking the path favored by learning.

### 5.1 Recurrent depth and adaptive computation

The [Universal Transformer](https://arxiv.org/abs/1807.03819) introduced shared recurrent computation across depth. [Looped Transformers for Length Generalization](https://arxiv.org/abs/2409.15647) later showed that adaptive repeated application of a shared block can generalize on iterative algorithmic tasks. Sharing makes the same transition rule available beyond the depth used during training, which is a natural bias for longer reasoning.

Recent work expands this idea:

- [Loop, Think, and Generalize](https://arxiv.org/abs/2604.07822) studies systematic transfer and depth extrapolation in recurrent-depth transformers, while also finding overthinking at excessive recurrence.
- [Thinking Deeper, Not Longer](https://arxiv.org/abs/2603.21676) stabilizes 20 or more latent iterations through LayerScale, identity-biased recurrence, and final-output supervision.
- [Think Shallow, Solve Deep](https://arxiv.org/abs/2608.18222) links safe extra depth to whether the learned operator settles rather than drifts.
- [Stabilizing Extrapolation in Looped Transformers](https://arxiv.org/abs/2606.29983) uses stochastic stopping to reduce dependence between input length and loop count.
- [Equilibrium Reasoners](https://arxiv.org/abs/2605.21488) trains attractor-like computation, making convergence part of the reasoning design.
- [Universal Transformers for Circuit Computations](https://arxiv.org/abs/2608.31067) gives a small parameterization that learns depth-one circuit reduction and applies it repeatedly with autonomous halting.

Recurrence provides **compute extrapolation**, but not automatically **algorithm extrapolation**. If the shared update is wrong, applying it more often compounds the error. If training correlates length with iteration count, the model may learn a clock rather than a stopping rule. Strong recurrent designs therefore combine parameter sharing with stability objectives, varied loop counts, progress signals, or learned halting.

Recurrent computation is also useful beyond synthetic algorithms. [Looped Language Models Improve Compositional Tool Calling](https://arxiv.org/abs/2608.18171) reports larger gains on multi-step dependency-aware tool use than on isolated calls. [RELAY](https://arxiv.org/abs/2502.08482) uses a looped model as a teacher for autoregressive CoT. These works suggest that recurrence can serve either as the deployed architecture or as a generator of transferable training traces.

### 5.2 Position, attention, and locality

Length generalization is partly a problem of positional distribution shift. Learned absolute positions leave unseen indices unconstrained, while relative schemes can reuse a learned relation at new positions. [Relative Positions Generalize, Absolute Positions Memorize](https://arxiv.org/abs/2607.18759) gives an optimization account in a controlled retrieval setting: rotary features impose exact relative equivariance, while learned absolute encodings can pin to positions inside the training range. [How Data Shapes RoPE Frequency Usage](https://arxiv.org/abs/2607.07678) argues that selected frequencies match the dependency scales in the training data, and that test-time scaling works only when longer-context dependencies approximately dilate those scales.

Randomized positions can reduce train-test mismatch. [Random Float Sampling](https://arxiv.org/abs/2602.14050) exposes models to diverse continuous indices during training. [Randomized YaRN](https://arxiv.org/abs/2606.23687) couples randomized positional representations with a curriculum. These methods target the coordinate system, not the reasoning algorithm, so they are most effective when the main shift is positional.

Locality is another inductive bias. [On Locality and Length Generalization in Visual Reasoning](https://arxiv.org/abs/2607.09061) finds that global vision models exploit shortcuts while recurrent local policies generalize better on state-tracking tasks. As a conceptual comparison outside the LLM evidence base, [On the Mirage of Long-Range Dependency](https://arxiv.org/abs/2603.29069) shows that a two-dimensional multiplication representation lets a 321-parameter neural cellular automaton extrapolate far beyond its training range. It illustrates how representation changes locality, but it is not evidence about LLM reasoning generalization.

### 5.3 Modular, sparse, and neuro-symbolic structure

Modularity can isolate subproblems and make composition explicit. [What You Can't See Is What You Learn](https://arxiv.org/abs/2608.20054) finds that restricting modules to local evidence changes which compositional solution is learned, although its preregistered success threshold is narrowly missed and role information is a remaining confound. This supports modularity as a solution-selection bias, while leaving open whether the result scales beyond its controlled language-model society.

Neuro-symbolic systems externalize structure. [AGEL-Comp](https://arxiv.org/abs/2604.26522) combines a causal program graph, inductive logic programming, and neural theorem verification for interactive compositional tasks. [Reasoners or Translators?](https://arxiv.org/abs/2605.16052) finds that symbolic inference over formalized tax law can be more robust to unseen rule and case variants than monolithic generation. These systems can improve reliability when formalization is correct, but parsing natural language into symbols becomes the new generalization bottleneck.

[Rational Transductors](https://arxiv.org/abs/2602.07599) provide a different hybrid: matrix-valued recurrence from weighted finite automata is injected into attention to support regular languages and sequential state tracking. The architecture is relevant because it directly tests length transfer on algorithmic reasoning tasks, rather than reporting only generic language or vision accuracy.

### 5.4 Latent recurrence, energy minimization, and alternative representations

Autoregressive CoT expands computation by producing visible tokens. Latent architectures instead iterate or optimize internal states. [Unlocking OOD Generalization via Recursive Latent Space Reasoning](https://arxiv.org/abs/2510.14095) and [Generalizable Reasoning through Compositional Energy Minimization](https://arxiv.org/abs/2510.20607) represent two forms of this idea: repeated latent transformation and iterative minimization of a compositional objective. Their appeal is that computation can scale without requiring a natural-language token for every step.

For generalization, the decisive test is whether the same latent update remains effective at unseen composition or depth, not whether its internal state is easy to interpret. Evidence on frontier-scale models remains limited.

[Barriers to Universal Reasoning](https://arxiv.org/abs/2604.25800) shows that representational design also matters for visible CoT. Under its formal assumptions, standard finite-alphabet CoT does not obtain unrestricted length-general learnability. Signpost tokens and change-only logs overcome repeated copying and last-occurrence retrieval in the construction. This result illustrates a recurring theme: a scratchpad helps when its representation decomposes the target computation into locally learnable updates.

### 5.5 Architecture synthesis

| Structural property | Intended benefit | Supporting evidence | Open limitation |
| --- | --- | --- | --- |
| Shared recurrent depth | Reuse one transition for more reasoning steps | Algorithmic depth transfer and compositional tool use | Stability and halting under much deeper inference |
| Relative or randomized position | Reuse relations at unseen indices | Retrieval theory and long-context evaluations | Does not fix a wrong reasoning algorithm |
| Local computation | Block global shortcuts and align with local updates | Visual state tracking; cellular automata provide a non-LLM comparison | Requires a task-suitable representation |
| Modular routing | Recombine reusable subroutines | Masked modules and controller composition | Modules may specialize to training roles |
| Neuro-symbolic execution | Exact state updates and verifiable constraints | Planning, legal rules, and interactive agents | Semantic parsing remains brittle |
| Latent or equilibrium computation | Add iterative compute without long visible traces | Recurrent and attractor-based reasoners | Harder oversight and limited large-scale causal evidence |

The architecture literature provides the clearest gains on tasks with known iterative, algebraic, or local structure. The main unresolved question is whether these biases can be combined with the broad language competence and optimization stability of large autoregressive models.

## 6. Analysis of generalization

Analysis asks two different questions. Behavioral studies describe where transfer succeeds or fails. Mechanistic and theoretical studies explain why. The distinction prevents three common errors: treating a benchmark score as evidence of an algorithm, treating a probe as a causal explanation, and applying a theorem beyond its assumptions.

## 6.1 Behavioral observations

### 6.1.1 Generalization is a vector, not one score

A model can generalize along one axis and fail along another. In a controlled shortest-path setting, models transfer to unseen maps but fail when paths require more recursive steps ([Shortest Path Generalization, 2026](https://arxiv.org/abs/2604.15306)). In visual planning, text representations and mixed trace formats can transfer better than image inputs when map size changes ([OOD Visual Planning, 2026](https://arxiv.org/abs/2602.15460)). In arithmetic, structurally equivalent word problems respond differently to small number remappings across datasets ([Numeric-Remapping Attacks, 2026](https://arxiv.org/abs/2606.03606)). A single OOD average hides these distinctions.

The same task can also support several notions of novelty. New instances test interpolation within a generator. New symbol names test invariance. New combinations test systematicity. Greater depth tests whether a transition can be repeated. New domains test whether a procedure survives different semantics and background knowledge. These shifts should be reported separately before any aggregate is computed.

### 6.1.2 Surface invariance is necessary but incomplete

Surface changes are attractive because they preserve the underlying answer while removing familiar cues. [GRAIN](https://arxiv.org/abs/2608.27142) varies graph node names and narrative form. [Shared Circuits](https://arxiv.org/abs/2609.04463) compares numeric and verbal arithmetic across languages. [When Symbol Names Should Not Matter](https://arxiv.org/abs/2605.07120) studies classification with fresh vocabularies. These studies show that symbol and format invariance cannot be assumed even for simple operations.

Passing a surface test does not establish deeper transfer. A model may normalize several familiar formats into a common representation while still failing at longer composition or new rules. Evaluation should therefore cross surface shifts with at least one structural shift.

### 6.1.3 Composition weakens as depth and interaction grow

Compositional failure appears in language, math, science, graphs, and tools. [XDomainBench](https://arxiv.org/abs/2605.14754) reports a systematic collapse as scientific composition order rises and interactive errors accumulate. [Why Knowing Both Hops Is Not Enough](https://arxiv.org/abs/2608.07261) shows that possession of the two atomic facts does not ensure two-hop transfer. [Looped Language Models Improve Compositional Tool Calling](https://arxiv.org/abs/2608.18171) finds larger recurrent gains for dependency-aware multi-step calls than for isolated calls, suggesting that composition, not API syntax alone, is the limiting factor. Controlled post-training studies refine this pattern: [RL Post-Training Builds Compositional Reasoning Strategies](https://arxiv.org/abs/2607.07646) observes reuse of learned sequential and parallel rewrite procedures, while [Skill Composition on Countdown](https://arxiv.org/abs/2512.01775) finds persistent failure on particular expression-tree shapes even when depth is held constant.

Compositional benchmarks must guard against template coverage. If every test composition has a close training analogue, high accuracy may reflect interpolation over combinations. Prospectively sealed generators, disjoint primitive-combination splits, and collision audits provide stronger evidence.

### 6.1.4 Length, depth, and difficulty are different shifts

Length can mean more tokens, more entities, more reasoning steps, or a longer output. These variables often correlate in training but require different mechanisms. [Exploring Length Generalization](https://arxiv.org/abs/2207.04901) showed that ordinary fine-tuning is weak on longer algorithmic inputs and that scratchpads can help. [What Algorithms Can Transformers Learn?](https://arxiv.org/abs/2310.16028) links empirical extrapolation to compact RASP programs. [Length Generalization Transfer](https://arxiv.org/abs/2506.09251) asks whether extrapolation learned on one task transfers to another.

Difficulty is not a reliable proxy for length. [Revisiting Generalization Across Difficulty Levels](https://arxiv.org/abs/2511.21692), published at EACL 2026, estimates example difficulty from many model responses and finds that neither easy-only nor hard-only training consistently transfers across the full range. [Rethinking Easy-to-Hard](https://arxiv.org/abs/2603.27226) likewise finds no robust advantage for curriculum order on controlled deductive tasks. A valid curriculum claim must define difficulty for the model and separate sample selection from presentation order.

### 6.1.5 Reasoning gains remain tied to priors and representation

Reasoning-augmented inference helps most when the model has useful knowledge or subroutines to work with. [Disentangling Generalization and Memorization Using Chess](https://arxiv.org/abs/2601.16823) finds that performance and the marginal value of extra reasoning decline when relevant priors are sparse. [Improving Latent Generalization Using Test-Time Compute](https://arxiv.org/abs/2604.01430) improves deduction over stored knowledge but still finds reversal tasks difficult. [On the Emergence and Test-Time Use of Structural Information](https://arxiv.org/abs/2601.17869) reports that learning abstract structure correlates with complex reasoning while compositional use remains limited.

This evidence argues against a strict recall-versus-reasoning split. Generalization often requires both suitable primitives and a procedure that can compose them. Tests should vary familiarity of the primitives independently from novelty of the composition.

RLVR evidence supports the same qualification. [Generalization of RLVR Using Causal Reasoning](https://arxiv.org/abs/2512.20760) finds that transfer across probabilistic query levels and graph complexity emerges only above a threshold of initial reasoning competence. [Does RL Really Incentivize Reasoning Capacity?](https://arxiv.org/abs/2504.13837) shows that small-$k$ improvements can coexist with no expansion, or even a contraction, of the large-pass@$k$ boundary. Base sampling support and deployment budget must therefore be measured before an RL result is interpreted as new capability.

### 6.1.6 Contamination-resistant and capability-controlled evaluation

Opaque pre-training data makes it difficult to know whether a benchmark is truly unseen. Several studies reduce this uncertainty with generated rules, fictional facts, new symbol systems, or new languages. [EsoLang-Bench](https://arxiv.org/abs/2603.09678) translates equivalent programming problems into esoteric languages that are unlikely to occur at useful scale in training. [Reasoners or Translators?](https://arxiv.org/abs/2605.16052) combines contamination tests with rule and case variations in tax law. [Beyond Memorization](https://arxiv.org/abs/2601.13392) compares factual and public construction tasks against handcrafted and systematically generated automata tasks.

These controls trade realism for identification. Synthetic tasks provide exact rules and clean splits but may favor architectures aligned with the generator. Natural tasks provide ecological value but have uncertain exposure. A strong survey conclusion should rely on convergence across both.

### 6.1.7 Representative 2026 diagnostic suite

| Diagnostic | Primary shift | Main observation | Interpretation limit |
| --- | --- | --- | --- |
| [General365](https://arxiv.org/abs/2604.11778) | Domain and variant | Broad reasoning remains below specialized math performance | Variant construction may not remove all model familiarity |
| [XDomainBench](https://arxiv.org/abs/2605.14754) | Cross-domain composition | Failure grows with composition order and interaction | Scientific knowledge and reasoning are partly entangled |
| [Numeric-Remapping Attacks](https://arxiv.org/abs/2606.03606) | Surface-preserving numeric change | Robustness differs sharply by dataset structure | Covers arithmetic word problems only |
| [Elenchos](https://arxiv.org/abs/2607.12733) | Abductive rule mutation | Detection is stronger than causal attribution | Formal systems are controlled but narrow |
| [OOD Visual Planning](https://arxiv.org/abs/2602.15460) | Map size and representation | CoT helps ID results more than most OOD settings | Simple navigation does not span open-world planning |
| [Chess generalization](https://arxiv.org/abs/2601.16823) | Density of useful priors | Performance falls as relevant priors become sparse | Training exposure is inferred rather than observed |
| [OMEGA](https://arxiv.org/abs/2506.18880) | Mathematical novelty type | Separates exploratory, compositional, and transformative transfer | The novelty taxonomy still depends on task construction |
| [CoT as a Mirage?](https://arxiv.org/abs/2508.01191) | Data distribution | Apparent reasoning depends strongly on distribution design | Behavioral evidence does not identify a unique mechanism |

## 6.2 Empirical mechanistic analysis

Empirical mechanistic analysis seeks internal variables that predict and causally affect transfer. The strongest designs combine a controlled task, a localized representation or circuit, and an intervention that changes OOD behavior while preserving relevant controls.

### 6.2.1 Shared circuits and reusable computation

[Shared Circuits Predict Cross-Format Generalization](https://arxiv.org/abs/2609.04463) localizes arithmetic circuits independently for numeric and verbal formats, then shows that overlap with the numeric circuit predicts relative format difficulty, model-level transfer, and item-level correctness. The result supports a **circuit reuse** account: a format generalizes when it accesses computation already used by a mastered format.

[Finite State Automata Inside Transformers with CoT](https://arxiv.org/abs/2502.20129) identifies state-tracking computations in small transformers. [Discovering Interpretable Algorithms by Decompiling Transformers to RASP](https://arxiv.org/abs/2602.08857) reparameterizes a transformer as a RASP program and applies causal pruning to extract a sufficient subprogram. These works provide stronger algorithmic evidence than output accuracy because the proposed computation can be inspected and intervened on.

Circuit reuse can still be local. [Is Grokking Worthwhile?](https://arxiv.org/abs/2601.09049) finds that a mature generalization circuit has limited transfer when new knowledge is integrated. A circuit that supports one data regime should not be assumed to be a portable reasoning module.

### 6.2.2 Representation alignment and the knowing-using gap

Several studies explain failure as a mismatch between representations needed at consecutive steps. [Why Knowing Both Hops Is Not Enough](https://arxiv.org/abs/2608.07261) finds that successful two-hop transfer depends on consistent intermediate entity representations, while upper layers can learn output mappings that do not reuse lower-layer reasoning states. [Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize](https://arxiv.org/abs/2607.08393) uses self-patching to show that newly memorized facts may be present but routed to layers where they cannot support downstream computation.

[Representational Homomorphism](https://arxiv.org/abs/2601.18858) measures whether hidden-state composition respects the algebra of the input expressions. Its homomorphism error predicts OOD SCAN performance and regularizing the error improves transfer in the controlled setting.

The shared principle is **interface compatibility**. A model may encode every required fact but fail if one layer, module, language, or step expresses an intermediate in a form the next step cannot use. This connects mechanistic analysis to training methods that align pivots, views, and loop steps.

### 6.2.3 Learning dynamics, grokking, and simplicity bias

Grokking is delayed generalization after training accuracy has saturated. [Grokking of Hierarchical Structure](https://arxiv.org/abs/2305.18741) observes delayed hierarchical transfer and a non-monotonic relation with model depth. [Grokked Transformers Are Implicit Reasoners](https://arxiv.org/abs/2405.15071) finds delayed in-distribution rule use without reliable systematic transfer to new atomic facts. The two results show that delayed generalization and systematicity are not identical.

Recent work makes the dynamics more precise. [Critical Windows of Complexity Control](https://arxiv.org/abs/2605.04396) finds a narrow training interval in which weight decay strongly changes whether a model selects reasoning or memorization on a controlled task. [Early-Warning Signals of Grokking](https://arxiv.org/abs/2602.16967) identifies a commutator-defect signal before generalization and uses interventions on gradient non-commutativity to speed up or delay grokking. [Complexity Control Facilitates Reasoning-Based Compositional Generalization](https://arxiv.org/abs/2501.08537) provides the preceding evidence that initialization and regularization can select a lower-complexity solution.

These findings support an optimization account: generalization is not only about whether a good circuit exists but also which basin training reaches and when. The results remain concentrated in small models and generated tasks, so transfer to frontier-scale pre-training is a hypothesis rather than an established fact.

Post-training introduces an analogous temporal effect at larger scales. [RL Fine-Tuning Heals OOD Forgetting](https://arxiv.org/abs/2509.12235) finds that SFT first improves and then erodes OOD reasoning, while subsequent RL restores part of the lost capability; the transition correlates with singular-vector rotation rather than large singular-value changes. [When RL Fails after SFT](https://arxiv.org/abs/2606.09932) links over-trained SFT checkpoints to overconfident outputs, sharper parameter landscapes, and reduced RL plasticity. [Why Does Self-Distillation Degrade Reasoning?](https://arxiv.org/abs/2603.24472) identifies another trajectory-level variable, epistemic verbalization, whose suppression accompanies OOD decline. Together, these studies show that the order and duration of post-training stages change the reachable transfer regime.

### 6.2.4 Faithful traces and shortcut formation

Visible reasoning is evidence about a generated trajectory, not direct access to the model's causal computation. [Protoreasoning in Tiny Transformers](https://arxiv.org/abs/2608.04980) shows that contentful traces, rather than extra tokens alone, close part of the OOD gap on Dyck tasks. This supports the claim that scratchpads help when their states encode useful intermediate computation. It does not imply that natural-language CoT is always faithful.

A mechanistic claim about generalization should test the proposed trace or state intervention on the same shifted split used to establish transfer. Probe accuracy or trace plausibility alone does not show that the decoded information causes the OOD answer.

## 6.3 Theoretical analysis

Theory clarifies which kind of claim is being made. **Expressivity** asks whether parameters exist that solve a task. **Learnability** asks whether training can find such parameters from finite data. **Length generalization** asks whether the learned rule remains correct at unseen sizes. **Optimization theory** asks which fitting solution is selected. Results at one level do not automatically imply the next.

### 6.3.1 Scratchpads and length-generalizable computation

[How Far Can Transformers Reason?](https://arxiv.org/abs/2406.06467) studies learnability through a globality barrier and finds that educated, inductive scratchpads can reduce the effective globality of intermediate targets and improve OOD length transfer. [Barriers to Universal Reasoning](https://arxiv.org/abs/2604.25800) shows that standard finite-alphabet CoT remains restricted under a stricter length-generalizable learnability setting, while a growing vocabulary with signpost tokens and change logs supports a Turing-machine simulation. Together, these results explain why the representation of a scratchpad matters as much as its existence. Pure CoT expressivity results are outside the core corpus unless they also establish learnability or transfer at unseen lengths.

### 6.3.2 RASP, regular languages, and algorithm classes

[What Algorithms Can Transformers Learn?](https://arxiv.org/abs/2310.16028) proposes that tasks with short length-general RASP programs are the ones transformers tend to extrapolate on. [Algebraic Decomposition Theory for Transformer Length Generalization](https://arxiv.org/abs/2608.13433) gives a complete characterization of the regular languages expressible in C-RASP and a polynomial-time decision procedure based on an algebraic decomposition. [Discovering Interpretable Algorithms](https://arxiv.org/abs/2602.08857) complements this formal line by extracting small RASP programs from trained models.

[Length Generalization Bounds for Transformers](https://arxiv.org/abs/2603.02238) proves that computable bounds do not exist for general C-RASP already at two layers, while positive C-RASP and fixed-precision transformers admit exponential bounds. The result warns against finite test suites being treated as proof of unbounded extrapolation.

[On the Ability of Transformers to Verify Plans](https://arxiv.org/abs/2603.19954) extends the formal language framework to settings where both sequence length and the effective input alphabet grow, and identifies planning domains with learnable length-general verification. This distinction between generating a plan and verifying one supports inference systems that separate candidate production from checking.

### 6.3.3 Optimization, equivariance, and structural risk

[Transformers Provably Learn CoT Reasoning with Length Generalization](https://arxiv.org/abs/2511.07378) links state-tracking structure, attention concentration, and gradient-based learning, and studies recursive self-training as a way to extend solvable lengths. [Relative Positions Generalize, Absolute Positions Memorize](https://arxiv.org/abs/2607.18759) explains a length-transfer gap through implicit bias: relative rotary structure enforces an equivariant rule, while unseen absolute positions are unconstrained.

[A Measure-Theoretic Analysis of Reasoning](https://arxiv.org/abs/2605.19944) uses Wasserstein shift, Lipschitz structure, and approximation limits to connect positional invariance and circuit depth to OOD risk on combinatorial search. [When Symbol Names Should Not Matter](https://arxiv.org/abs/2605.07120) derives margin-transfer guarantees for fresh-symbol classification in a transformer-kernel regime, with failures controlled by token-collision structure. These works formalize two recurring empirical factors: invariance to nuisance variables and sufficient computational depth.

[Learning to Reason with Curriculum II](https://arxiv.org/abs/2606.27721) analyzes compositional state tracking under two explicit supervision models. With intermediate-state feedback, recursive decomposition reduces the token requirement relative to direct simulation; with verifier-only feedback, it replaces full-length reference coverage with a weaker short-block coverage condition. The theorem does not establish that ordinary LLM training finds the required decomposition, but it makes precise how curricula can reduce the coverage burden rather than merely reorder examples.

### 6.3.4 Architecture-level guarantees and limits

[Universal Transformers for Circuit Computations](https://arxiv.org/abs/2608.31067) provides a constructive recurrent transformer that reduces one circuit level per iteration, halts after the required depth, and learns from shallow examples in its controlled setting. [Rational Transductors](https://arxiv.org/abs/2602.07599) augment attention with weighted-automata recurrence and establish expressivity for regular languages and selected harder problems. These results show how adding a state transition with an algebraic interpretation can close gaps left by shallow self-attention.

Such guarantees depend on input encoding, precision, attention type, training distribution, or well-formedness assumptions. They establish useful design targets, not direct guarantees for an unrestricted natural-language LLM.

### 6.3.5 What theory currently agrees on

Across different formal models, five points recur:

1. A transformer can represent more computations than gradient-based learning will reliably select from short data.
2. Intermediate steps help when they reduce a global computation into local, reusable transitions.
3. Relative structure and equivariance constrain behavior at unseen positions or symbol choices.
4. Recurrent application of a shared transition is a natural route to variable computational depth.
5. Finite success does not prove unbounded length generalization, and general computable certification may be impossible.

The open theoretical task is to connect these controlled results to realistic pre-trained models whose data distribution, precision, internal algorithms, and optimization history are only partly known.

## 6.4 Cross-pillar synthesis

The four pillars interact through a common pipeline:

1. **Training supplies primitives and selects a candidate procedure.** Data coverage, supervision targets, reward shape, regularization, and optimization geometry decide whether the model learns a reusable rule or a shortcut.
2. **Architecture defines the low-cost procedures.** Recurrence, position, locality, modularity, and external symbolic state make some rules easier to learn and repeat.
3. **Inference allocates and controls computation.** Prompts, search, verification, memory, and tools choose how the available procedure is used on one example.
4. **Analysis tests the transfer claim.** Controlled shifts identify behavior, causal experiments test mechanisms, and theory states guarantees under assumptions.

This view explains several apparent conflicts. CoT can improve length transfer when it exposes an inductive local update, but generic long traces can fail because added tokens do not supply the right state. SFT can teach reusable modules early and then lose OOD performance through continued fitting. RL can recover those modules, compose them, or favor invariant steps, but its benefit depends on initial competence and it can narrow the sampling distribution or suppress useful exploratory primitives. Recurrence can extrapolate when the shared operator is stable, but extra loops hurt when the operator drifts. Tools can solve longer instances, but the result belongs to the system unless the base model itself learns the algorithm.

## 6.5 Recommended evaluation protocol

A reasoning-generalization paper should report the following minimum design.

### 6.5.1 Define the boundary

- Name the training stages: pre-training, continued or mid-training, SFT, preference or RL training, and test-time adaptation.
- State what is known about exposure to test tasks, templates, and documents.
- Define the unit that is fixed: model weights, prompt, controller, verifier, memory, retriever, and tools.

### 6.5.2 Factor the shift

- Include an ID split and at least one controlled OOD split.
- Change one shift axis at a time where possible.
- Cross a surface shift with a structural shift to rule out simple normalization.
- Report distance or difficulty continuously when the generator permits it, not only near and far bins.

### 6.5.3 Separate generation from selection

- Report greedy or single-sample accuracy.
- For search, report pass@$k$, selected accuracy, oracle selection, number of calls, tokens, wall time, and tool cost.
- For adaptive methods, report calibration and the full accuracy-cost frontier.
- For verifiers, report false rejection of correct paths and false acceptance of incorrect paths under shift.

### 6.5.4 Test mechanism and robustness

- Compare semantic-preserving perturbations, counterfactual rule changes, and new compositions.
- Use multiple model families, sizes, seeds, and training checkpoints.
- For internal explanations, include necessity, sufficiency, placebo, and off-target controls.
- For architecture claims, match parameter count, training compute, data, and inference compute where possible.

### 6.5.5 Report a generalization matrix

The most informative summary is a matrix with training distributions as rows and test shifts as columns. Each cell should include performance and cost. A method that improves only the diagonal is specialized. A method that improves several off-diagonal cells provides evidence of transfer. Confidence intervals or paired tests should be given for the comparisons used to support the main claim.

## 6.6 Open problems

### 6.6.1 Generalization beyond unknown pre-training data

Generated symbols and sealed tasks reduce contamination but do not fully match natural reasoning. A central need is a benchmark process that is created after model training, uses auditable generators, and still requires realistic knowledge and interaction.

### 6.6.2 Transfer across several simultaneous shifts

Most controlled work changes one factor. Deployed systems face new domains, languages, tools, and difficulty at once. Factorial evaluation is needed to determine whether gains compose or interfere.

### 6.6.3 Portable reasoning modules

Circuits, memories, controllers, and distilled skills often transfer within one model family or task cluster. It remains unclear which representation can serve as a stable interface across model versions and modalities.

### 6.6.4 Learning when more computation helps

Most inference controllers are calibrated and evaluated on closely related data. A stronger test would fit the controller on one set of reasoning families or difficulty ranges, then evaluate its stopping, branching, retrieval, or tool choices on unseen families and shifts. The result should report target accuracy and cost jointly; saving tokens on the source distribution is not evidence of reasoning generalization.

### 6.6.5 Bridging small-model mechanisms and frontier models

Small controlled transformers permit causal analysis and exact generators. Frontier models provide the capabilities of interest but have opaque data and expensive interventions. Work that reproduces the same mechanism across these scales would make mechanistic conclusions more credible.

### 6.6.6 Generalization and faithfulness together

A model may reach the correct OOD answer through an invalid shortcut, while a faithful but incomplete process may receive no credit. Evaluation needs outcome, process validity, counterfactual sensitivity, and calibration in the same test.

### 6.6.7 Unified training-inference design

Training often optimizes one decoding procedure, while deployment uses another. Controller-aware training, entropy-preserving RL, and search-generated data point toward joint design. The unresolved question is how to train for a family of unseen inference procedures without making the model dependent on one controller.

## 6.7 Limitations of this survey

This review prioritizes papers available by 8 September 2026 and is dominated by arXiv, ACL, and OpenReview-accessible work. Very recent 2026 preprints have not received the same peer review as conference papers, so their quantitative claims should be treated as reported evidence rather than settled results. The manifest records publication and update dates to support later revision.

The inclusion rule requires an explicit reasoning-related shift. This excludes many papers on broad robustness, domain adaptation, long context, and agent transfer when the reasoning component cannot be separated. As a result, the survey should not be read as a complete review of generalization in all language-model applications.

Taxonomy assignments identify each paper's primary role, but many methods span pillars. Randomized positions are both training and architecture; controller-aware post-training links training and inference; neuro-symbolic agents combine architecture and tools. The manifest keeps one primary label for readability, while the prose describes these links.

Finally, several mechanistic and theoretical results use small models, synthetic tasks, hard or fixed-precision attention, known generators, or special encodings. These results identify mechanisms and limits under controlled assumptions. They do not by themselves establish how a frontier model trained on internet-scale data reasons.

## 6.8 Conclusion

Reasoning generalization is not a single capability that rises automatically with scale or benchmark accuracy. The evidence supports a conditional account: transfer improves when learning selects reusable operations, architecture makes those operations easy to repeat, inference allocates compute to the observed failure, and evaluation isolates the target shift. The practical next step is to replace broad OOD claims with generalization matrices that name the training boundary, shift axis, system components, cost, and evidence level.
