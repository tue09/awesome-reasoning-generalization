<div align="center">

# Awesome Reasoning Generalization

### Beyond the Training Distribution: A Survey of Reasoning Generalization in Large Language Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Papers](https://img.shields.io/badge/papers-233-6f42c1)
![2024 papers](https://img.shields.io/badge/2024%20papers-34-yellow)
![2025 papers](https://img.shields.io/badge/2025%20papers-68-orange)
![2026 papers](https://img.shields.io/badge/2026%20papers-110-red)
[![GitHub last commit](https://img.shields.io/github/last-commit/tue09/awesome-reasoning-generalization?logo=github&color=blue)](https://github.com/tue09/awesome-reasoning-generalization/commits/main)

</div>

> **Status:** Updated on 7 October 2026 to follow the latest version of the survey (literature search through 10 September 2026). The list contains 233 studies organized by the survey's taxonomy: 208 discussed in the survey and 25 more from the same screening. By first arXiv posting, 110 are from 2026, 68 from 2025, 34 from 2024, and 21 are earlier foundations.

Progress in LLM reasoning is commonly measured on benchmarks that stay close to the training distribution. This list collects work on whether the reasoning procedures that large language models acquire still hold when notation, problem structure, length, difficulty, domain, language, or tool interface changes, and on what training, inference, and architecture contribute to that transfer. The companion text is in [survey.md](survey.md).

## Contents

- [Reading a Generalization Claim](#reading-a-generalization-claim)
- [Taxonomy](#taxonomy)
- [Paper List](#paper-list)
  - [Training for Generalization](#training-for-generalization)
    - [Pre-Training](#pre-training): [Data Curation](#data-curation), [Optimizer Design](#optimizer-design)
    - [Mid-Training](#mid-training): [Continued Pre-Training](#continued-pre-training), [Intermediate Fine-Tuning](#intermediate-fine-tuning)
    - [Post-Training: Supervised Fine-Tuning](#post-training-supervised-fine-tuning): [Problem-Level Methods](#problem-level-methods), [Solution-Level Methods](#solution-level-methods), [Objective-Level Methods](#objective-level-methods)
    - [Post-Training: Reinforcement Learning](#post-training-reinforcement-learning): [Scope and Limits of RL Transfer](#scope-and-limits-of-rl-transfer), [Environment Design](#environment-design), [Reward Design](#reward-design), [Policy Optimization](#policy-optimization)
    - [Post-Training: Hybrid Training](#post-training-hybrid-training): [Sequential Training](#sequential-training), [Joint Training](#joint-training)
  - [Inference for Generalization](#inference-for-generalization)
    - [Trajectory Restructuring](#trajectory-restructuring): [Structural Decomposition](#structural-decomposition), [Upfront Structural Injection](#upfront-structural-injection)
    - [Compute Scaling](#compute-scaling): [Search Expansion and Allocation](#search-expansion-and-allocation), [Targeted Revision](#targeted-revision)
    - [State Assessment](#state-assessment): [Internal State Assessment](#internal-state-assessment), [Verifier-Mediated Assessment](#verifier-mediated-assessment)
    - [Inference-Time Externalization](#inference-time-externalization): [Within-Episode Externalization](#within-episode-externalization), [Cross-Episode Externalization](#cross-episode-externalization)
  - [Architecture for Generalization](#architecture-for-generalization)
    - [Recurrent Depth](#recurrent-depth): [Weight Sharing](#weight-sharing), [Adaptive Computation](#adaptive-computation), [Stable Recurrent Dynamics](#stable-recurrent-dynamics)
    - [Information Routing](#information-routing): [Positional Encoding](#positional-encoding), [Attention Patterns](#attention-patterns), [Modular Reasoning](#modular-reasoning)
    - [Recurrent Memory](#recurrent-memory): [Token-Level Recurrence](#token-level-recurrence), [Chunk-Level Recurrence](#chunk-level-recurrence)
  - [Analysis of Generalization](#analysis-of-generalization)
    - [Behavioral Evidence](#behavioral-evidence): [Compositional Generalization](#compositional-generalization), [Length, Depth, and Difficulty](#length-depth-and-difficulty), [Prior Exposure](#prior-exposure)
    - [Mechanistic Evidence](#mechanistic-evidence): [Shared Circuits](#shared-circuits), [Interfaces Between Steps](#interfaces-between-steps), [Learning Dynamics](#learning-dynamics)
    - [Theoretical Evidence](#theoretical-evidence): [Learnability](#learnability), [Certification](#certification)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Related Surveys](#related-surveys)
- [Citation](#citation)

## Reading a Generalization Claim

A reasoning-generalization result can be interpreted only when four elements are explicit.

| Element | Question |
| --- | --- |
| **Task and fixed unit** | What is solved, and which components stay constant: weights, prompt, controller, verifier, memory, or tools? |
| **Reference exposure** | Which stage defines "seen": pre-training, mid-training, SFT or distillation, RL, or test-time adaptation? |
| **Test shift** | Which axis moves: instance, surface form, composition, length or depth, difficulty, task or domain, language or modality, environment or tool, or required knowledge? |
| **Evidence** | How is transfer shown: behavioral outcomes, representational associations, causal interventions, or formal results under explicit assumptions? |

Claims also differ in what is held fixed. *Model-level* generalization concerns the learned parameters under a fixed decoding procedure; *procedure-level* generalization adds prompting, search, or another inference algorithm; *system-level* generalization further adds retrievers, verifiers, memories, tools, and environments. A tool-augmented system can solve longer problems even when the base model has not learned a length-general algorithm.

## Taxonomy

| Pillar | Central question | Branches |
| --- | --- | --- |
| **Training for Generalization** | How can parameter updates make a learned reasoning procedure survive shifts in form, composition, length, difficulty, domain, or language? | Pre-training, mid-training, and post-training (SFT, RL, hybrid) |
| **Inference for Generalization** | How can a fixed model be deployed on unfamiliar problems by changing computation at test time, and where does that stop helping? | Trajectory restructuring, compute scaling, state assessment, externalization |
| **Architecture for Generalization** | Which computational structures let a learned operation be reused at greater length, depth, or compositional complexity? | Recurrent depth, information routing, recurrent memory |
| **Analysis of Generalization** | When does reasoning transfer, and what explains its successes and failures? | Behavioral, mechanistic, and theoretical evidence |

<p align="center">
  <a href="assets/main_taxonomy.pdf"><img src="assets/main_taxonomy.svg" width="100%" alt="Taxonomy of reasoning generalization in large language models"></a>
</p>

The first three pillars are interventions, ordered by where they act: on parameters, on test-time computation with the parameters fixed, and on the model's computational structure. Training is organized by stage and, within each stage, by the component a method modifies. The fourth pillar separates behavioral, mechanistic, and theoretical evidence. Each paper appears once, under its primary source of generalization.

## Paper List

### Training for Generalization

*How can parameter updates make a learned reasoning procedure survive shifts in form, composition, length, difficulty, domain, or language?*

#### Pre-Training

*Pre-training sets the reasoning primitives that later stages can recombine but can hardly supply.*

##### Data Curation

> Control what the corpus contains: coverage of the required primitives and the share of facts that must be inferred rather than recalled.

- **On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models**. *Charlie Zhang, Graham Neubig, Xiang Yue*. [[Paper]](https://arxiv.org/abs/2512.07783) [[PDF]](https://arxiv.org/pdf/2512.07783) ![](https://img.shields.io/badge/year-2025-orange)
- **Grokking in the Wild: Data Augmentation for Real-World Multi-Hop Reasoning with Transformers**. *Roman Abramov, Felix Steinbauer, Gjergji Kasneci*. [[Paper]](https://arxiv.org/abs/2504.20752) [[PDF]](https://arxiv.org/pdf/2504.20752) ![](https://img.shields.io/badge/year-2025-orange)
- **Teaching Transformers Causal Reasoning through Axiomatic Training**. *Aniket Vashishtha, Abhinav Kumar, Atharva Pandey, Abbavaram Gowtham Reddy, Kabir Ahuja, Vineeth N Balasubramanian, Amit Sharma*. [[Paper]](https://arxiv.org/abs/2407.07612) [[PDF]](https://arxiv.org/pdf/2407.07612) ![](https://img.shields.io/badge/year-2024-yellow)

##### Optimizer Design

> Change how the corpus is fit, since solutions with equal training loss can behave differently out of distribution.

- **Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima**. *Huanran Chen, Huaqing Zhang, Xiao Li, Yinpeng Dong, Ke Shen, Jun Zhu*. [[Paper]](https://arxiv.org/abs/2604.09258) [[PDF]](https://arxiv.org/pdf/2604.09258) ![](https://img.shields.io/badge/year-2026-red)
- **Complexity Control Facilitates Reasoning-Based Compositional Generalization in Transformers**. *Zhongwang Zhang, Pengxiao Lin, Zhiwei Wang, Yaoyu Zhang, Zhi-Qin John Xu*. [[Paper]](https://arxiv.org/abs/2501.08537) [[PDF]](https://arxiv.org/pdf/2501.08537) ![](https://img.shields.io/badge/year-2025-orange)
- **Initialization is Critical to Whether Transformers Fit Composite Functions by Reasoning or Memorizing**. *Zhongwang Zhang, Pengxiao Lin, Zhiwei Wang, Yaoyu Zhang, Zhi-Qin John Xu*. [[Paper]](https://arxiv.org/abs/2405.05409) [[PDF]](https://arxiv.org/pdf/2405.05409) ![](https://img.shields.io/badge/year-2024-yellow)

#### Mid-Training

*An intermediate stage before task adaptation, so that post-training does not start out of distribution.*

##### Continued Pre-Training

> Next-token prediction on raw text from an under-represented domain.

- **How Post-Training Shapes Biological Reasoning Models**. *Lukas Fesser, Hanlin Zhang, Michelle M. Li, Eric Wang, Bryan Perozzi, Shekoofeh Azizi, Sham M. Kakade, Marinka Zitnik*. [[Paper]](https://arxiv.org/abs/2606.16517) [[PDF]](https://arxiv.org/pdf/2606.16517) ![](https://img.shields.io/badge/year-2026-red)
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**. *Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo*. [[Paper]](https://arxiv.org/abs/2402.03300) [[PDF]](https://arxiv.org/pdf/2402.03300) ![](https://img.shields.io/badge/year-2024-yellow)

##### Intermediate Fine-Tuning

> Training on solved problems that need no domain knowledge, so that the acquired reasoning transfers and prepares later RL.

- **Mid-Training with Self-Generated Data Improves Reinforcement Learning in Language Models**. *Aswin RRV, Jacob Dineen, Divij Handa, Mihir Parmar, Ben Zhou, Swaroop Mishra, Chitta Baral*. [[Paper]](https://arxiv.org/abs/2605.08472) [[PDF]](https://arxiv.org/pdf/2605.08472) ![](https://img.shields.io/badge/year-2026-red)
- **Fundamental Reasoning Paradigms Induce Out-of-Domain Generalization in Language Models**. *Mingzi Cao, Xingwei Tan, Mahmud Elahi Akhter, Marco Valentino, Maria Liakata, Xi Wang, Nikolaos Aletras*. [[Paper]](https://arxiv.org/abs/2602.08658) [[PDF]](https://arxiv.org/pdf/2602.08658) ![](https://img.shields.io/badge/year-2026-red)
- **Atomic Skills are the Prerequisite: When Reinforcement Learning Synthesizes Compositional Reasoning, and When It Only Amplifies**. *Sitao Cheng, Xunjian Yin, Ruiwen Zhou, Yuxuan Li, Xinyi Wang, Liangming Pan, William Yang Wang, Victor Zhong*. [[Paper]](https://arxiv.org/abs/2512.01970) [[PDF]](https://arxiv.org/pdf/2512.01970) ![](https://img.shields.io/badge/year-2025-orange)
- **Warm Up Before You Train: Unlocking General Reasoning in Resource-Constrained Settings**. *Safal Shrestha, Minwu Kim, Aadim Nepal, Anubhav Shrestha, Keith Ross*. [[Paper]](https://arxiv.org/abs/2505.13718) [[PDF]](https://arxiv.org/pdf/2505.13718) ![](https://img.shields.io/badge/year-2025-orange)
- **Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus**. *Terufumi Morishita, Gaku Morio, Atsuki Yamaguchi, Yasuhiro Sogawa*. [[Paper]](https://arxiv.org/abs/2411.12498) [[PDF]](https://arxiv.org/pdf/2411.12498) ![](https://img.shields.io/badge/year-2024-yellow)

#### Post-Training: Supervised Fine-Tuning

*Imitation of demonstrations, which tends to absorb spurious surface correlations unless the problems, solutions, or objective are changed.*

##### Problem-Level Methods

> Represent the anticipated test shift among the training problems.

**Problem rewriting**

- **Mitigating Spurious Correlations in LLMs via Causality-Aware Post-Training**. *Shurui Gui, Shuiwang Ji*. [[Paper]](https://arxiv.org/abs/2506.09433) [[PDF]](https://arxiv.org/pdf/2506.09433) ![](https://img.shields.io/badge/year-2025-orange)
- **Your Language Model May Think Too Rigidly: Achieving Reasoning Consistency with Symmetry-Enhanced Training**. *Yihang Yao, Zhepeng Cen, Miao Li, William Han, Yuyou Zhang, Emerson Liu, Zuxin Liu, Chuang Gan, Ding Zhao*. [[Paper]](https://arxiv.org/abs/2502.17800) [[PDF]](https://arxiv.org/pdf/2502.17800) ![](https://img.shields.io/badge/year-2025-orange)

**Problem generation**

- **Ladders of Thought: A Self-Evolving Curriculum of Progressively Simplified Reasoning Traces**. *Minghui Liu, Thomas Magelinski, Dehao Yuan, Qi Yu, Furong Huang*. [[Paper]](https://arxiv.org/abs/2609.25643) [[PDF]](https://arxiv.org/pdf/2609.25643) ![](https://img.shields.io/badge/year-2026-red)
- **Selective Left-Shift: Turning Test-Time Compute and Difficulty-based Curation into Training Data for Low-Resource Code Generation**. *Didula Samaraweera, Anjana Supun, Srinath Perera*. [[Paper]](https://arxiv.org/abs/2607.07748) [[PDF]](https://arxiv.org/pdf/2607.07748) ![](https://img.shields.io/badge/year-2026-red)
- **Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning**. *Siyuan Liu, Tinghong Chen, Xinghan Li, Yifei Wang, Jingzhao Zhang*. [[Paper]](https://arxiv.org/abs/2605.12906) [[PDF]](https://arxiv.org/pdf/2605.12906) ![](https://img.shields.io/badge/year-2026-red)
- **Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training for Deductive Reasoning**. *Maximilian Mordig, Andreas Opedal, Weiyang Liu, Bernhard Schölkopf*. [[Paper]](https://arxiv.org/abs/2603.27226) [[PDF]](https://arxiv.org/pdf/2603.27226) ![](https://img.shields.io/badge/year-2026-red)
- **Towards Compositional Generalization of LLMs via Skill Taxonomy Guided Data Synthesis**. *Yifan Wei, Li Du, Xiaoyan Yu, Yang Feng, Angsheng Li*. [[Paper]](https://arxiv.org/abs/2601.03676) [[PDF]](https://arxiv.org/pdf/2601.03676) ![](https://img.shields.io/badge/year-2026-red)
- **Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges**. *Nayoung Lee, Ziyang Cai, Avi Schwarzschild, Kangwook Lee, Dimitris Papailiopoulos*. [[Paper]](https://arxiv.org/abs/2502.01612) [[PDF]](https://arxiv.org/pdf/2502.01612) ![](https://img.shields.io/badge/year-2025-orange)

##### Solution-Level Methods

> Rewrite the imitated solutions, since imitation generalizes only as far as its target exposes a reusable procedure.

**Abstract solutions**

- **Zipping the Thought: When and How Compressed Reasoning Data Works in LLM Post-Training**. *Kohsei Matsutani, Gouki Minegishi, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo*. [[Paper]](https://arxiv.org/abs/2605.28008) [[PDF]](https://arxiv.org/pdf/2605.28008) ![](https://img.shields.io/badge/year-2026-red)
- **Learning to Adapt SFT Data for Better Reasoning Generalization**. *Lisong Sun, Li Wang, Chen Zhang, Jinyang Wu, Kui Zhang, Tianhao Peng, Wenjun Wu*. [[Paper]](https://arxiv.org/abs/2605.26924) [[PDF]](https://arxiv.org/pdf/2605.26924) ![](https://img.shields.io/badge/year-2026-red)
- **Memorize Theorems, Not Instances: Probing SFT Generalization through Mathematical Reasoning**. *Ruiying Peng, Mengyu Yang, Jing Lei, Xiaohui Li, Xueyu Wu, Xinlei Chen*. [[Paper]](https://arxiv.org/abs/2605.09270) [[PDF]](https://arxiv.org/pdf/2605.09270) ![](https://img.shields.io/badge/year-2026-red)
- **From Meta-Thought to Execution: Cognitively Aligned Post-Training for Generalizable and Reliable LLM Reasoning**. *Shaojie Wang, Liang Zhang*. [[Paper]](https://arxiv.org/abs/2601.21909) [[PDF]](https://arxiv.org/pdf/2601.21909) ![](https://img.shields.io/badge/year-2026-red)

**Stepwise solutions**

- **The Imitation Game: Turing Machine Imitator is Length Generalizable Reasoner**. *Zhouqi Hua, Wenwei Zhang, Chengqi Lyu, Yuzhe Gu, Songyang Gao, Kuikun Liu, Dahua Lin, Kai Chen*. [[Paper]](https://arxiv.org/abs/2507.13332) [[PDF]](https://arxiv.org/pdf/2507.13332) ![](https://img.shields.io/badge/year-2025-orange)
- **Learning Composable Chains-of-Thought**. *Fangcong Yin, Zeyu Leo Liu, Liu Leqi, Xi Ye, Greg Durrett*. [[Paper]](https://arxiv.org/abs/2505.22635) [[PDF]](https://arxiv.org/pdf/2505.22635) ![](https://img.shields.io/badge/year-2025-orange)
- **Beyond In-Distribution Success: Scaling Curves of CoT Granularity for Language Model Generalization**. *Ru Wang, Wei Huang, Selena Song, Haoyu Zhang, Qian Niu, Yusuke Iwasawa, Yutaka Matsuo, Jiaxian Guo*. [[Paper]](https://arxiv.org/abs/2502.18273) [[PDF]](https://arxiv.org/pdf/2502.18273) ![](https://img.shields.io/badge/year-2025-orange)
- **Beyond Single-Task: Robust Multi-Task Length Generalization for LLMs**. *Yi Hu, Shijia Kang, Haotong Yang, Haotian Xu, Muhan Zhang*. [[Paper]](https://arxiv.org/abs/2502.11525) [[PDF]](https://arxiv.org/pdf/2502.11525) ![](https://img.shields.io/badge/year-2025-orange)
- **Case-Based or Rule-Based: How Do Transformers Do the Math?** *Yi Hu, Xiaojuan Tang, Haotong Yang, Muhan Zhang*. [[Paper]](https://arxiv.org/abs/2402.17709) [[PDF]](https://arxiv.org/pdf/2402.17709) ![](https://img.shields.io/badge/year-2024-yellow)

##### Objective-Level Methods

> Change the loss so that what would not hold under shift contributes less to the update.

**Teacher-based objectives**

- **Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models**. *Zhaoyi Li, Deyang Kong, Yuan Wei, Evan Yang, Ranran Shen, Mahardika Krisna Ihsani, Ming Yang, Wei Zhang, Chuan Hao, Jian Yang, et al.* [[Paper]](https://arxiv.org/abs/2608.16647) [[PDF]](https://arxiv.org/pdf/2608.16647) ![](https://img.shields.io/badge/year-2026-red)
- **RP-OPSD: Reasoning-Pivot-Guided On-Policy Self-Distillation for Multilingual Reasoning Transfer**. *Xinye Wang, Junxiao Liu, Shujian Huang*. [[Paper]](https://arxiv.org/abs/2608.06347) [[PDF]](https://arxiv.org/pdf/2608.06347) ![](https://img.shields.io/badge/year-2026-red)
- **Geometric Self-Distillation for Reasoning Generalization**. *Josip Jukić, Ivan Titov*. [[Paper]](https://arxiv.org/abs/2607.06855) [[PDF]](https://arxiv.org/pdf/2607.06855) ![](https://img.shields.io/badge/year-2026-red)
- **Beyond Imitation: Reflective On-Policy Self-Distillation for LLM Reasoning**. *Ziqi Zhao, Xinyu Ma, Liu Yang, Yujie Feng, Daiting Shi, Jingzhou He, Xin Xin, Zhaochun Ren, Xiao-Ming Wu*. [[Paper]](https://arxiv.org/abs/2605.28014) [[PDF]](https://arxiv.org/pdf/2605.28014) ![](https://img.shields.io/badge/year-2026-red)
- **Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?** *Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, Yuqing Yang*. [[Paper]](https://arxiv.org/abs/2603.24472) [[PDF]](https://arxiv.org/pdf/2603.24472) ![](https://img.shields.io/badge/year-2026-red)
- **Making Expert Reasoning Learnable with Self-Distillation**. *Ethan Mendes, Jungsoo Park, Alan Ritter*. [[Paper]](https://arxiv.org/abs/2602.02405) [[PDF]](https://arxiv.org/pdf/2602.02405) ![](https://img.shields.io/badge/year-2026-red)

**Regularization-based objectives**

- **Invariant Gradient Alignment for Robust Reasoning Distillation**. *Zehua Cheng, Wei Dai, Jiahao Sun*. [[Paper]](https://arxiv.org/abs/2606.05025) [[PDF]](https://arxiv.org/pdf/2606.05025) ![](https://img.shields.io/badge/year-2026-red)
- **Learning from Mistakes: Negative Reasoning Samples Enhance Out-of-Domain Generalization**. *Xueyun Tian, Minghua Ma, Bingbing Xu, Nuoyan Lyu, Wei Li, Heng Dong, Zheng Chu, Yuanzhuo Wang, Huawei Shen*. [[Paper]](https://arxiv.org/abs/2601.04992) [[PDF]](https://arxiv.org/pdf/2601.04992) ![](https://img.shields.io/badge/year-2026-red)

#### Post-Training: Reinforcement Learning

*Rewards replace reference solutions; transfer depends on the practiced environments, the rewarded behavior, and the update rule, and stays bounded by the base model.*

##### Scope and Limits of RL Transfer

> Controlled evidence on when RL transfers where SFT degrades, and on the base-model and depth limits of that transfer.

- **RL Post-Training Builds Compositional Reasoning Strategies**. *Azwar Abdulsalam, Nishil Patel, Andrew Saxe*. [[Paper]](https://arxiv.org/abs/2607.07646) [[PDF]](https://arxiv.org/pdf/2607.07646) ![](https://img.shields.io/badge/year-2026-red)
- **How Deep Can LLMs Learn to Reason? Expressiveness Is Key**. *Tianle Wang, Zhaoyang Wang, Guangchen Lan, Xinpeng Wei, Sipeng Zhang, Guanwen Qiu, Abulhair Saparov*. [[Paper]](https://arxiv.org/abs/2605.06638) [[PDF]](https://arxiv.org/pdf/2605.06638) ![](https://img.shields.io/badge/year-2026-red)
- **Why Does RL Generalize Better Than SFT? A Data-Centric Perspective on VLM Post-Training**. *Aojun Lu, Tao Feng, Hangjie Yuan, Wei Li, Yanan Sun*. [[Paper]](https://arxiv.org/abs/2602.10815) [[PDF]](https://arxiv.org/pdf/2602.10815) ![](https://img.shields.io/badge/year-2026-red)
- **When Domains Interact: Asymmetric and Order-Sensitive Cross-Domain Effects in Reinforcement Learning for Reasoning**. *Wang Yang, Shouren Wang, Chaoda Song, Chuang Ma, Xinpeng Li, Nengbo Wang, Kaixiong Zhou, Vipin Chaudhary, Xiaotian Han*. [[Paper]](https://arxiv.org/abs/2602.01365) [[PDF]](https://arxiv.org/pdf/2602.01365) ![](https://img.shields.io/badge/year-2026-red)
- **Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents**. *Zhihan Liu, Lin Guan, Yixin Nie, Kai Zhang, Zhuoqun Hao, Lin Chen, Asli Celikyilmaz, Zhaoran Wang, Na Zhang*. [[Paper]](https://arxiv.org/abs/2601.18217) [[PDF]](https://arxiv.org/pdf/2601.18217) ![](https://img.shields.io/badge/year-2026-red)
- **How Does RL Post-training Induce Skill Composition? A Case Study on Countdown**. *Simon Park, Simran Kaur, Sanjeev Arora*. [[Paper]](https://arxiv.org/abs/2512.01775) [[PDF]](https://arxiv.org/pdf/2512.01775) ![](https://img.shields.io/badge/year-2025-orange)
- **Can GRPO Help LLMs Transcend Their Pretraining Origin?** *Kangqi Ni, Zhen Tan, Zijie Liu, Pingzhi Li, Tianlong Chen*. [[Paper]](https://arxiv.org/abs/2510.15990) [[PDF]](https://arxiv.org/pdf/2510.15990) ![](https://img.shields.io/badge/year-2025-orange)
- **From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones**. *Lifan Yuan, Weize Chen, Yuchen Zhang, Ganqu Cui, Hanbin Wang, Ziming You, Ning Ding, Zhiyuan Liu, Maosong Sun, Hao Peng*. [[Paper]](https://arxiv.org/abs/2509.25123) [[PDF]](https://arxiv.org/pdf/2509.25123) ![](https://img.shields.io/badge/year-2025-orange)
- **RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?** *Yiyou Sun, Yuhan Cao, Pohao Huang, Haoyue Bai, Hannaneh Hajishirzi, Nouha Dziri, Dawn Song*. [[Paper]](https://arxiv.org/abs/2509.21016) [[PDF]](https://arxiv.org/pdf/2509.21016) ![](https://img.shields.io/badge/year-2025-orange)
- **Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning**. *Yu Li, Zhuoshi Pan, Honglin Lin, Mengyuan Sun, Conghui He, Lijun Wu*. [[Paper]](https://arxiv.org/abs/2507.17512) [[PDF]](https://arxiv.org/pdf/2507.17512) ![](https://img.shields.io/badge/year-2025-orange)
- **Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning**. *Maggie Huan, Yuetai Li, Tuney Zheng, Xiaoyu Xu, Seungone Kim, Minxin Du, Radha Poovendran, Graham Neubig, Xiang Yue*. [[Paper]](https://arxiv.org/abs/2507.00432) [[PDF]](https://arxiv.org/pdf/2507.00432) ![](https://img.shields.io/badge/year-2025-orange)
- **Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?** *Chuxuan Hu, Yuxuan Zhu, Antony Kellermann, Caleb Biddulph, Suppakit Waiwitlikhit, Jason Benn, Daniel Kang*. [[Paper]](https://arxiv.org/abs/2506.19733) [[PDF]](https://arxiv.org/pdf/2506.19733) ![](https://img.shields.io/badge/year-2025-orange)
- **Decomposing Elements of Problem Solving: What "Math" Does RL Teach?** *Tian Qin, Core Francisco Park, Mujin Kwun, Aaron Walsman, Eran Malach, Nikhil Anand, Hidenori Tanaka, David Alvarez-Melis*. [[Paper]](https://arxiv.org/abs/2505.22756) [[PDF]](https://arxiv.org/pdf/2505.22756) ![](https://img.shields.io/badge/year-2025-orange)
- **Reinforcement Learning for Reasoning in Large Language Models with One Training Example**. *Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al.* [[Paper]](https://arxiv.org/abs/2504.20571) [[PDF]](https://arxiv.org/pdf/2504.20571) ![](https://img.shields.io/badge/year-2025-orange)
- **Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?** *Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang*. [[Paper]](https://arxiv.org/abs/2504.13837) [[PDF]](https://arxiv.org/pdf/2504.13837) ![](https://img.shields.io/badge/year-2025-orange)
- **SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training**. *Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, Yi Ma*. [[Paper]](https://arxiv.org/abs/2501.17161) [[PDF]](https://arxiv.org/pdf/2501.17161) ![](https://img.shields.io/badge/year-2025-orange)

##### Environment Design

> Change the problems and checkers the model practices on: generated puzzles, composed environments, chained horizons, games, and controllers.

- **Training Language Models to Cooperate with Inference-Time Controllers**. *Moumita Choudhury, Vanshaj Khattar, Jing Liu, Toshiaki Koike-Akino, Ankush Chakrabarty, Shlomo Zilberstein, Ye Wang*. [[Paper]](https://arxiv.org/abs/2607.23771) [[PDF]](https://arxiv.org/pdf/2607.23771) ![](https://img.shields.io/badge/year-2026-red)
- **Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR**. *Yongjin Yang, Jiarui Liu, Yinghui He, Lechen Zhang, Bernhard Schölkopf, Zhijing Jin*. [[Paper]](https://arxiv.org/abs/2606.25178) [[PDF]](https://arxiv.org/pdf/2606.25178) ![](https://img.shields.io/badge/year-2026-red)
- **Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization**. *Hao Xiang, Qiaoyu Tang, Le Yu, Yaojie Lu, Xianpei Han, Ben He, Le Sun, Bowen Yu, Peng Wang, Hongyu Lin, Dayiheng Liu*. [[Paper]](https://arxiv.org/abs/2606.12373) [[PDF]](https://arxiv.org/pdf/2606.12373) ![](https://img.shields.io/badge/year-2026-red)
- **Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play**. *Xiachong Feng, Deyi Yin, Xiaocheng Feng, Yi Jiang, Libo Qin, Yangfan Ye, Lei Huang, Weitao Ma, Qiming Li, Yuxuan Gu, Bing Qin, Lingpeng Kong*. [[Paper]](https://arxiv.org/abs/2604.17696) [[PDF]](https://arxiv.org/pdf/2604.17696) ![](https://img.shields.io/badge/year-2026-red)
- **SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions**. *Ashima Suvarna, Kendrick Phan, Mehrab Beikzadeh, Hritik Bansal, Saadia Gabriel*. [[Paper]](https://arxiv.org/abs/2604.08477) [[PDF]](https://arxiv.org/pdf/2604.08477) ![](https://img.shields.io/badge/year-2026-red)
- **Learning from Synthetic Data Improves Multi-hop Reasoning**. *Anmol Kabra, Yilun Yin, Albert Gong, Kamilė Stankevičiūtė, Dongyoung Go, Johann Lee, Katie Z. Luo, Carla P. Gomes, Kilian Q. Weinberger*. [[Paper]](https://arxiv.org/abs/2603.02091) [[PDF]](https://arxiv.org/pdf/2603.02091) ![](https://img.shields.io/badge/year-2026-red)
- **GraphDancer: Training LLMs to Explore and Reason over Graphs via Two-Stage Curriculum Post-Training**. *Yuyang Bai, Zhuofeng Li, Ping Nie, Jianwen Xie, Yu Zhang*. [[Paper]](https://arxiv.org/abs/2602.02518) [[PDF]](https://arxiv.org/pdf/2602.02518) ![](https://img.shields.io/badge/year-2026-red)
- **h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning**. *Sumeet Ramesh Motwani, Alesia Ivanova, Ziyang Cai, Philip Torr, Riashat Islam, Shital Shah, Christian Schroeder de Witt, Charles London*. [[Paper]](https://arxiv.org/abs/2510.07312) [[PDF]](https://arxiv.org/pdf/2510.07312) ![](https://img.shields.io/badge/year-2025-orange)
- **LogicPuzzleRL: Cultivating Robust Mathematical Reasoning in LLMs via Reinforcement Learning**. *Zhen Hao Wong, Jingwen Deng, Runming He, Zirong Chen, Qijie You, Hejun Dong, Hao Liang, Chengyu Shen, Bin Cui, Wentao Zhang*. [[Paper]](https://arxiv.org/abs/2506.04821) [[PDF]](https://arxiv.org/pdf/2506.04821) ![](https://img.shields.io/badge/year-2025-orange)
- **Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles**. *Jiangjie Chen, Qianyu He, Siyu Yuan, Aili Chen, Zhicheng Cai, Weinan Dai, Hongli Yu, Qiying Yu, Xuefeng Li, Jiaze Chen, Hao Zhou, Mingxuan Wang*. [[Paper]](https://arxiv.org/abs/2505.19914) [[PDF]](https://arxiv.org/pdf/2505.19914) ![](https://img.shields.io/badge/year-2025-orange)
- **SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond**. *Junteng Liu, Yuanxiang Fan, Zhuo Jiang, Han Ding, Yongyi Hu, Chi Zhang, Yiqi Shi, Shitong Weng, Aili Chen, Shiqi Chen, et al.* [[Paper]](https://arxiv.org/abs/2505.19641) [[PDF]](https://arxiv.org/pdf/2505.19641) ![](https://img.shields.io/badge/year-2025-orange)

##### Reward Design

> Change what is rewarded beyond the final answer: process, structure, abstraction, cross-lingual consistency, and diversity.

- **GRAIN: Bridging Name and Narrative Shifts in Real-World Graph Reasoning through Invariance-Rewarded Agentic RL**. *Zike Yuan, Han Zhang, Jianzhi Yan, Le Liu, Cai Ke, Huozhi Zhou, Jian Xie, Jiran Yin, Yukun Cao, Yue Yu, et al.* [[Paper]](https://arxiv.org/abs/2608.27142) [[PDF]](https://arxiv.org/pdf/2608.27142) ![](https://img.shields.io/badge/year-2026-red)
- **CEDAR-GRPO: Process-Aware Reinforcement Learning for General Abductive Reasoning in LLMs**. *Moein Salimi, Danial Parnian, Shaygan Adim, Amirmohammad Ebrahiminasab, Nima Alighardashi, Parsa Gholami, Sahand Akramipour, Mahdi Jafari Siavoshani, Mohammad Hossein Rohban*. [[Paper]](https://arxiv.org/abs/2608.14791) [[PDF]](https://arxiv.org/pdf/2608.14791) ![](https://img.shields.io/badge/year-2026-red)
- **Cross-lingual Self-Consistency for Multilingual Reasoning with Language Models**. *Ahmed Elhady, Eneko Agirre, Mikel Artetxe*. [[Paper]](https://arxiv.org/abs/2606.01464) [[PDF]](https://arxiv.org/pdf/2606.01464) ![](https://img.shields.io/badge/year-2026-red)
- **When RL Suppresses Its Own Vocabulary: Recovering Reasoning Diversity in Puzzle-to-Math Transfer**. *Mayug Maniparambil, Arjun Karuvally, Terrence Sejnowski, Fergal Reid*. [[Paper]](https://arxiv.org/abs/2605.29190) [[PDF]](https://arxiv.org/pdf/2605.29190) ![](https://img.shields.io/badge/year-2026-red)
- **Rubric-Grounded RL: Structured Judge Rewards for Generalizable Reasoning**. *Manish Bhattarai, Ismael Boureima, Nishath Rajiv Ranasinghe, Scott Pakin, Dan O'Malley*. [[Paper]](https://arxiv.org/abs/2605.08061) [[PDF]](https://arxiv.org/pdf/2605.08061) ![](https://img.shields.io/badge/year-2026-red)
- **Can LLMs Learn to Reason Robustly under Noisy Supervision?** *Shenzhi Yang, Guangcheng Zhu, Bowen Song, Sharon Li, Haobo Wang, Xing Zheng, Yingfan Ma, Zhongqi Chen, Weiqiang Wang, Gang Chen*. [[Paper]](https://arxiv.org/abs/2604.03993) [[PDF]](https://arxiv.org/pdf/2604.03993) ![](https://img.shields.io/badge/year-2026-red)
- **Towards Generalizable Reasoning: Group Causal Counterfactual Policy Optimization for LLM Reasoning**. *Jingyao Wang, Peizheng Guo, Wenwen Qiang, Jiahuan Zhou, Huijie Guo, Changwen Zheng, Hui Xiong*. [[Paper]](https://arxiv.org/abs/2602.06475) [[PDF]](https://arxiv.org/pdf/2602.06475) ![](https://img.shields.io/badge/year-2026-red)
- **Diversity-Incentivized Exploration for Versatile Reasoning**. *Zican Hu, Shilin Zhang, Yafu Li, Jianhao Yan, Xuyang Hu, Leyang Cui, Xiaoye Qu, Chunlin Chen, Yu Cheng, Zhi Wang*. [[Paper]](https://arxiv.org/abs/2509.26209) [[PDF]](https://arxiv.org/pdf/2509.26209) ![](https://img.shields.io/badge/year-2025-orange)
- **Evolving Language Models without Labels: Majority Drives Selection, Novelty Promotes Variation**. *Yujun Zhou, Zhenwen Liang, Haolin Liu, Wenhao Yu, Kishan Panaganti, Linfeng Song, Dian Yu, Xiangliang Zhang, Haitao Mi, Dong Yu*. [[Paper]](https://arxiv.org/abs/2509.15194) [[PDF]](https://arxiv.org/pdf/2509.15194) ![](https://img.shields.io/badge/year-2025-orange)
- **AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking**. *Silin Gao, Antoine Bosselut, Samy Bengio, Emmanuel Abbe*. [[Paper]](https://arxiv.org/abs/2506.07751) [[PDF]](https://arxiv.org/pdf/2506.07751) ![](https://img.shields.io/badge/year-2025-orange)
- **Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners**. *Miao Peng, Nuo Chen, Zongrui Suo, Jia Li*. [[Paper]](https://arxiv.org/abs/2503.00845) [[PDF]](https://arxiv.org/pdf/2503.00845) ![](https://img.shields.io/badge/year-2025-orange)

##### Policy Optimization

> Change how rewards become updates: advantage correction, external solutions, and gradient or sharpness regularization.

- **Gradient Regularization Mitigates Reward Hacking in Reinforcement Learning from Human Feedback and Verifiable Rewards**. *Johannes Ackermann, Michael Noukhovitch, Takashi Ishida, Masashi Sugiyama*. [[Paper]](https://arxiv.org/abs/2602.18037) [[PDF]](https://arxiv.org/pdf/2602.18037) ![](https://img.shields.io/badge/year-2026-red)
- **CoRPO: Adding a Correctness Bias to GRPO Improves Generalization**. *Anisha Garg, Claire Zhang, Nishit Neema, David Bick, Ganesh Venkatesh, Joel Hestness*. [[Paper]](https://arxiv.org/abs/2511.04439) [[PDF]](https://arxiv.org/pdf/2511.04439) ![](https://img.shields.io/badge/year-2025-orange)
- **Sharpness-Guided Group Relative Policy Optimization via Probability Shaping**. *Tue Le, Linh Ngo Van, Trung Le*. [[Paper]](https://arxiv.org/abs/2511.00066) [[PDF]](https://arxiv.org/pdf/2511.00066) ![](https://img.shields.io/badge/year-2025-orange)
- **RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization**. *Yihong Dong, Xue Jiang, Yongding Tao, Huanyu Liu, Kechi Zhang, Lili Mou, Rongyu Cao, Yingwei Ma, Jue Chen, Binhua Li, et al.* [[Paper]](https://arxiv.org/abs/2508.00222) [[PDF]](https://arxiv.org/pdf/2508.00222) ![](https://img.shields.io/badge/year-2025-orange)

#### Post-Training: Hybrid Training

*Demonstrations introduce procedures the model cannot find alone; rewards recover the OOD accuracy that SFT reduces.*

##### Sequential Training

> SFT followed by RL, including methods that restore plasticity before the handoff.

- **When RL Fails after SFT: Rejuvenating Model Plasticity for Robust SFT-to-RL Handoff**. *Runze Liu, Jiashun Liu, Xu Wan, Yuqian Fu, Ling Pan*. [[Paper]](https://arxiv.org/abs/2606.09932) [[PDF]](https://arxiv.org/pdf/2606.09932) ![](https://img.shields.io/badge/year-2026-red)
- **SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning**. *Alexis Limozin, Eduard Durech, Torsten Hoefler, Imanol Schlag, Valentina Pyatkin*. [[Paper]](https://arxiv.org/abs/2604.23747) [[PDF]](https://arxiv.org/pdf/2604.23747) ![](https://img.shields.io/badge/year-2026-red)
- **ReasonXL: Shifting LLM Reasoning Language Without Sacrificing Performance**. *Daniil Gurgurov, Tom Röhr, Sebastian von Rohrscheidt, Josef van Genabith, Alexander Löser, Simon Ostermann*. [[Paper]](https://arxiv.org/abs/2604.12378) [[PDF]](https://arxiv.org/pdf/2604.12378) ![](https://img.shields.io/badge/year-2026-red)
- **X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains**. *Qianchu Liu, Sheng Zhang, Guanghui Qin, Timothy Ossowski, Yu Gu, Ying Jin, Sid Kiblawi, Sam Preston, Mu Wei, Paul Vozila, Tristan Naumann, Hoifung Poon*. [[Paper]](https://arxiv.org/abs/2505.03981) [[PDF]](https://arxiv.org/pdf/2505.03981) ![](https://img.shields.io/badge/year-2025-orange)

##### Joint Training

> Demonstration and reward signals combined within one stage, in the sampled solutions or in the loss.

- **GAC: Noise-Aware Adaptive Mixing for Hybrid SFT-RL Post-Training**. *Yuelin Hu, Zhenbo Yu, Zhengxue Cheng, Wei Liu, Li Song*. [[Paper]](https://arxiv.org/abs/2605.26184) [[PDF]](https://arxiv.org/pdf/2605.26184) ![](https://img.shields.io/badge/year-2026-red)
- **Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning**. *Taojie Zhu, Dongyang Xu, Ding Zou, Sen Zhao, Qiaobo Hao, Zhiguo Yang, Yonghong He*. [[Paper]](https://arxiv.org/abs/2604.08926) [[PDF]](https://arxiv.org/pdf/2604.08926) ![](https://img.shields.io/badge/year-2026-red)
- **Towards a Unified View of Large Language Model Post-Training**. *Xingtai Lv, Yuxin Zuo, Youbang Sun, Hongyi Liu, Yuntian Wei, Zhekai Chen, Xuekai Zhu, Kaiyan Zhang, Bingning Wang, Ning Ding, Bowen Zhou*. [[Paper]](https://arxiv.org/abs/2509.04419) [[PDF]](https://arxiv.org/pdf/2509.04419) ![](https://img.shields.io/badge/year-2025-orange)
- **Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling**. *Zeyu Huang, Tianhao Cheng, Zihan Qiu, Zili Wang, Yinghui Xu, Edoardo M. Ponti, Ivan Titov*. [[Paper]](https://arxiv.org/abs/2507.01679) [[PDF]](https://arxiv.org/pdf/2507.01679) ![](https://img.shields.io/badge/year-2025-orange)
- **SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning**. *Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, Dongbin Zhao*. [[Paper]](https://arxiv.org/abs/2506.19767) [[PDF]](https://arxiv.org/pdf/2506.19767) ![](https://img.shields.io/badge/year-2025-orange)
- **Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions**. *Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Yanhao Li, Bin Cui, Wentao Zhang*. [[Paper]](https://arxiv.org/abs/2506.07527) [[PDF]](https://arxiv.org/pdf/2506.07527) ![](https://img.shields.io/badge/year-2025-orange)
- **Learning to Reason under Off-Policy Guidance**. *Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, Yue Zhang*. [[Paper]](https://arxiv.org/abs/2504.14945) [[PDF]](https://arxiv.org/pdf/2504.14945) ![](https://img.shields.io/badge/year-2025-orange)

### Inference for Generalization

*How can a fixed model be deployed on unfamiliar problems by changing computation at test time, and where does that stop helping?*

#### Trajectory Restructuring

*Impose structural priors on the reasoning trajectory without updating weights.*

##### Structural Decomposition

> Split long or hard problems into units that are solved in sequence or by independent micro-steps.

- **Solving a Million-Step LLM Task with Zero Errors**. *Elliot Meyerson, Giuseppe Paolo, Roberto Dailey, Hormoz Shahrzad, Olivier Francon, Conor F. Hayes, Xin Qiu, Babak Hodjat, Risto Miikkulainen*. [[Paper]](https://arxiv.org/abs/2511.09030) [[PDF]](https://arxiv.org/pdf/2511.09030) ![](https://img.shields.io/badge/year-2025-orange)
- **An Examination on the Effectiveness of Divide-and-Conquer Prompting in Large Language Models**. *Yizhou Zhang, Lun Du, Defu Cao, Qiang Fu, Yan Liu*. [[Paper]](https://arxiv.org/abs/2402.05359) [[PDF]](https://arxiv.org/pdf/2402.05359) ![](https://img.shields.io/badge/year-2024-yellow)
- **Skills-in-Context Prompting: Unlocking Compositionality in Large Language Models**. *Jiaao Chen, Xiaoman Pan, Dian Yu, Kaiqiang Song, Xiaoyang Wang, Dong Yu, Jianshu Chen*. [[Paper]](https://arxiv.org/abs/2308.00304) [[PDF]](https://arxiv.org/pdf/2308.00304) ![](https://img.shields.io/badge/year-2023-lightgrey)
- **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models**. *Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, Ed Chi*. [[Paper]](https://arxiv.org/abs/2205.10625) [[PDF]](https://arxiv.org/pdf/2205.10625) ![](https://img.shields.io/badge/year-2022-lightgrey)

##### Upfront Structural Injection

> Fix constraints, a task-specific procedure, query-specific demonstrations, or hints before reasoning begins.

- **Constraint-First Reasoning: A Training-Free Protocol for Exploiting Answer-Space Constraints in Mathematical Problem Solving**. *Hongbo Ma, Bangji Yang, Yunqian Selina Cheng, Jiajun Fan, Hanwen Zhang, Ge Liu*. [[Paper]](https://arxiv.org/abs/2608.05254) [[PDF]](https://arxiv.org/pdf/2608.05254) ![](https://img.shields.io/badge/year-2026-red)
- **Test-Time Hinting for Black-Box Vision-Language Models**. *Kaihua Hou, Abhijith Varma Mudunuri, Jiaxing Qiu, Roxana Daneshjou, Thomas Hartvigsen, Ahmed Alaa*. [[Paper]](https://arxiv.org/abs/2605.16410) [[PDF]](https://arxiv.org/pdf/2605.16410) ![](https://img.shields.io/badge/year-2026-red)
- **Self-Demos: Eliciting Out-of-Demonstration Generalizability in Large Language Models**. *Wei He, Shichun Liu, Jun Zhao, Yiwen Ding, Yi Lu, Zhiheng Xi, Tao Gui, Qi Zhang, Xuanjing Huang*. [[Paper]](https://arxiv.org/abs/2404.00884) [[PDF]](https://arxiv.org/pdf/2404.00884) ![](https://img.shields.io/badge/year-2024-yellow)
- **Self-Discover: Large Language Models Self-Compose Reasoning Structures**. *Pei Zhou, Jay Pujara, Xiang Ren, Xinyun Chen, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, Denny Zhou, Swaroop Mishra, Huaixiu Steven Zheng*. [[Paper]](https://arxiv.org/abs/2402.03620) [[PDF]](https://arxiv.org/pdf/2402.03620) ![](https://img.shields.io/badge/year-2024-yellow)

#### Compute Scaling

*Allocate additional test-time compute to explore alternative paths or to revise defective ones.*

##### Search Expansion and Allocation

> Direct search toward promising branches and balance exploration against revision by estimated difficulty.

- **Exploration-Driven Optimization for Test-Time Large Language Model Reasoning**. *Changhao Li, Yuchen Zhuang, Chenxiao Gao, Haotian Sun, Rushi Qiang, Chao Zhang, Bo Dai*. [[Paper]](https://arxiv.org/abs/2605.09853) [[PDF]](https://arxiv.org/pdf/2605.09853) ![](https://img.shields.io/badge/year-2026-red)
- **Boosting Policy and Process Reward Models with Monte Carlo Tree Search in Open-Domain QA**. *Chi-Min Chan, Chunpu Xu, Junqi Zhu, Jiaming Ji, Donghai Hong, Pengcheng Wen, Chunyang Jiang, Zhen Ye, Yaodong Yang, Wei Xue, Sirui Han, Yike Guo*. [[Paper]](https://aclanthology.org/2025.findings-acl.388/) [[PDF]](https://aclanthology.org/2025.findings-acl.388.pdf) ![](https://img.shields.io/badge/year-2025-orange)
- **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters**. *Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar*. [[Paper]](https://arxiv.org/abs/2408.03314) [[PDF]](https://arxiv.org/pdf/2408.03314) ![](https://img.shields.io/badge/year-2024-yellow)

##### Targeted Revision

> Concentrate computation on localized faulty steps instead of regenerating the whole solution.

- **Test-Time Scaling via Error Localization**. *Rajiv Shailesh Chitale, Rahul Madhavan, Taneesh Gupta, Deepanway Ghosal, Aravindan Raghuveer*. [[Paper]](https://arxiv.org/abs/2607.21453) [[PDF]](https://arxiv.org/pdf/2607.21453) ![](https://img.shields.io/badge/year-2026-red)
- **Improving Latent Generalization Using Test-time Compute**. *Arslan Chaudhry, Sridhar Thiagarajan, Andrew Lampinen*. [[Paper]](https://arxiv.org/abs/2604.01430) [[PDF]](https://arxiv.org/pdf/2604.01430) ![](https://img.shields.io/badge/year-2026-red)

#### State Assessment

*Decide which intermediate reasoning states remain trustworthy under distribution shift.*

##### Internal State Assessment

> Use hidden-state and uncertainty signals to continue, switch strategy, stop, or recalibrate online.

- **Reasoning Errors Have a Region and a Direction in the Residual-Stream Trajectory of LLMs**. *Hamed Damirchi, Ignacio Meza De la Jara, Damith Ranasinghe, Yuhang Liu, Javen Shi*. [[Paper]](https://arxiv.org/abs/2608.05660) [[PDF]](https://arxiv.org/pdf/2608.05660) ![](https://img.shields.io/badge/year-2026-red)
- **UPAIR: Diagnosing Reasoning States via Uncertainty-Progress Alignment for Selective Intervention**. *Cheng Yan, Zhijun Fan, Guangyang Ye, Fan Xu, Xiang Xia, Yawei Wang, Wuyang Zhang*. [[Paper]](https://arxiv.org/abs/2607.17188) [[PDF]](https://arxiv.org/pdf/2607.17188) ![](https://img.shields.io/badge/year-2026-red)
- **Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning**. *Cai Zhou, Zekai Wang, Menghua Wu, Qianyu Julie Zhu, Flora C. Shi, Chenyu Wang, Ashia Wilson, Tommi Jaakkola, Stephen Bates*. [[Paper]](https://arxiv.org/abs/2604.01170) [[PDF]](https://arxiv.org/pdf/2604.01170) ![](https://img.shields.io/badge/year-2026-red)
- **Thought calibration: Efficient and confident test-time scaling**. *Menghua Wu, Cai Zhou, Stephen Bates, Tommi Jaakkola*. [[Paper]](https://arxiv.org/abs/2505.18404) [[PDF]](https://arxiv.org/pdf/2505.18404) ![](https://img.shields.io/badge/year-2025-orange)

##### Verifier-Mediated Assessment

> Use auxiliary verifiers to select candidates or to build pseudo-labels for test-time training.

- **Continuous Self-Improvement of Large Language Models by Test-time Training with Verifier-Driven Sample Selection**. *Mohammad Mahdi Moradi, Hossam Amer, Sudhir Mudur, Weiwei Zhang, Yang Liu, Walid Ahmed*. [[Paper]](https://arxiv.org/abs/2505.19475) [[PDF]](https://arxiv.org/pdf/2505.19475) ![](https://img.shields.io/badge/year-2025-orange)
- **Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers**. *Kusha Sareen, Morgane M Moss, Alessandro Sordoni, Rishabh Agarwal, Arian Hosseini*. [[Paper]](https://arxiv.org/abs/2505.04842) [[PDF]](https://arxiv.org/pdf/2505.04842) ![](https://img.shields.io/badge/year-2025-orange)

#### Inference-Time Externalization

*Move working state, exact operations, or accumulated experience outside a single forward reasoning process.*

##### Within-Episode Externalization

> External memory, abstract reasoning with tool-supplied values, and decisions about when a tool is needed.

- **To Infinity and Beyond: Tool-Use Unlocks Length Generalization in State Space Models**. *Eran Malach, Omid Saremi, Sinead Williamson, Arwen Bradley, Aryo Lotfi, Emmanuel Abbe, Josh Susskind, Etai Littwin*. [[Paper]](https://arxiv.org/abs/2510.14826) [[PDF]](https://arxiv.org/pdf/2510.14826) ![](https://img.shields.io/badge/year-2025-orange)
- **SMART: Self-Aware Agent for Tool Overuse Mitigation**. *Cheng Qian, Emre Can Acikgoz, Hongru Wang, Xiusi Chen, Avirup Sil, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji*. [[Paper]](https://arxiv.org/abs/2502.11435) [[PDF]](https://arxiv.org/pdf/2502.11435) ![](https://img.shields.io/badge/year-2025-orange)
- **Meta-Reasoning Improves Tool Use in Large Language Models**. *Lisa Alazraki, Marek Rei*. [[Paper]](https://arxiv.org/abs/2411.04535) [[PDF]](https://arxiv.org/pdf/2411.04535) ![](https://img.shields.io/badge/year-2024-yellow)
- **Efficient Tool Use with Chain-of-Abstraction Reasoning**. *Silin Gao, Jane Dwivedi-Yu, Ping Yu, Xiaoqing Ellen Tan, Ramakanth Pasunuru, Olga Golovneva, Koustuv Sinha, Asli Celikyilmaz, Antoine Bosselut, Tianlu Wang*. [[Paper]](https://arxiv.org/abs/2401.17464) [[PDF]](https://arxiv.org/pdf/2401.17464) ![](https://img.shields.io/badge/year-2024-yellow)

##### Cross-Episode Externalization

> Persistent memories of strategies distilled from earlier successes and failures, retrieved for later tasks.

- **MILES: Modular Instruction Memory with Learnable Selection for Self-Improving LLM Reasoning**. *Ruilin Tong, Dong Gong*. [[Paper]](https://arxiv.org/abs/2607.06974) [[PDF]](https://arxiv.org/pdf/2607.06974) ![](https://img.shields.io/badge/year-2026-red)
- **ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory**. *Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, et al.* [[Paper]](https://arxiv.org/abs/2509.25140) [[PDF]](https://arxiv.org/pdf/2509.25140) ![](https://img.shields.io/badge/year-2025-orange)

### Architecture for Generalization

*Which computational structures let a learned operation be reused at greater length, depth, or compositional complexity?*

#### Recurrent Depth

*Apply a shared transformation repeatedly within one prediction.*

##### Weight Sharing

> Universal and looped transformers, recurrence over facts or tool calls, and recurrent blocks retrofitted into pretrained models.

- **Universal Transformers for Circuit Computations: Perfect Length Generalization in Tiny Transformers**. *Takuya Ito, Ruchir Puri, Murray Campbell, Parikshit Ram*. [[Paper]](https://arxiv.org/abs/2608.31067) [[PDF]](https://arxiv.org/pdf/2608.31067) ![](https://img.shields.io/badge/year-2026-red)
- **Looped Language Models Improve Compositional Tool Calling**. *Andrei Cristian Popescu, Haitz Sáez de Ocáriz Borde, Pietro Liò*. [[Paper]](https://arxiv.org/abs/2608.18171) [[PDF]](https://arxiv.org/pdf/2608.18171) ![](https://img.shields.io/badge/year-2026-red)
- **Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets**. *Mark Shapiro*. [[Paper]](https://arxiv.org/abs/2608.11233) [[PDF]](https://arxiv.org/pdf/2608.11233) ![](https://img.shields.io/badge/year-2026-red)
- **Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers**. *Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, Yuekun Yao*. [[Paper]](https://arxiv.org/abs/2604.07822) [[PDF]](https://arxiv.org/pdf/2604.07822) ![](https://img.shields.io/badge/year-2026-red)
- **Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with Depth-Recurrent Transformers for Compositional Generalization**. *Hung-Hsuan Chen*. [[Paper]](https://arxiv.org/abs/2603.21676) [[PDF]](https://arxiv.org/pdf/2603.21676) ![](https://img.shields.io/badge/year-2026-red)
- **Unlocking Out-of-Distribution Generalization in Transformers via Recursive Latent Space Reasoning**. *Awni Altabaa, Siyu Chen, John Lafferty, Zhuoran Yang*. [[Paper]](https://arxiv.org/abs/2510.14095) [[PDF]](https://arxiv.org/pdf/2510.14095) ![](https://img.shields.io/badge/year-2025-orange)
- **Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning**. *Qifan Yu, Zhenyu He, Sijie Li, Xun Zhou, Jun Zhang, Jingjing Xu, Di He*. [[Paper]](https://arxiv.org/abs/2502.08482) [[PDF]](https://arxiv.org/pdf/2502.08482) ![](https://img.shields.io/badge/year-2025-orange)
- **Looped Transformers for Length Generalization**. *Ying Fan, Yilun Du, Kannan Ramchandran, Kangwook Lee*. [[Paper]](https://arxiv.org/abs/2409.15647) [[PDF]](https://arxiv.org/pdf/2409.15647) ![](https://img.shields.io/badge/year-2024-yellow)
- **Universal Transformers**. *Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser*. [[Paper]](https://arxiv.org/abs/1807.03819) [[PDF]](https://arxiv.org/pdf/1807.03819) ![](https://img.shields.io/badge/year-2018-lightgrey)

##### Adaptive Computation

> Learned controllers that choose how many iterations, layers, or latent passes an input receives.

- **Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping**. *Hsun-Yu Kuo, El Mahdi Chayti, Patrik Reizinger, Wieland Brendel, Martin Jaggi*. [[Paper]](https://arxiv.org/abs/2606.29983) [[PDF]](https://arxiv.org/pdf/2606.29983) ![](https://img.shields.io/badge/year-2026-red)
- **Skip a Layer or Loop It? Learning Program-of-Layers in LLMs**. *Ziyue Li, Yang Li, Tianyi Zhou*. [[Paper]](https://arxiv.org/abs/2606.06574) [[PDF]](https://arxiv.org/pdf/2606.06574) ![](https://img.shields.io/badge/year-2026-red)
- **Adaptive Recurrence as Algorithmic Time for Length Generalization in Addition**. *Imran Ibrahimli, Stefan Wermter, Jae Hee Lee*. [[Paper]](https://openreview.net/forum?id=XW18V4H9sq) [[PDF]](https://openreview.net/pdf?id=XW18V4H9sq) ![](https://img.shields.io/badge/year-2026-red)
- **Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning**. *Tianyu Fu, Yichen You, Zekai Chen, Guohao Dai, Huazhong Yang, Yu Wang*. [[Paper]](https://arxiv.org/abs/2511.08577) [[PDF]](https://arxiv.org/pdf/2511.08577) ![](https://img.shields.io/badge/year-2025-orange)
- **Adaptivity and Modularity for Efficient Generalization Over Task Complexity**. *Samira Abnar, Omid Saremi, Laurent Dinh, Shantel Wilson, Miguel Angel Bautista, Chen Huang, Vimal Thilak, Etai Littwin, Jiatao Gu, Josh Susskind, Samy Bengio*. [[Paper]](https://arxiv.org/abs/2310.08866) [[PDF]](https://arxiv.org/pdf/2310.08866) ![](https://img.shields.io/badge/year-2023-lightgrey)

##### Stable Recurrent Dynamics

> Fixed points, attractors, and energy descent that keep extra iterations useful instead of drifting.

- **Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth**. *Ivan Viakhirev, Kirill Borodin, Amirah Almutairi, Serguei Barannikov, Maxim Abramov, Grach Mkrtchian*. [[Paper]](https://arxiv.org/abs/2608.18222) [[PDF]](https://arxiv.org/pdf/2608.18222) ![](https://img.shields.io/badge/year-2026-red)
- **Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers**. *Sajad Movahedi, Vera Milovanović, Shlomo Libo Feigin, Alexander Theus, Thomas Hofmann, Valentina Boeva, T. Konstantin Rusch, Antonio Orvieto*. [[Paper]](https://arxiv.org/abs/2606.18206) [[PDF]](https://arxiv.org/pdf/2606.18206) ![](https://img.shields.io/badge/year-2026-red)
- **Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning**. *Benhao Huang, Zhengyang Geng, Zico Kolter*. [[Paper]](https://arxiv.org/abs/2605.21488) [[PDF]](https://arxiv.org/pdf/2605.21488) ![](https://img.shields.io/badge/year-2026-red)
- **Stability and Generalization in Looped Transformers**. *Asher Labovich*. [[Paper]](https://arxiv.org/abs/2604.15259) [[PDF]](https://arxiv.org/pdf/2604.15259) ![](https://img.shields.io/badge/year-2026-red)
- **Generalizable Reasoning through Compositional Energy Minimization**. *Alexandru Oarga, Yilun Du*. [[Paper]](https://arxiv.org/abs/2510.20607) [[PDF]](https://arxiv.org/pdf/2510.20607) ![](https://img.shields.io/badge/year-2025-orange)

#### Information Routing

*Specify how tokens are addressed, which dependencies attention reaches, and which modules process them.*

##### Positional Encoding

> Encode each token's computational role so that alignment survives longer or differently composed inputs.

- **How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization**. *Xinyi Wu, Siyuan Liu, Ali Jadbabaie*. [[Paper]](https://arxiv.org/abs/2607.07678) [[PDF]](https://arxiv.org/pdf/2607.07678) ![](https://img.shields.io/badge/year-2026-red)
- **Randomized YaRN Improves Length Generalization for Long-Context Reasoning**. *Manas Mehta, Fangcong Yin, Greg Durrett*. [[Paper]](https://arxiv.org/abs/2606.23687) [[PDF]](https://arxiv.org/pdf/2606.23687) ![](https://img.shields.io/badge/year-2026-red)
- **Position Encoding with Random Float Sampling Enhances Length Generalization of Transformers**. *Atsushi Shimizu, Shohei Taniguchi, Yutaka Matsuo*. [[Paper]](https://arxiv.org/abs/2602.14050) [[PDF]](https://arxiv.org/pdf/2602.14050) ![](https://img.shields.io/badge/year-2026-red)
- **Rethinking Addressing in Language Models via Contexualized Equivariant Positional Encoding**. *Jiajun Zhu, Peihao Wang, Ruisi Cai, Jason D. Lee, Pan Li, Zhangyang Wang*. [[Paper]](https://arxiv.org/abs/2501.00712) [[PDF]](https://arxiv.org/pdf/2501.00712) ![](https://img.shields.io/badge/year-2025-orange)
- **Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count**. *Hanseul Cho, Jaeyoung Cha, Srinadh Bhojanapalli, Chulhee Yun*. [[Paper]](https://arxiv.org/abs/2410.15787) [[PDF]](https://arxiv.org/pdf/2410.15787) ![](https://img.shields.io/badge/year-2024-yellow)
- **Explicitly Encoding Structural Symmetry is Key to Length Generalization in Arithmetic Tasks**. *Mahdi Sabbaghi, George Pappas, Hamed Hassani, Surbhi Goel*. [[Paper]](https://arxiv.org/abs/2406.01895) [[PDF]](https://arxiv.org/pdf/2406.01895) ![](https://img.shields.io/badge/year-2024-yellow)
- **Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure**. *Hanseul Cho, Jaeyoung Cha, Pranjal Awasthi, Srinadh Bhojanapalli, Anupam Gupta, Chulhee Yun*. [[Paper]](https://arxiv.org/abs/2405.20671) [[PDF]](https://arxiv.org/pdf/2405.20671) ![](https://img.shields.io/badge/year-2024-yellow)
- **Transformers Can Do Arithmetic with the Right Embeddings**. *Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, Tom Goldstein*. [[Paper]](https://arxiv.org/abs/2405.17399) [[PDF]](https://arxiv.org/pdf/2405.17399) ![](https://img.shields.io/badge/year-2024-yellow)
- **Positional Description Matters for Transformers Arithmetic**. *Ruoqi Shen, Sébastien Bubeck, Ronen Eldan, Yin Tat Lee, Yuanzhi Li, Yi Zhang*. [[Paper]](https://arxiv.org/abs/2311.14737) [[PDF]](https://arxiv.org/pdf/2311.14737) ![](https://img.shields.io/badge/year-2023-lightgrey)

##### Attention Patterns

> Constrain which intermediate results later computation can retrieve, preserving local operations at longer lengths.

- **On Locality and Length Generalization in Visual Reasoning**. *Pulkit Madan, Sanjay Haresh, Reza Ebrahimi, Sunny Panchal, Apratim Bhattacharyya, Roland Memisevic*. [[Paper]](https://arxiv.org/abs/2607.09061) [[PDF]](https://arxiv.org/pdf/2607.09061) ![](https://img.shields.io/badge/year-2026-red)
- **From Interpolation to Extrapolation: Complete Length Generalization for Arithmetic Transformers**. *Shaoxiong Duan, Yining Shi, Wei Xu*. [[Paper]](https://arxiv.org/abs/2310.11984) [[PDF]](https://arxiv.org/pdf/2310.11984) ![](https://img.shields.io/badge/year-2023-lightgrey)
- **Transformer Working Memory Enables Regular Language Reasoning and Natural Language Length Extrapolation**. *Ta-Chung Chi, Ting-Han Fan, Alexander I. Rudnicky, Peter J. Ramadge*. [[Paper]](https://arxiv.org/abs/2305.03796) [[PDF]](https://arxiv.org/pdf/2305.03796) ![](https://img.shields.io/badge/year-2023-lightgrey)

##### Modular Reasoning

> Route parts of an input to specialized heads, experts, or neuro-symbolic components that can be recombined.

- **What You Can't See Is What You Learn: Slot-Selective Evidence Masking Favors Compositional Generalization in Shared-Genome Language-Model Societies**. *Narcis Marincat*. [[Paper]](https://arxiv.org/abs/2608.20054) [[PDF]](https://arxiv.org/pdf/2608.20054) ![](https://img.shields.io/badge/year-2026-red)
- **AGEL-Comp: A Neuro-Symbolic Framework for Compositional Generalization in Interactive Agents**. *Mahnoor Shahid, Hannes Rothe*. [[Paper]](https://arxiv.org/abs/2604.26522) [[PDF]](https://arxiv.org/pdf/2604.26522) ![](https://img.shields.io/badge/year-2026-red)
- **NeSyCoCo: A Neuro-Symbolic Concept Composer for Compositional Generalization**. *Danial Kamali, Elham J. Barezi, Parisa Kordjamshidi*. [[Paper]](https://arxiv.org/abs/2412.15588) [[PDF]](https://arxiv.org/pdf/2412.15588) ![](https://img.shields.io/badge/year-2024-yellow)
- **Sparse Mixture-of-Experts for Compositional Generalization: Empirical Evidence and Theoretical Foundations of Optimal Sparsity**. *Jinze Zhao, Peihao Wang, Junjie Yang, Ruisi Cai, Gaowen Liu, Jayanth Srinivasa, Ramana Rao Kompella, Yingbin Liang, Zhangyang Wang*. [[Paper]](https://arxiv.org/abs/2410.13964) [[PDF]](https://arxiv.org/pdf/2410.13964) ![](https://img.shields.io/badge/year-2024-yellow)
- **Dynamic MOdularized Reasoning for Compositional Structured Explanation Generation**. *Xiyan Fu, Anette Frank*. [[Paper]](https://arxiv.org/abs/2309.07624) [[PDF]](https://arxiv.org/pdf/2309.07624) ![](https://img.shields.io/badge/year-2023-lightgrey)

#### Recurrent Memory

*Carry an explicit state across tokens or reasoning chunks so later computation can reuse earlier results.*

##### Token-Level Recurrence

> Each input or observation updates a carried state.

- **State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence for Latent Space Reasoning**. *Thea Aviss*. [[Paper]](https://arxiv.org/abs/2605.00206) [[PDF]](https://arxiv.org/pdf/2605.00206) ![](https://img.shields.io/badge/year-2026-red)
- **Exploration of Fast-Slow Latent Recurrence for Train-Short, Test-Long Generalization**. *Shota Takashiro, Masanori Koyama, Takeru Miyato, Yusuke Iwasawa, Yutaka Matsuo, Kohei Hayashi*. [[Paper]](https://arxiv.org/abs/2604.01577) [[PDF]](https://arxiv.org/pdf/2604.01577) ![](https://img.shields.io/badge/year-2026-red)
- **Rational Transductors**. *Mehryar Mohri*. [[Paper]](https://arxiv.org/abs/2602.07599) [[PDF]](https://arxiv.org/pdf/2602.07599) ![](https://img.shields.io/badge/year-2026-red)

##### Chunk-Level Recurrence

> A compact state summarizes one reasoning segment for the next.

- **Training Transformers as a Universal Computer**. *Ruize Xu, Chenxiao Yang, Yanhong Li, David McAllester*. [[Paper]](https://arxiv.org/abs/2604.25166) [[PDF]](https://arxiv.org/pdf/2604.25166) ![](https://img.shields.io/badge/year-2026-red)
- **Latent Reasoning with Supervised Thinking States**. *Ido Amos, Avi Caciularu, Mor Geva, Amir Globerson, Jonathan Herzig, Lior Shani, Idan Szpektor*. [[Paper]](https://arxiv.org/abs/2602.08332) [[PDF]](https://arxiv.org/pdf/2602.08332) ![](https://img.shields.io/badge/year-2026-red)

### Analysis of Generalization

*When does reasoning transfer, and what explains its successes and failures?*

#### Behavioral Evidence

*Performance under named shifts: one row of the transfer matrix, not a single OOD score.*

##### Compositional Generalization

> Fixed primitives, new combinations: deeper and wider computation graphs, multi-hop chaining, and rule extrapolation.

- **XDomainBench: Diagnosing Reasoning Collapse in High-Dimensional Scientific Knowledge Composition**. *Gong Zhiren, Tiantong Wu, Jiaming Zhang, Fuyao Zhang, Che Wang, Yurong Hao, Yikun Hou, Foo Ping, Yilei Zhao, Fei Huang, Chau Yuen, Wei Yang Bryan Lim*. [[Paper]](https://arxiv.org/abs/2605.14754) [[PDF]](https://arxiv.org/pdf/2605.14754) ![](https://img.shields.io/badge/year-2026-red)
- **OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization**. *Yiyou Sun, Shawn Hu, Georgia Zhou, Ken Zheng, Hannaneh Hajishirzi, Nouha Dziri, Dawn Song*. [[Paper]](https://arxiv.org/abs/2506.18880) [[PDF]](https://arxiv.org/pdf/2506.18880) ![](https://img.shields.io/badge/year-2025-orange)
- **Compositional-ARC: Assessing Systematic Generalization in Abstract Spatial Reasoning**. *Philipp Mondorf, Shijia Zhou, Monica Riedler, Barbara Plank*. [[Paper]](https://arxiv.org/abs/2504.01445) [[PDF]](https://arxiv.org/pdf/2504.01445) ![](https://img.shields.io/badge/year-2025-orange)
- **MathGAP: Out-of-Distribution Evaluation on Problems with Arbitrarily Complex Proofs**. *Andreas Opedal, Haruki Shirakami, Bernhard Schölkopf, Abulhair Saparov, Mrinmaya Sachan*. [[Paper]](https://arxiv.org/abs/2410.13502) [[PDF]](https://arxiv.org/pdf/2410.13502) ![](https://img.shields.io/badge/year-2024-yellow)
- **The Mystery of Compositional Generalization in Graph-based Generative Commonsense Reasoning**. *Xiyan Fu, Anette Frank*. [[Paper]](https://arxiv.org/abs/2410.06272) [[PDF]](https://arxiv.org/pdf/2410.06272) ![](https://img.shields.io/badge/year-2024-yellow)
- **Rule Extrapolation in Language Models: A Study of Compositional Generalization on OOD Prompts**. *Anna Mészáros, Szilvia Ujváry, Wieland Brendel, Patrik Reizinger, Ferenc Huszár*. [[Paper]](https://arxiv.org/abs/2409.13728) [[PDF]](https://arxiv.org/pdf/2409.13728) ![](https://img.shields.io/badge/year-2024-yellow)
- **Faith and Fate: Limits of Transformers on Compositionality**. *Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, et al.* [[Paper]](https://arxiv.org/abs/2305.18654) [[PDF]](https://arxiv.org/pdf/2305.18654) ![](https://img.shields.io/badge/year-2023-lightgrey)
- **Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples**. *Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Seyed Mehran Kazemi, Najoung Kim, He He*. [[Paper]](https://arxiv.org/abs/2305.15269) [[PDF]](https://arxiv.org/pdf/2305.15269) ![](https://img.shields.io/badge/year-2023-lightgrey)
- **Measuring and Narrowing the Compositionality Gap in Language Models**. *Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, Mike Lewis*. [[Paper]](https://arxiv.org/abs/2210.03350) [[PDF]](https://arxiv.org/pdf/2210.03350) ![](https://img.shields.io/badge/year-2022-lightgrey)
- **Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks**. *Brenden M. Lake, Marco Baroni*. [[Paper]](https://arxiv.org/abs/1711.00350) [[PDF]](https://arxiv.org/pdf/1711.00350) ![](https://img.shields.io/badge/year-2017-lightgrey)

##### Length, Depth, and Difficulty

> Longer inputs, deeper nesting or planning horizons, and easy-to-hard transfer, which diverge even within one task.

- **Generalization in LLM Problem Solving: The Case of the Shortest Path**. *Yao Tong, Jiayuan Ye, Anastasia Borovykh, Reza Shokri*. [[Paper]](https://arxiv.org/abs/2604.15306) [[PDF]](https://arxiv.org/pdf/2604.15306) ![](https://img.shields.io/badge/year-2026-red)
- **On the Out-of-Distribution Generalization of Reasoning in Multimodal LLMs for Simple Visual Planning Tasks**. *Yannic Neuhaus, Nicolas Flammarion, Matthias Hein, Francesco Croce*. [[Paper]](https://arxiv.org/abs/2602.15460) [[PDF]](https://arxiv.org/pdf/2602.15460) ![](https://img.shields.io/badge/year-2026-red)
- **Exploring Depth Generalization in Large Language Models for Solving Recursive Logic Tasks**. *Zhiyuan He*. [[Paper]](https://arxiv.org/abs/2512.02677) [[PDF]](https://arxiv.org/pdf/2512.02677) ![](https://img.shields.io/badge/year-2025-orange)
- **Revisiting Generalization Across Difficulty Levels: It's Not So Easy**. *Yeganeh Kordi, Nihal V. Nayak, Max Zuo, Ilana Nguyen, Stephen H. Bach*. [[Paper]](https://arxiv.org/abs/2511.21692) [[PDF]](https://arxiv.org/pdf/2511.21692) ![](https://img.shields.io/badge/year-2025-orange)
- **Extrapolation by Association: Length Generalization Transfer in Transformers**. *Ziyang Cai, Nayoung Lee, Avi Schwarzschild, Samet Oymak, Dimitris Papailiopoulos*. [[Paper]](https://arxiv.org/abs/2506.09251) [[PDF]](https://arxiv.org/pdf/2506.09251) ![](https://img.shields.io/badge/year-2025-orange)
- **The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity**. *Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, Mehrdad Farajtabar*. [[Paper]](https://arxiv.org/abs/2506.06941) [[PDF]](https://arxiv.org/pdf/2506.06941) ![](https://img.shields.io/badge/year-2025-orange)
- **Easy2Hard-Bench: Standardized Difficulty Labels for Profiling LLM Performance and Generalization**. *Mucong Ding, Chenghao Deng, Jocelyn Choo, Zichu Wu, Aakriti Agrawal, Avi Schwarzschild, Tianyi Zhou, Tom Goldstein, John Langford, Anima Anandkumar, Furong Huang*. [[Paper]](https://arxiv.org/abs/2409.18433) [[PDF]](https://arxiv.org/pdf/2409.18433) ![](https://img.shields.io/badge/year-2024-yellow)
- **Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process**. *Tian Ye, Zicheng Xu, Yuanzhi Li, Zeyuan Allen-Zhu*. [[Paper]](https://arxiv.org/abs/2407.20311) [[PDF]](https://arxiv.org/pdf/2407.20311) ![](https://img.shields.io/badge/year-2024-yellow)
- **NATURAL PLAN: Benchmarking LLMs on Natural Language Planning**. *Huaixiu Steven Zheng, Swaroop Mishra, Hugh Zhang, Xinyun Chen, Minmin Chen, Azade Nova, Le Hou, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, Denny Zhou*. [[Paper]](https://arxiv.org/abs/2406.04520) [[PDF]](https://arxiv.org/pdf/2406.04520) ![](https://img.shields.io/badge/year-2024-yellow)
- **Chain of Thoughtlessness? An Analysis of CoT in Planning**. *Kaya Stechly, Karthik Valmeekam, Subbarao Kambhampati*. [[Paper]](https://arxiv.org/abs/2405.04776) [[PDF]](https://arxiv.org/pdf/2405.04776) ![](https://img.shields.io/badge/year-2024-yellow)
- **Transformers Can Achieve Length Generalization But Not Robustly**. *Yongchao Zhou, Uri Alon, Xinyun Chen, Xuezhi Wang, Rishabh Agarwal, Denny Zhou*. [[Paper]](https://arxiv.org/abs/2402.09371) [[PDF]](https://arxiv.org/pdf/2402.09371) ![](https://img.shields.io/badge/year-2024-yellow)
- **Exploring Length Generalization in Large Language Models**. *Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, Behnam Neyshabur*. [[Paper]](https://arxiv.org/abs/2207.04901) [[PDF]](https://arxiv.org/pdf/2207.04901) ![](https://img.shields.io/badge/year-2022-lightgrey)

##### Prior Exposure

> How far a test lies from earlier exposure: perturbed templates, counterfactual conventions, unseen directions, and contamination-limited tasks.

- **LLMs Can See the Smoke but not the Fire: Evaluating Abductive Reasoning with Elenchos**. *Julius Steiglechner, Lucas Mahler, Gabriele Lohmann*. [[Paper]](https://arxiv.org/abs/2607.12733) [[PDF]](https://arxiv.org/pdf/2607.12733) ![](https://img.shields.io/badge/year-2026-red)
- **Testing LLM Arithmetic Reasoning Generalization with Automatic Numeric-Remapping Attacks**. *Malia Barker, Bishal Lakha, Edoardo Serra, Francesco Gullo*. [[Paper]](https://arxiv.org/abs/2606.03606) [[PDF]](https://arxiv.org/pdf/2606.03606) ![](https://img.shields.io/badge/year-2026-red)
- **Reasoners or Translators? Contamination-aware Evaluation and Neuro-Symbolic Robustness in Tax Law**. *Parisa Kordjamshidi, Samer Aslan, Madhavan Seshadri, Leslie Barrett, Enrico Santus*. [[Paper]](https://arxiv.org/abs/2605.16052) [[PDF]](https://arxiv.org/pdf/2605.16052) ![](https://img.shields.io/badge/year-2026-red)
- **General365: Benchmarking General Reasoning in Large Language Models Across Diverse and Challenging Tasks**. *Junlin Liu, Shengnan An, Shuang Zhou, Dan Ma, Shixiong Luo, Ying Xie, Yuan Zhang, Wenling Yuan, Yifan Zhou, Xiaoyu Li, et al.* [[Paper]](https://arxiv.org/abs/2604.11778) [[PDF]](https://arxiv.org/pdf/2604.11778) ![](https://img.shields.io/badge/year-2026-red)
- **EsoLang-Bench: Evaluating Genuine Reasoning in Large Language Models via Esoteric Programming Languages**. *Aman Sharma, Paras Chopra*. [[Paper]](https://arxiv.org/abs/2603.09678) [[PDF]](https://arxiv.org/pdf/2603.09678) ![](https://img.shields.io/badge/year-2026-red)
- **On the Emergence and Test-Time Use of Structural Information in Large Language Models**. *Michelle Chao Chen, Moritz Miller, Bernhard Schölkopf, Siyuan Guo*. [[Paper]](https://arxiv.org/abs/2601.17869) [[PDF]](https://arxiv.org/pdf/2601.17869) ![](https://img.shields.io/badge/year-2026-red)
- **Disentangling generalization and memorization in large language models using chess**. *Leonard S. Pleiss, Maximilian Schiffer, Robert K. von Weizsaecker*. [[Paper]](https://arxiv.org/abs/2601.16823) [[PDF]](https://arxiv.org/pdf/2601.16823) ![](https://img.shields.io/badge/year-2026-red)
- **Beyond Memorization: Testing LLM Reasoning on Unseen Theory of Computation Tasks**. *Shlok Shelat, Jay Raval, Souvik Roy, Manas Gaur*. [[Paper]](https://arxiv.org/abs/2601.13392) [[PDF]](https://arxiv.org/pdf/2601.13392) ![](https://img.shields.io/badge/year-2026-red)
- **Generalization of RLVR Using Causal Reasoning as a Testbed**. *Brian Lu, Hongyu Zhao, Shuo Sun, Hao Peng, Rui Ding, Hongyuan Mei*. [[Paper]](https://arxiv.org/abs/2512.20760) [[PDF]](https://arxiv.org/pdf/2512.20760) ![](https://img.shields.io/badge/year-2025-orange)
- **Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens**. *Chengshuai Zhao, Zhen Tan, Pingchuan Ma, Dawei Li, Bohan Jiang, Yancheng Wang, Yingzhen Yang, Huan Liu*. [[Paper]](https://arxiv.org/abs/2508.01191) [[PDF]](https://arxiv.org/pdf/2508.01191) ![](https://img.shields.io/badge/year-2025-orange)
- **MATH-Perturb: Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations**. *Kaixuan Huang, Jiacheng Guo, Zihao Li, Xiang Ji, Jiawei Ge, Wenzhe Li, Yingqing Guo, Tianle Cai, Hui Yuan, Runzhe Wang, et al.* [[Paper]](https://arxiv.org/abs/2502.06453) [[PDF]](https://arxiv.org/pdf/2502.06453) ![](https://img.shields.io/badge/year-2025-orange)
- **GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models**. *Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar*. [[Paper]](https://arxiv.org/abs/2410.05229) [[PDF]](https://arxiv.org/pdf/2410.05229) ![](https://img.shields.io/badge/year-2024-yellow)
- **modeLing: A Novel Dataset for Testing Linguistic Reasoning in Language Models**. *Nathan A. Chi, Teodor Malchev, Riley Kong, Ryan A. Chi, Lucas Huang, Ethan A. Chi, R. Thomas McCoy, Dragomir Radev*. [[Paper]](https://arxiv.org/abs/2406.17038) [[PDF]](https://arxiv.org/pdf/2406.17038) ![](https://img.shields.io/badge/year-2024-yellow)
- **GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers**. *Qintong Li, Leyang Cui, Xueliang Zhao, Lingpeng Kong, Wei Bi*. [[Paper]](https://arxiv.org/abs/2402.19255) [[PDF]](https://arxiv.org/pdf/2402.19255) ![](https://img.shields.io/badge/year-2024-yellow)
- **Embers of Autoregression: Understanding Large Language Models Through the Problem They are Trained to Solve**. *R. Thomas McCoy, Shunyu Yao, Dan Friedman, Matthew Hardy, Thomas L. Griffiths*. [[Paper]](https://arxiv.org/abs/2309.13638) [[PDF]](https://arxiv.org/pdf/2309.13638) ![](https://img.shields.io/badge/year-2023-lightgrey)
- **The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"**. *Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, Owain Evans*. [[Paper]](https://arxiv.org/abs/2309.12288) [[PDF]](https://arxiv.org/pdf/2309.12288) ![](https://img.shields.io/badge/year-2023-lightgrey)
- **Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks**. *Zhaofeng Wu, Linlu Qiu, Alexis Ross, Ekin Akyürek, Boyuan Chen, Bailin Wang, Najoung Kim, Jacob Andreas, Yoon Kim*. [[Paper]](https://arxiv.org/abs/2307.02477) [[PDF]](https://arxiv.org/pdf/2307.02477) ![](https://img.shields.io/badge/year-2023-lightgrey)

#### Mechanistic Evidence

*Whether one shared computation or separate shortcuts produce the behavior, mostly in small controlled models.*

##### Shared Circuits

> Overlap between the circuits used on both sides of a shift predicts transfer, but may fail on new knowledge.

- **Shared circuits predict whether LLMs generalize across formats in arithmetic reasoning**. *Andrea Gregor de Varda, Sana Pandey, Pengrui Han, Jacob Andreas, Evelina Fedorenko*. [[Paper]](https://arxiv.org/abs/2609.04463) [[PDF]](https://arxiv.org/pdf/2609.04463) ![](https://img.shields.io/badge/year-2026-red)
- **From Reasoning Traces to Reusable Modules: Understanding Compositional Generalization in Language Model Reasoning**. *Lingjing Kong, Xin Liu, Guangyi Chen, Martin Q. Ma, Xiangchen Song, Yuekai Sun, Mikhail Yurochkin, Taylor W. Killian, Ruslan Salakhutdinov, Kun Zhang, Eric P. Xing, Zhengzhong Liu*. [[Paper]](https://arxiv.org/abs/2606.18089) [[PDF]](https://arxiv.org/pdf/2606.18089) ![](https://img.shields.io/badge/year-2026-red)
- **Discovering Interpretable Algorithms by Decompiling Transformers to RASP**. *Xinting Huang, Aleksandra Bakalova, Satwik Bhattamishra, William Merrill, Michael Hahn*. [[Paper]](https://arxiv.org/abs/2602.08857) [[PDF]](https://arxiv.org/pdf/2602.08857) ![](https://img.shields.io/badge/year-2026-red)
- **Is Grokking Worthwhile? Functional Analysis and Transferability of Generalization Circuits in Transformers**. *Kaiyu He, Zhang Mian, Peilin Wu, Xinya Du, Zhiyu Chen*. [[Paper]](https://arxiv.org/abs/2601.09049) [[PDF]](https://arxiv.org/pdf/2601.09049) ![](https://img.shields.io/badge/year-2026-red)
- **Characterizing Pattern Matching and Its Limits on Compositional Task Structures**. *Hoyeon Chang, Jinho Park, Hanseul Cho, Sohee Yang, Miyoung Ko, Hyeonbin Hwang, Seungpil Won, Dohaeng Lee, Youbin Ahn, Minjoon Seo*. [[Paper]](https://arxiv.org/abs/2505.20278) [[PDF]](https://arxiv.org/pdf/2505.20278) ![](https://img.shields.io/badge/year-2025-orange)
- **Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking**. *Yifan Zhang, Wenyu Du, Dongming Jin, Jie Fu, Zhi Jin*. [[Paper]](https://arxiv.org/abs/2502.20129) [[PDF]](https://arxiv.org/pdf/2502.20129) ![](https://img.shields.io/badge/year-2025-orange)
- **Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning**. *Xinhao Yao, Ruifeng Ren, Yun Liao, Lizhong Ding, Yong Liu*. [[Paper]](https://arxiv.org/abs/2502.04667) [[PDF]](https://arxiv.org/pdf/2502.04667) ![](https://img.shields.io/badge/year-2025-orange)

##### Interfaces Between Steps

> Intermediate results that are computed but do not reach the next step in a usable form.

- **Why Knowing Both Hops Is Not Enough: Understanding Two-Hop Generalization in Language Models**. *Zili Zhang, Yilin Wang, Heng Wang, Herun Wan, Minnan Luo*. [[Paper]](https://arxiv.org/abs/2608.07261) [[PDF]](https://arxiv.org/pdf/2608.07261) ![](https://img.shields.io/badge/year-2026-red)
- **Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning**. *Lu Dai, Ziyang Rao, Yili Wang, Hanqing Wang, Hao Liu, Hui Xiong*. [[Paper]](https://arxiv.org/abs/2607.08393) [[PDF]](https://arxiv.org/pdf/2607.08393) ![](https://img.shields.io/badge/year-2026-red)
- **Representational Homomorphism Predicts and Improves Compositional Generalization In Transformer Language Model**. *Zhiyu An, Wan Du*. [[Paper]](https://arxiv.org/abs/2601.18858) [[PDF]](https://arxiv.org/pdf/2601.18858) ![](https://img.shields.io/badge/year-2026-red)
- **Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries**. *Eden Biran, Daniela Gottesman, Sohee Yang, Mor Geva, Amir Globerson*. [[Paper]](https://arxiv.org/abs/2406.12775) [[PDF]](https://arxiv.org/pdf/2406.12775) ![](https://img.shields.io/badge/year-2024-yellow)

##### Learning Dynamics

> Which of several fitting solutions optimization selects, and when: grokking, critical windows, and checkpoint choice.

- **Critical Windows of Complexity Control: When Transformers Decide to Reason or Memorize**. *Sarwan Ali*. [[Paper]](https://arxiv.org/abs/2605.04396) [[PDF]](https://arxiv.org/pdf/2605.04396) ![](https://img.shields.io/badge/year-2026-red)
- **Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models**. *Dan Shi, Zhuowen Han, Simon Ostermann, Renren Jin, Josef van Genabith, Deyi Xiong*. [[Paper]](https://arxiv.org/abs/2604.25011) [[PDF]](https://arxiv.org/pdf/2604.25011) ![](https://img.shields.io/badge/year-2026-red)
- **Rethinking Generalization in Reasoning SFT: A Conditional Analysis on Optimization, Data, and Model Capability**. *Qihan Ren, Peng Wang, Ruikun Cai, Shuai Shao, Dadi Guo, Yuejin Xie, Yafu Li, Quanshi Zhang, Xia Hu, Jing Shao, Dongrui Liu*. [[Paper]](https://arxiv.org/abs/2604.06628) [[PDF]](https://arxiv.org/pdf/2604.06628) ![](https://img.shields.io/badge/year-2026-red)
- **Early-Warning Signals of Grokking via Loss-Landscape Geometry**. *Yongzhong Xu*. [[Paper]](https://arxiv.org/abs/2602.16967) [[PDF]](https://arxiv.org/pdf/2602.16967) ![](https://img.shields.io/badge/year-2026-red)
- **RL Fine-Tuning Heals OOD Forgetting in SFT**. *Hangzhan Jin, Sitao Luan, Tianwei Ni, Sicheng Lyu, Guillaume Rabusseau, Reihaneh Rabbany, Doina Precup, Mohammad Hamdaqa*. [[Paper]](https://arxiv.org/abs/2509.12235) [[PDF]](https://arxiv.org/pdf/2509.12235) ![](https://img.shields.io/badge/year-2025-orange)
- **Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization**. *Boshi Wang, Xiang Yue, Yu Su, Huan Sun*. [[Paper]](https://arxiv.org/abs/2405.15071) [[PDF]](https://arxiv.org/pdf/2405.15071) ![](https://img.shields.io/badge/year-2024-yellow)
- **Grokking of Hierarchical Structure in Vanilla Transformers**. *Shikhar Murty, Pratyusha Sharma, Jacob Andreas, Christopher D. Manning*. [[Paper]](https://arxiv.org/abs/2305.18741) [[PDF]](https://arxiv.org/pdf/2305.18741) ![](https://img.shields.io/badge/year-2023-lightgrey)
- **Progress measures for grokking via mechanistic interpretability**. *Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt*. [[Paper]](https://arxiv.org/abs/2301.05217) [[PDF]](https://arxiv.org/pdf/2301.05217) ![](https://img.shields.io/badge/year-2023-lightgrey)

#### Theoretical Evidence

*Results under explicit assumptions that separate expressivity, learnability, and certification.*

##### Learnability

> Whether finite training selects an extrapolating procedure over shortcuts: locality, inductive bias, scratchpads, curricula, and data diversity.

- **Protoreasoning in Tiny Transformers**. *Eduardo Valle, Fergal Reid*. [[Paper]](https://arxiv.org/abs/2608.04980) [[PDF]](https://arxiv.org/pdf/2608.04980) ![](https://img.shields.io/badge/year-2026-red)
- **Relative Positions Generalize, Absolute Positions Memorize: An Implicit-Bias Account of Length Generalization in Attention**. *Subham Singh, Ashutosh Mishra, Subha Raut*. [[Paper]](https://arxiv.org/abs/2607.18759) [[PDF]](https://arxiv.org/pdf/2607.18759) ![](https://img.shields.io/badge/year-2026-red)
- **Learning to Reason with Curriculum II: Compositional Generalization**. *Nived Rajaraman, Audrey Huang, Miroslav Dudik, Robert Schapire, Dylan Foster, Akshay Krishnamurthy*. [[Paper]](https://arxiv.org/abs/2606.27721) [[PDF]](https://arxiv.org/pdf/2606.27721) ![](https://img.shields.io/badge/year-2026-red)
- **A Measure-Theoretic Analysis of Reasoning: Structural Generalization and Approximation Limits**. *Yuyang Zhang, Yifu Zhang, Xuehai Zhou, Xiaoyin Chen*. [[Paper]](https://arxiv.org/abs/2605.19944) [[PDF]](https://arxiv.org/pdf/2605.19944) ![](https://img.shields.io/badge/year-2026-red)
- **When Symbol Names Should Not Matter: A Logistic Theory of Fresh-Symbol Classification**. *Wenjie Guan, Jelena Bradic*. [[Paper]](https://arxiv.org/abs/2605.07120) [[PDF]](https://arxiv.org/pdf/2605.07120) ![](https://img.shields.io/badge/year-2026-red)
- **Barriers to Universal Reasoning With Transformers (And How to Overcome Them)**. *Oliver Kraus, Yash Sarrof, Yuekun Yao, Alexander Koller, Michael Hahn*. [[Paper]](https://arxiv.org/abs/2604.25800) [[PDF]](https://arxiv.org/pdf/2604.25800) ![](https://img.shields.io/badge/year-2026-red)
- **Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization**. *Yu Huang, Zixin Wen, Aarti Singh, Yuejie Chi, Yuxin Chen*. [[Paper]](https://arxiv.org/abs/2511.07378) [[PDF]](https://arxiv.org/pdf/2511.07378) ![](https://img.shields.io/badge/year-2025-orange)
- **How Far Can Transformers Reason? The Globality Barrier and Inductive Scratchpad**. *Emmanuel Abbe, Samy Bengio, Aryo Lotfi, Colin Sandon, Omid Saremi*. [[Paper]](https://arxiv.org/abs/2406.06467) [[PDF]](https://arxiv.org/pdf/2406.06467) ![](https://img.shields.io/badge/year-2024-yellow)
- **On Provable Length and Compositional Generalization**. *Kartik Ahuja, Amin Mansouri*. [[Paper]](https://arxiv.org/abs/2402.04875) [[PDF]](https://arxiv.org/pdf/2402.04875) ![](https://img.shields.io/badge/year-2024-yellow)
- **What Algorithms can Transformers Learn? A Study in Length Generalization**. *Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Josh Susskind, Samy Bengio, Preetum Nakkiran*. [[Paper]](https://arxiv.org/abs/2310.16028) [[PDF]](https://arxiv.org/pdf/2310.16028) ![](https://img.shields.io/badge/year-2023-lightgrey)
- **The Impact of Positional Encoding on Length Generalization in Transformers**. *Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy*. [[Paper]](https://arxiv.org/abs/2305.19466) [[PDF]](https://arxiv.org/pdf/2305.19466) ![](https://img.shields.io/badge/year-2023-lightgrey)
- **Transformers Learn Shortcuts to Automata**. *Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, Cyril Zhang*. [[Paper]](https://arxiv.org/abs/2210.10749) [[PDF]](https://arxiv.org/pdf/2210.10749) ![](https://img.shields.io/badge/year-2022-lightgrey)

##### Certification

> Whether a finite test can establish extrapolation: length-generalization bounds and their limits.

- **Length Generalization for Transformers via Compression**. *Georg Zetzsche, Hongjian Jiang, Andy Yang, Pascal Bergsträßer, Marco Sälzer, David Chiang, Anthony W. Lin*. [[Paper]](https://arxiv.org/abs/2609.08851) [[PDF]](https://arxiv.org/pdf/2609.08851) ![](https://img.shields.io/badge/year-2026-red)
- **Algebraic Decomposition Theory for Transformer Length Generalization**. *Andy Yang, Blerta Veseli, Corentin Barloy, Michaël Cadilhac, Andreas Krebs, Charles Paperman, Howard Straubing, Michael Hahn*. [[Paper]](https://arxiv.org/abs/2608.13433) [[PDF]](https://arxiv.org/pdf/2608.13433) ![](https://img.shields.io/badge/year-2026-red)
- **On the Ability of Transformers to Verify Plans**. *Yash Sarrof, Yupei Du, Katharina Stein, Alexander Koller, Sylvie Thiébaux, Michael Hahn*. [[Paper]](https://arxiv.org/abs/2603.19954) [[PDF]](https://arxiv.org/pdf/2603.19954) ![](https://img.shields.io/badge/year-2026-red)
- **Length Generalization Bounds for Transformers**. *Andy Yang, Pascal Bergsträßer, Georg Zetzsche, David Chiang, Anthony W. Lin*. [[Paper]](https://arxiv.org/abs/2603.02238) [[PDF]](https://arxiv.org/pdf/2603.02238) ![](https://img.shields.io/badge/year-2026-red)
- **Quantitative Bounds for Length Generalization in Transformers**. *Zachary Izzo, Eshaan Nichani, Jason D. Lee*. [[Paper]](https://arxiv.org/abs/2510.27015) [[PDF]](https://arxiv.org/pdf/2510.27015) ![](https://img.shields.io/badge/year-2025-orange)
- **A Formal Framework for Understanding Length Generalization in Transformers**. *Xinting Huang, Andy Yang, Satwik Bhattamishra, Yash Sarrof, Andreas Krebs, Hattie Zhou, Preetum Nakkiran, Michael Hahn*. [[Paper]](https://arxiv.org/abs/2410.02140) [[PDF]](https://arxiv.org/pdf/2410.02140) ![](https://img.shields.io/badge/year-2024-yellow)

## Datasets and Benchmarks

Resources summarized in Appendix C of the survey. A benchmark is not in or out of distribution by itself: its status depends on the evaluated model's pre-training, adaptation data, and task definition. Counts follow the cited releases, and "~" marks rounded counts.

### Training Datasets

**Math: verifiable prompts for RL**

| Dataset | Year | Domain | Problems / tasks | Supervision |
| --- | --- | --- | --- | --- |
| [GSM8K (train)](https://arxiv.org/abs/2110.14168) | 2021 | Grade-school math | 7,473 | Answer + rationale |
| [MATH (train)](https://arxiv.org/abs/2103.03874) | 2021 | Competition math | 7,500 | Answer + solution |
| [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) | 2025 | Math | ~8K per split | Answer |
| [DAPO-Math-17k](https://arxiv.org/abs/2503.14476) | 2025 | Competition math | ~17K | Integer answer |
| [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | 2025 | Competition math | ~40K | Answer |
| [DeepMath-103K](https://arxiv.org/abs/2504.11456) | 2025 | Math | 103K | Answer + traces |
| [Skywork-OR1](https://arxiv.org/abs/2505.22312) | 2025 | Math / code | ~105K / 13.7K | Answer / tests |
| [ORZ-Math](https://arxiv.org/abs/2503.24290) | 2025 | Competition math | ~129K | Answer |
| [Big-Math](https://arxiv.org/abs/2502.17387) | 2025 | Math | >250K | Answer |

**Math: demonstrations for SFT, distillation, and hybrid training**

| Dataset | Year | Domain | Problems / tasks | Supervision |
| --- | --- | --- | --- | --- |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 2024 | Math | ~860K | CoT solution |
| [OpenR1-Math-220k (all)](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 2025 | Math | ~225K problems | 2-4 R1 traces per problem |
| [OpenR1-Math-46k-8192](https://huggingface.co/datasets/Elliott/Openr1-Math-46k-8192) | 2025 | Math | ~46K problems | Long CoT (R1) + answer |
| [LIMO (released set)](https://arxiv.org/abs/2502.03387) | 2025 | Math | 817 problems | Long CoT |
| [s1K](https://arxiv.org/abs/2501.19393) | 2025 | Math, science | 1,000 problems | Long CoT |
| [OpenThoughts releases](https://arxiv.org/abs/2506.04178) | 2025 | Math, code, science | 114K / 1M / 1.2M problems | Long CoT |

**Logic, puzzles, and procedural environments**

| Dataset | Year | Domain | Problems / tasks | Supervision |
| --- | --- | --- | --- | --- |
| [Knights and Knaves](https://arxiv.org/abs/2502.14768) | 2025 | Logic puzzles | Procedural | Assignment checker |
| [Reasoning Gym](https://arxiv.org/abs/2505.24760) | 2025 | Algorithmic, logic | 100+ generators | Task verifier |
| [SynLogic](https://arxiv.org/abs/2505.19641) | 2025 | Logic | 35 task types | Task verifier |
| [Enigmata](https://arxiv.org/abs/2505.19914) | 2025 | Puzzles | 36 task types | Task verifier |

**Multi-domain**

| Dataset | Year | Domain | Problems / tasks | Supervision |
| --- | --- | --- | --- | --- |
| [SciQ (train)](https://arxiv.org/abs/1707.06209) | 2017 | Science MCQ | 11,679 | Correct option |
| [GURU](https://arxiv.org/abs/2506.14965) | 2025 | Six domains | 92K | Domain verifiers |
| [WebInstruct-verified](https://arxiv.org/abs/2505.14652) | 2025 | Multi-domain | ~230K | Model-based verifier |
| [DataFlex-RL mixture](https://arxiv.org/abs/2609.06107) | 2026 | Math, logic, science | 15K | Rule-based checks |

### Evaluation Benchmarks

**Mathematics**

| Benchmark | Year | Evaluation axis | Train | Test | Metric |
| --- | --- | --- | --- | --- | --- |
| [GSM8K](https://arxiv.org/abs/2110.14168) | 2021 | Instance | 7,473 | 1,319 | Accuracy |
| [Minerva / OCW](https://arxiv.org/abs/2206.14858) | 2022 | Task / domain | - | 272 | Accuracy |
| AMC | 2022-23 | Instance | - | 40 / 83 (harness subsets) | avg@k |
| [MATH-500](https://arxiv.org/abs/2305.20050) | 2023 | Instance | - | 500 | Accuracy |
| AIME 2024 | 2024 | Instance | - | 30 | avg@k |
| [OlympiadBench](https://arxiv.org/abs/2402.14008) | 2024 | Difficulty | - | 675 (English text-only math) | Accuracy |
| AIME 2025 | 2025 | Instance (temporal check) | - | 30 | avg@k |

**Science and knowledge**

| Benchmark | Year | Evaluation axis | Train | Test | Metric |
| --- | --- | --- | --- | --- | --- |
| [ARC-Challenge](https://arxiv.org/abs/1803.05457) | 2018 | Task / domain | 1,119 | 1,172 | Accuracy |
| [GPQA-Diamond](https://arxiv.org/abs/2311.12022) | 2023 | Task / domain | - | 198 | Accuracy |
| [MMLU-Pro](https://arxiv.org/abs/2406.01574) | 2024 | Task / domain | - | 12,032 | Accuracy |
| [SuperGPQA](https://arxiv.org/abs/2502.14739) | 2025 | Task / domain | - | 26,529 | Accuracy |
| [Humanity's Last Exam](https://arxiv.org/abs/2501.14249) | 2025 | Difficulty | - | 2,500 | Accuracy |
| [ScienceArena](https://arxiv.org/abs/2608.30517) | 2026 | Task / domain | - | 13 competitions | Rubric credit |

**Logic, puzzles, and code**

| Benchmark | Year | Evaluation axis | Train | Test | Metric |
| --- | --- | --- | --- | --- | --- |
| [ARC-AGI](https://arxiv.org/abs/1911.01547) | 2019 | Task / domain | 400 | 400 (public) | Accuracy |
| [HumanEval](https://arxiv.org/abs/2107.03374) | 2021 | Task / domain | - | 164 | pass@k |
| [BBH](https://arxiv.org/abs/2210.09261) | 2022 | Task / domain | - | 23 tasks | Accuracy |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) | 2024 | Task / domain (temporal check) | - | Rolling | pass@1 |
| [Knights and Knaves](https://arxiv.org/abs/2502.14768) | 2025 | Difficulty | Procedural | Procedural | Accuracy |
| [ZebraLogic](https://arxiv.org/abs/2502.01100) | 2025 | Difficulty | - | 1,000 | Accuracy |

**Surface perturbation and distractors**

| Benchmark | Year | Evaluation axis | Train | Test | Metric |
| --- | --- | --- | --- | --- | --- |
| [GSM-IC](https://arxiv.org/abs/2302.00093) | 2023 | Distractors | - | 58,052 | Micro / macro accuracy |
| [GSM-Plus](https://arxiv.org/abs/2402.19255) | 2024 | Perturbation / difficulty | - | 10,552 | Accuracy |
| [GSM-Symbolic](https://arxiv.org/abs/2410.05229) | 2024 | Surface / format | - | 100 templates x 50 instances | Mean and std |

**Composition and length**

| Benchmark | Year | Evaluation axis | Train | Test | Metric |
| --- | --- | --- | --- | --- | --- |
| [SCAN (length split)](https://arxiv.org/abs/1711.00350) | 2017 | Length / depth | 16,990 | 3,920 | Exact match |
| [CFQ](https://arxiv.org/abs/1912.09713) | 2019 | Composition | 239,357 in total | - | Exact match |
| [COGS](https://arxiv.org/abs/2010.05465) | 2020 | Composition | 24,155 | 21,000 | Exact match |
| [Last-letter concatenation](https://arxiv.org/abs/2205.10625) | 2022 | Length / depth | - | 500 per length | Accuracy |
| [CLRS-30](https://arxiv.org/abs/2205.15659) | 2022 | Size / length | 1,000 per algorithm | 32 per algorithm | Micro-F1 |
| [MathGAP](https://arxiv.org/abs/2410.13502) | 2024 | Length / depth | - | Generated | Accuracy |
| [Multi-digit addition](https://arxiv.org/abs/2402.09371) | 2024 | Length / depth | Up to 40 digits | Up to 100 digits | Accuracy |
| [Compositional-ARC](https://arxiv.org/abs/2504.01445) | 2025 | Composition | 82,908 episodes | 8,546 episodes | Accuracy |
| [OMEGA](https://arxiv.org/abs/2506.18880) | 2025 | Difficulty; composition | Generated | Generated | Accuracy |
| [XDomainBench](https://arxiv.org/abs/2605.14754) | 2026 | Composition | - | 8,598 sessions | Recall / F1 |

**Planning, agents, and algorithmic or modality shifts**

| Benchmark | Year | Evaluation axis | Train | Test | Metric |
| --- | --- | --- | --- | --- | --- |
| [A-OKVQA](https://arxiv.org/abs/2206.01718) | 2022 | Modality (vision) | 17,056 | ~1.1K (val) | Exact match / accuracy |
| [FACTOR (wiki/news)](https://arxiv.org/abs/2307.06908) | 2024 | Knowledge shift | - | Generated | Accuracy |
| [WebArena](https://arxiv.org/abs/2307.13854) | 2024 | Environment / tool | - | 812 | Success rate |
| [Mind2Web](https://arxiv.org/abs/2306.06070) | 2023 | Domain / environment | - | 2,000+ | Success rate |
| [SWE-bench Verified](https://arxiv.org/abs/2310.06770) | 2024 | Task / length | - | 500 | Code-fixing pass rate |
| [Reversal Curse](https://arxiv.org/abs/2309.12288) | 2024 | Algorithmic direction | Generated | Generated | Exact string match |

**Recent and contamination-limited benchmarks**

| Benchmark | Year | Evaluation axis | Train | Test | Metric |
| --- | --- | --- | --- | --- | --- |
| [GSM1K](https://arxiv.org/abs/2405.00332) | 2024 | Instance (temporal check) | - | 1,205 | Accuracy |
| [LiveBench](https://arxiv.org/abs/2406.19314) | 2024 | Instance (temporal check) | - | Monthly | Accuracy |
| [General365](https://arxiv.org/abs/2604.11778) | 2026 | Task; variants | - | 365 seeds + 1,095 variants | Accuracy |
| [EsoLang-Bench](https://arxiv.org/abs/2603.09678) | 2026 | Programming language | - | 80 per language | Accuracy |

## Related Surveys

- **A Survey of Inductive Reasoning for Large Language Models**. *Kedi Chen, Dezhao Ruan, Yuhao Dan, Yaoting Wang, Siyu Yan, Xuecheng Wu, Yinqi Zhang, Qin Chen, Jie Zhou, Liang He, et al.* [[Paper]](https://arxiv.org/abs/2510.10182) [[PDF]](https://arxiv.org/pdf/2510.10182) ![](https://img.shields.io/badge/year-2025-orange)
- **Generalizability of Large Language Model-Based Agents: A Comprehensive Survey**. *Minxing Zhang, Yi Yang, Roy Xie, Bhuwan Dhingra, Shuyan Zhou, Jian Pei*. [[Paper]](https://arxiv.org/abs/2509.16330) [[PDF]](https://arxiv.org/pdf/2509.16330) ![](https://img.shields.io/badge/year-2025-orange)
- **State-of-the-art generalisation research in NLP: A taxonomy and review**. *Dieuwke Hupkes, Mario Giulianelli, Verna Dankers, Mikel Artetxe, Yanai Elazar, Tiago Pimentel, Christos Christodoulopoulos, Karim Lasri, Naomi Saphra, Arabella Sinclair, et al.* [[Paper]](https://arxiv.org/abs/2210.03050) [[PDF]](https://arxiv.org/pdf/2210.03050) ![](https://img.shields.io/badge/year-2022-lightgrey)
- **Towards Out-Of-Distribution Generalization: A Survey**. *Jiashuo Liu, Zheyan Shen, Yue He, Xingxuan Zhang, Renzhe Xu, Han Yu, Peng Cui*. [[Paper]](https://arxiv.org/abs/2108.13624) [[PDF]](https://arxiv.org/pdf/2108.13624) ![](https://img.shields.io/badge/year-2021-lightgrey)

## Citation

The paper citation will be added when the survey is released. To cite the living repository in the meantime:

```bibtex
@misc{awesome_reasoning_generalization_2026,
  title        = {Awesome Reasoning Generalization},
  author       = {{Awesome Reasoning Generalization Contributors}},
  year         = {2026},
  howpublished = {\url{https://github.com/tue09/awesome-reasoning-generalization}},
  note         = {Accessed: YYYY-MM-DD}
}
```
