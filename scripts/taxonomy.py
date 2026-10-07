"""Taxonomy of the survey, shared by the README and figure builders.

The structure follows the section and paragraph headings of the survey body (Sections 3-6).
Each paper in ``data/paper_manifest.tsv`` points to one leaf through its ``taxonomy_id``
(``pillar.section.leaf``); supervised fine-tuning leaves additionally carry a subgroup.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Leaf:
    id: str
    name: str
    description: str
    subgroups: tuple[str, ...] = ()


@dataclass(frozen=True)
class Section:
    id: str
    name: str
    description: str
    leaves: tuple[Leaf, ...]
    figure_name: str = ""
    parent: str = ""


@dataclass(frozen=True)
class Pillar:
    id: str
    name: str
    question: str
    sections: tuple[Section, ...] = field(default_factory=tuple)


TAXONOMY: tuple[Pillar, ...] = (
    Pillar(
        "training",
        "Training for Generalization",
        "How can parameter updates make a learned reasoning procedure survive shifts in form, composition, length, difficulty, domain, or language?",
        (
            Section(
                "pre",
                "Pre-Training",
                "Pre-training sets the reasoning primitives that later stages can recombine but can hardly supply.",
                (
                    Leaf("data", "Data Curation", "Control what the corpus contains: coverage of the required primitives and the share of facts that must be inferred rather than recalled."),
                    Leaf("opt", "Optimizer Design", "Change how the corpus is fit, since solutions with equal training loss can behave differently out of distribution."),
                ),
            ),
            Section(
                "mid",
                "Mid-Training",
                "An intermediate stage before task adaptation, so that post-training does not start out of distribution.",
                (
                    Leaf("cpt", "Continued Pre-Training", "Next-token prediction on raw text from an under-represented domain."),
                    Leaf("ift", "Intermediate Fine-Tuning", "Training on solved problems that need no domain knowledge, so that the acquired reasoning transfers and prepares later RL."),
                ),
            ),
            Section(
                "sft",
                "Post-Training: Supervised Fine-Tuning",
                "Imitation of demonstrations, which tends to absorb spurious surface correlations unless the problems, solutions, or objective are changed.",
                (
                    Leaf("problem", "Problem-Level Methods", "Represent the anticipated test shift among the training problems.", ("Problem rewriting", "Problem generation")),
                    Leaf("solution", "Solution-Level Methods", "Rewrite the imitated solutions, since imitation generalizes only as far as its target exposes a reusable procedure.", ("Abstract solutions", "Stepwise solutions")),
                    Leaf("objective", "Objective-Level Methods", "Change the loss so that what would not hold under shift contributes less to the update.", ("Teacher-based objectives", "Regularization-based objectives")),
                ),
                figure_name="SFT",
                parent="Post-Training",
            ),
            Section(
                "rl",
                "Post-Training: Reinforcement Learning",
                "Rewards replace reference solutions; transfer depends on the practiced environments, the rewarded behavior, and the update rule, and stays bounded by the base model.",
                (
                    Leaf("scope", "Scope and Limits of RL Transfer", "Controlled evidence on when RL transfers where SFT degrades, and on the base-model and depth limits of that transfer."),
                    Leaf("env", "Environment Design", "Change the problems and checkers the model practices on: generated puzzles, composed environments, chained horizons, games, and controllers."),
                    Leaf("reward", "Reward Design", "Change what is rewarded beyond the final answer: process, structure, abstraction, cross-lingual consistency, and diversity."),
                    Leaf("policy", "Policy Optimization", "Change how rewards become updates: advantage correction, external solutions, and gradient or sharpness regularization."),
                ),
                figure_name="RL",
                parent="Post-Training",
            ),
            Section(
                "hybrid",
                "Post-Training: Hybrid Training",
                "Demonstrations introduce procedures the model cannot find alone; rewards recover the OOD accuracy that SFT reduces.",
                (
                    Leaf("seq", "Sequential Training", "SFT followed by RL, including methods that restore plasticity before the handoff."),
                    Leaf("joint", "Joint Training", "Demonstration and reward signals combined within one stage, in the sampled solutions or in the loss."),
                ),
                figure_name="Hybrid",
                parent="Post-Training",
            ),
        ),
    ),
    Pillar(
        "inference",
        "Inference for Generalization",
        "How can a fixed model be deployed on unfamiliar problems by changing computation at test time, and where does that stop helping?",
        (
            Section(
                "trajectory",
                "Trajectory Restructuring",
                "Impose structural priors on the reasoning trajectory without updating weights.",
                (
                    Leaf("decomposition", "Structural Decomposition", "Split long or hard problems into units that are solved in sequence or by independent micro-steps."),
                    Leaf("injection", "Upfront Structural Injection", "Fix constraints, a task-specific procedure, query-specific demonstrations, or hints before reasoning begins."),
                ),
            ),
            Section(
                "compute",
                "Compute Scaling",
                "Allocate additional test-time compute to explore alternative paths or to revise defective ones.",
                (
                    Leaf("search", "Search Expansion and Allocation", "Direct search toward promising branches and balance exploration against revision by estimated difficulty."),
                    Leaf("revision", "Targeted Revision", "Concentrate computation on localized faulty steps instead of regenerating the whole solution."),
                ),
            ),
            Section(
                "assessment",
                "State Assessment",
                "Decide which intermediate reasoning states remain trustworthy under distribution shift.",
                (
                    Leaf("internal", "Internal State Assessment", "Use hidden-state and uncertainty signals to continue, switch strategy, stop, or recalibrate online."),
                    Leaf("verifier", "Verifier-Mediated Assessment", "Use auxiliary verifiers to select candidates or to build pseudo-labels for test-time training."),
                ),
            ),
            Section(
                "externalization",
                "Inference-Time Externalization",
                "Move working state, exact operations, or accumulated experience outside a single forward reasoning process.",
                (
                    Leaf("within", "Within-Episode Externalization", "External memory, abstract reasoning with tool-supplied values, and decisions about when a tool is needed."),
                    Leaf("cross", "Cross-Episode Externalization", "Persistent memories of strategies distilled from earlier successes and failures, retrieved for later tasks."),
                ),
            ),
        ),
    ),
    Pillar(
        "architecture",
        "Architecture for Generalization",
        "Which computational structures let a learned operation be reused at greater length, depth, or compositional complexity?",
        (
            Section(
                "depth",
                "Recurrent Depth",
                "Apply a shared transformation repeatedly within one prediction.",
                (
                    Leaf("sharing", "Weight Sharing", "Universal and looped transformers, recurrence over facts or tool calls, and recurrent blocks retrofitted into pretrained models."),
                    Leaf("adaptive", "Adaptive Computation", "Learned controllers that choose how many iterations, layers, or latent passes an input receives."),
                    Leaf("stability", "Stable Recurrent Dynamics", "Fixed points, attractors, and energy descent that keep extra iterations useful instead of drifting."),
                ),
            ),
            Section(
                "routing",
                "Information Routing",
                "Specify how tokens are addressed, which dependencies attention reaches, and which modules process them.",
                (
                    Leaf("position", "Positional Encoding", "Encode each token's computational role so that alignment survives longer or differently composed inputs."),
                    Leaf("attention", "Attention Patterns", "Constrain which intermediate results later computation can retrieve, preserving local operations at longer lengths."),
                    Leaf("modular", "Modular Reasoning", "Route parts of an input to specialized heads, experts, or neuro-symbolic components that can be recombined."),
                ),
            ),
            Section(
                "memory",
                "Recurrent Memory",
                "Carry an explicit state across tokens or reasoning chunks so later computation can reuse earlier results.",
                (
                    Leaf("token", "Token-Level Recurrence", "Each input or observation updates a carried state."),
                    Leaf("chunk", "Chunk-Level Recurrence", "A compact state summarizes one reasoning segment for the next."),
                ),
            ),
        ),
    ),
    Pillar(
        "analysis",
        "Analysis of Generalization",
        "When does reasoning transfer, and what explains its successes and failures?",
        (
            Section(
                "behavioral",
                "Behavioral Evidence",
                "Performance under named shifts: one row of the transfer matrix, not a single OOD score.",
                (
                    Leaf("composition", "Compositional Generalization", "Fixed primitives, new combinations: deeper and wider computation graphs, multi-hop chaining, and rule extrapolation."),
                    Leaf("length", "Length, Depth, and Difficulty", "Longer inputs, deeper nesting or planning horizons, and easy-to-hard transfer, which diverge even within one task."),
                    Leaf("prior", "Prior Exposure", "How far a test lies from earlier exposure: perturbed templates, counterfactual conventions, unseen directions, and contamination-limited tasks."),
                ),
            ),
            Section(
                "mechanistic",
                "Mechanistic Evidence",
                "Whether one shared computation or separate shortcuts produce the behavior, mostly in small controlled models.",
                (
                    Leaf("circuits", "Shared Circuits", "Overlap between the circuits used on both sides of a shift predicts transfer, but may fail on new knowledge."),
                    Leaf("interfaces", "Interfaces Between Steps", "Intermediate results that are computed but do not reach the next step in a usable form."),
                    Leaf("dynamics", "Learning Dynamics", "Which of several fitting solutions optimization selects, and when: grokking, critical windows, and checkpoint choice."),
                ),
            ),
            Section(
                "theoretical",
                "Theoretical Evidence",
                "Results under explicit assumptions that separate expressivity, learnability, and certification.",
                (
                    Leaf("learnability", "Learnability", "Whether finite training selects an extrapolating procedure over shortcuts: locality, inductive bias, scratchpads, curricula, and data diversity."),
                    Leaf("certification", "Certification", "Whether a finite test can establish extrapolation: length-generalization bounds and their limits."),
                ),
            ),
        ),
    ),
)


def leaves() -> list[tuple[Pillar, Section, Leaf]]:
    """All leaves in reading order."""
    return [(p, s, leaf) for p in TAXONOMY for s in p.sections for leaf in s.leaves]


def leaf_index() -> dict[str, tuple[Pillar, Section, Leaf]]:
    return {f"{p.id}.{s.id}.{leaf.id}": (p, s, leaf) for p, s, leaf in leaves()}
