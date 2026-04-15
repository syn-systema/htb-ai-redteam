---
module_number: 01
module_name: "Fundamentals of AI"
status: in-progress
difficulty: "Medium"
tier: ""
estimated_time: ""
sections_total: 24
sections_done: 11
started: "2026-04-14"
completed: ""
---

# Module 01 — Fundamentals of AI

## Overview

Theoretical foundation for AI/ML/DL. Covers supervised, unsupervised, and reinforcement learning, plus deep learning and generative AI. No hands-on exercises in this module — pure concept-building. Math (linear algebra, calculus, statistics) is touched but not the focus.

**Why it matters for AI Red Teaming:** every later module in the path (data attacks, evasion, privacy) assumes you know what an "MLP" is, what "gradient descent" optimizes, and why "overfitting" creates an attack surface. This module is the vocabulary base.

## Prerequisites

- Basic statistics, linear algebra, calculus (recommended)
- Python programming fundamentals

## Sections

### 1. Introduction to Machine Learning

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

The taxonomy: **AI ⊃ ML ⊃ Neural Nets ⊃ DL.** Concentric circles — each inner ring is a specialization of the outer one.

#### AI — the umbrella

Field of building systems that perform tasks normally requiring human cognition: language, vision, reasoning, decision-making, learning. Major sub-areas:

| Sub-area | What it does |
|---|---|
| Natural Language Processing (NLP) | Understand, interpret, generate human language |
| Computer Vision | Interpret images and video |
| Robotics | Autonomous or guided physical action |
| Expert Systems | Mimic human expert decision-making (rule-based) |

Goal: **augment** human capabilities, not replace. Domains: healthcare (diagnosis, drug discovery), finance (fraud detection, trading), cybersecurity (threat detection).

#### ML — the learning sub-field

ML systems learn patterns from data instead of following hard-coded rules. Three paradigms:

| Paradigm | Data | Feedback signal | Examples |
|---|---|---|---|
| **Supervised** | Labeled (input → known output) | Loss vs. label | Image classification, spam detection, fraud prevention |
| **Unsupervised** | Unlabeled | Internal structure (e.g. cluster cohesion) | Customer segmentation, anomaly detection, dimensionality reduction |
| **Reinforcement** | State + action transitions | Reward / penalty from environment | Game playing, robotics, autonomous driving |

Canonical example: train a classifier on labeled cat/dog images → it learns features that distinguish them → predicts the class of an unseen image.

#### DL — the multi-layer-network sub-sub-field

Subset of ML using neural networks with **many layers** (hence "deep"). Three properties that make DL special:

1. **Hierarchical feature learning** — early layers detect primitives (edges, textures), deeper layers compose them into higher-level concepts (shapes, objects).
2. **End-to-end learning** — raw input → desired output, no manual feature engineering between.
3. **Scalability** — performance keeps improving with more data + more compute, unlike most classical ML which plateaus.

Common DL architectures:

| Architecture | Best for | Mechanism |
|---|---|---|
| **CNN** (Convolutional NN) | Images, video | Convolutional kernels detect local spatial patterns; pooling builds hierarchy |
| **RNN** (Recurrent NN) | Sequential data (text, speech, time series) | Loops carry hidden state across time steps |
| **Transformer** | NLP, increasingly vision | Self-attention models long-range dependencies in parallel (no recurrence) |

State-of-the-art results in: image classification + object detection (CV), translation + sentiment + text generation (NLP), speech-to-text + TTS, RL agents (AlphaGo-class).

#### How they fit together

```
┌─────────────────────────── Artificial Intelligence ─────────────────────────────┐
│  goal: build systems that exhibit intelligent behavior                         │
│                                                                                  │
│   ┌────────────────── Machine Learning ──────────────────────────┐              │
│   │  goal: learn patterns from data instead of being programmed  │              │
│   │                                                                │              │
│   │   ┌──────── Neural Networks ────────┐                          │              │
│   │   │                                  │                          │              │
│   │   │   ┌─── Deep Learning ───┐       │                          │              │
│   │   │   │  many-layer NNs     │       │                          │              │
│   │   │   └──────────────────────┘       │                          │              │
│   │   └────────────────────────────────────┘                        │              │
│   └──────────────────────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────────────────┘
```

ML provides the *learning* mechanism that lets AI adapt. DL provides the *representation power* that lets ML handle messy, high-dimensional data. Together they enable the modern AI stack — autonomous driving (CV + RL), conversational AI (transformers), drug discovery (graph NNs), etc.

#### Red-team-relevant takeaways

- **Every ML/DL system has a training-time and inference-time attack surface.** Training-time = data poisoning, label flipping, trojans. Inference-time = adversarial examples, prompt injection, model evasion.
- **Hierarchical feature learning is what evasion attacks exploit** — small perturbations that don't change the human-readable input can drastically change what deep features the model extracts.
- **Supervised models overfit on memorized data points** → membership inference, the basis of AI Privacy module 11.
- **Unstructured-data DL models (CNN, transformer) are the targets across modules 06–11.** Knowing the architecture tells you which attack family applies.

**Takeaways:**
- AI is the goal (intelligent behavior); ML is the dominant *technique* (learning from data); DL is the dominant *ML technique* for unstructured data (deep neural nets).
- Three ML paradigms (supervised / unsupervised / reinforcement) — each maps to specific real-world tasks and to specific attack classes later in the path.
- CNN / RNN / Transformer are the three architecture families to know by name.

---

### 2. Mathematics Refresher for AI

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

Reference page — bookmark, return when an unfamiliar symbol shows up. HTB explicitly says you don't need to memorize this; treat it as a lookup. Math here is rendered with MathJax (works natively in Obsidian + GitHub; Antigravity needs the **Markdown All in One** extension installed).

#### Algebraic notation

| Symbol | Meaning | Example |
|---|---|---|
| $x_t$ | Variable $x$ at index/time $t$ | RNN hidden state $h_t$ at timestep $t$ |
| $x^n$ | $x$ raised to the $n$-th power | $x^2 = x \cdot x$ |
| $\sum_{i=1}^{n} a_i$ | Sum $a_1 + a_2 + \dots + a_n$ | Loss = $\sum_{i=1}^{N} \ell(y_i, \hat{y}_i)$ |
| $\dots$ (ellipsis) | "and so on" — a continuing pattern | $a_1 + a_2 + \dots + a_n$ |
| $\frac{1}{x}$ | Reciprocal of $x$ | $\frac{1}{5} = 0.2$ |

#### Norms — **critical for evasion attacks (modules 08–10)**

A norm measures the "size" of a vector. Different norms = different geometric shapes for the perturbation budget. **The choice of norm IS the attack family.**

| Norm | Formula | Geometric shape | Attack family |
|---|---|---|---|
| $L_\infty$ | $\lVert v \rVert_\infty = \max_i \lvert v_i \rvert$ | Hypercube (each pixel can move up to $\epsilon$) | **FGSM, I-FGSM** (module 09) |
| $L_2$ | $\lVert v \rVert_2 = \sqrt{\sum_i v_i^2}$ | Hypersphere (Euclidean) | **DeepFool** (module 09) |
| $L_1$ | $\lVert v \rVert_1 = \sum_i \lvert v_i \rvert$ | Octahedron (sparsity-inducing) | **ElasticNet** (module 10) |
| $L_0$ | $\lVert v \rVert_0 = \#\{i : v_i \neq 0\}$ | "How many features were touched" (pseudo-norm) | **JSMA** (module 10) |

Also used for: regularization (preventing overfitting), data normalization, distance metrics in clustering.

#### Logarithms and exponentials

| Symbol | Meaning | Where it shows up |
|---|---|---|
| $\log_2(x)$ | Log base 2 | Information theory: entropy = $-\sum p_i \log_2 p_i$ (bits) |
| $\ln(x)$ | Natural log (base $e$) | Cross-entropy loss, KL divergence, log-likelihood |
| $e^x$ | Euler's number to the $x$ | Softmax, sigmoid, normal distribution, growth/decay |
| $2^x$ | 2 to the $x$ | Binary, bit-counts |

The natural log is everywhere in ML because $\frac{d}{dx} \ln(x) = \frac{1}{x}$ — clean derivatives → clean gradients → trainable models.

#### Matrix and vector operations — **the language of neural networks**

| Op | Notation | What it does |
|---|---|---|
| Matrix-vector product | $A v$ | Linear layer forward pass: outputs = weights × inputs |
| Matrix-matrix product | $A B$ | Composing two linear transformations (stacked layers) |
| Transpose | $A^T$ | Swap rows/columns (e.g., $A^T A$ in normal equations) |
| Inverse | $A^{-1}$ | Undo a transformation; $A A^{-1} = I$ (only for square non-singular $A$) |
| Determinant | $\det(A)$ | Scalar; $\det(A) = 0$ ⇒ matrix is singular (not invertible). Volume scaling factor of the transformation |
| Trace | $\mathrm{tr}(A)$ | Sum of diagonal entries; equals sum of eigenvalues |

Worked example (matrix-vector multiply):

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad v = \begin{bmatrix} 5 \\ 6 \end{bmatrix}, \quad A v = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 6 \\ 3 \cdot 5 + 4 \cdot 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}
$$

Worked example (transpose):

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad A^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}
$$

#### Eigenvalues and eigenvectors — **the engine of PCA, saliency, and PageRank**

For a square matrix $A$, an **eigenvector** $v$ is a non-zero vector that $A$ only stretches (or shrinks), without rotating. The stretch factor $\lambda$ is the **eigenvalue**:

$$
A v = \lambda v
$$

| Concept | Meaning |
|---|---|
| Eigenvalue $\lambda$ | Scalar stretch factor along an eigenvector direction |
| Eigenvector $v$ | Direction preserved (up to scaling) by the transformation $A$ |

Where this matters in the path:
- **PCA:** principal components are the eigenvectors of the data's covariance matrix; eigenvalues = variance along each direction. Used for dimensionality reduction (module 01 §11).
- **Saliency maps:** gradient-based attacks (JSMA in module 10) compute Jacobian eigenstructure to find the most-influential input features.
- **Spectral analysis** of model weight matrices reveals which directions in input space the model is most/least sensitive to — relevant for adversarial transferability.

#### Set theory

| Symbol | Meaning | Example |
|---|---|---|
| $\lvert S \rvert$ | Cardinality (count of elements) | $\lvert \{1,2,3,4,5\} \rvert = 5$ |
| $A \cup B$ | Union (in either) | $\{1,2,3\} \cup \{3,4,5\} = \{1,2,3,4,5\}$ |
| $A \cap B$ | Intersection (in both) | $\{1,2,3\} \cap \{3,4,5\} = \{3\}$ |
| $A^c$ (or $\bar{A}$) | Complement (not in $A$, relative to a universe $U$) | $U=\{1..5\}, A=\{1,2,3\} \Rightarrow A^c = \{4,5\}$ |
| $x \in S$ | $x$ is an element of $S$ | $3 \in \{1,2,3\}$ |
| $A \subseteq B$ | $A$ is a subset of $B$ | $\{1,2\} \subseteq \{1,2,3\}$ |

Used in defining datasets (training set, test set, support of a distribution), measure-theoretic probability, and combinatorial bounds in privacy proofs (module 11).

#### Comparison and equality

| Symbol | Meaning |
|---|---|
| $a \geq b$ | $a$ greater than or equal to $b$ |
| $a \leq b$ | $a$ less than or equal to $b$ |
| $a = b$ | Equal (math) |
| `a == b` | Equality test (code) |
| $a \neq b$ | Not equal (math) |
| `a != b` | Inequality test (code) |
| $\max(\dots)$ | Largest element of a set |
| $\min(\dots)$ | Smallest element of a set |

#### Functions

| Symbol | Meaning |
|---|---|
| $f(x)$ | Function $f$ applied to input $x$ |
| $f: \mathcal{X} \to \mathcal{Y}$ | $f$ maps from set $\mathcal{X}$ to set $\mathcal{Y}$ |
| $f(x) = x^2 + 2x + 1$ | Concrete function definition |

In ML: a model is a function $f_\theta(x) \to \hat{y}$ parameterized by weights $\theta$. Training adjusts $\theta$ to minimize a loss function $\mathcal{L}(y, \hat{y})$.

#### Probability and statistics — **load-bearing for everything**

| Symbol | Meaning | Where it shows up |
|---|---|---|
| $P(x)$ | Probability of $x$ | Naive Bayes (module 02 spam) |
| $P(x \mid y)$ | Conditional probability of $x$ given $y$ | Bayesian inference, classifier outputs $P(\text{class} \mid \text{input})$ |
| $\mathbb{E}[X]$ | Expected value of random variable $X$: $\sum_i x_i P(x_i)$ | Loss is "expected risk" — every training algorithm |
| $\mathrm{Var}(X)$ | Variance: $\mathbb{E}[(X - \mathbb{E}[X])^2]$ | Spread of predictions, model confidence |
| $\sigma(X)$ | Standard deviation: $\sqrt{\mathrm{Var}(X)}$ | Same as variance, in original units |
| $\mathrm{Cov}(X, Y)$ | Covariance: $\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$ | PCA (covariance matrix → eigenvectors) |
| $\rho(X, Y)$ | Pearson correlation: $\frac{\mathrm{Cov}(X,Y)}{\sigma(X) \sigma(Y)} \in [-1, 1]$ | Feature analysis, dataset auditing |

Worked example — expectation:

$$
\mathbb{E}[X] = \sum_i x_i \, P(x_i)
$$

Worked example — variance:

$$
\mathrm{Var}(X) = \mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right]
$$

These show up *constantly*:
- **Cross-entropy loss** = expected negative log-likelihood under the true distribution.
- **Membership Inference Attacks** (module 11) exploit the variance gap between predictions on training vs. non-training data.
- **DP-SGD** (module 11) bounds the per-sample influence on parameters using Gaussian noise calibrated to the gradient's L2 sensitivity.

#### Greek letters you'll see constantly

| Letter | Common use |
|---|---|
| $\alpha$ (alpha) | Learning rate, attack step size |
| $\beta$ (beta) | Momentum coefficient, ElasticNet regularizer mix |
| $\gamma$ (gamma) | Discount factor in RL, kernel parameter in SVM |
| $\epsilon$ (epsilon) | Perturbation budget (the L_p radius), small constant for numeric stability |
| $\theta$ (theta) | Model parameters (weights + biases) |
| $\lambda$ (lambda) | Eigenvalue, regularization strength (L1/L2 penalty), DP privacy parameter |
| $\mu$ (mu) | Mean of a distribution |
| $\sigma$ (sigma) | Standard deviation; also the sigmoid function $\sigma(x) = \frac{1}{1+e^{-x}}$ |
| $\nabla$ (nabla) | Gradient operator: $\nabla_\theta \mathcal{L}$ = gradient of loss w.r.t. parameters |

The gradient $\nabla$ is the most important symbol for evasion attacks — modules 09 and 10 are entirely about exploiting it.

**Takeaways:**
- Norms ($L_\infty, L_2, L_1, L_0$) define perturbation budgets — each maps to a specific evasion attack family.
- Matrix-vector products are how every neural network layer actually computes.
- Eigenvalues / eigenvectors are the core of PCA and saliency.
- $\mathbb{E}[\cdot]$, $\mathrm{Var}(\cdot)$, $P(x \mid y)$ are the foundations of every loss function and privacy-attack metric.
- Greek-letter glossary: $\theta$ = weights, $\epsilon$ = attack budget, $\nabla$ = gradient, $\lambda$ = eigenvalue/regularizer.

---

### 3. Supervised Learning Algorithms

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

Supervised learning = learn a function $f_\theta(x) \to \hat{y}$ from a dataset of $(x, y)$ pairs where $y$ is the **known correct answer**. Algorithm adjusts parameters $\theta$ to make $\hat{y}$ match $y$ on training data, then we hope it generalizes to inputs it's never seen.

#### Two flavors

| Flavor | Output type | Examples | Where in path |
|---|---|---|---|
| **Classification** | Discrete category | Spam vs. ham, cat/dog/bird, malware family | **Modules 02, 06–11 — most attacks target classifiers** |
| **Regression** | Continuous value | House price, stock forecast, temperature | Less attacked but same vocabulary; regression heads on neural nets are common |

#### Core vocabulary

| Term | Meaning | Tactical note |
|---|---|---|
| **Training data** | The labeled $(x, y)$ examples the model learns from | Ground truth — if it's poisoned (module 06), the model is poisoned |
| **Features** | Measurable properties of $x$ (size, pixels, byte sequence) | Attackers manipulate these in evasion (modules 08–10) |
| **Labels** | The known correct outputs $y$ | Attackers flip these in label-flipping attacks (module 06) |
| **Model** | The learned function $f_\theta$ | The artifact under attack |
| **Training** | Optimizing $\theta$ to minimize loss on training data | Where data poisoning + trojan attacks land |
| **Prediction** | Producing $\hat{y}$ for a new $x$ | Output side — what users see |
| **Inference** | Broader: prediction + extracting structure (e.g. feature importances, parameters) | "**Inference attacks**" target this phase — membership inference (module 11), model extraction (module 07) |
| **Evaluation** | Measuring model quality on held-out data | Same metrics measure attack success ("attack success rate" = error rate of the attacked model) |

#### Evaluation metrics

For binary classification, predictions land in a confusion matrix:

|  | Predicted positive | Predicted negative |
|---|---|---|
| **Actual positive** | True Positive (TP) | False Negative (FN) |
| **Actual negative** | False Positive (FP) | True Negative (TN) |

The four standard metrics:

| Metric | Formula | What it measures | Best when |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall fraction correct | Classes are balanced |
| **Precision** | $\frac{TP}{TP + FP}$ | Of the positives we predicted, how many were right? | False positives are costly (e.g. spam: don't drop legit mail) |
| **Recall** (sensitivity) | $\frac{TP}{TP + FN}$ | Of the actual positives, how many did we catch? | False negatives are costly (e.g. malware detection, cancer screening) |
| **F1** | $2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$ | Harmonic mean of precision + recall | You need a single number balancing both |

Why the **harmonic** mean for F1, not arithmetic? Harmonic mean punishes imbalance — if precision = 1.0 and recall = 0.0, arithmetic mean = 0.5 (looks OK), harmonic mean = 0 (correctly says "useless").

**Red-team angle:** an evasion attack (modules 08–10) is exactly the act of pushing a malicious sample from "predicted positive (caught)" to "predicted negative (false negative)". Attack success rate = model's recall drop on adversarial inputs.

#### Generalization — the goal

A model that performs well on **unseen** data has **generalized** the patterns. The two failure modes:

```
        ┌─────────────────────────────────────────────────────────────┐
loss    │                                                              │
        │ ●  underfit                                                  │
        │  ● ─                                                         │
        │   ●  ─ ─                              train loss             │
        │    ●     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                          │
        │     ●─                          ─ ─ ─ ─ ─ ─                  │
        │      ●  sweet spot                          test loss        │
        │       ●─ ─ ─                                                 │
        │            ●─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ●  overfit   │
        │                                                          ●   │
        └─────────────────────────────────────────────────────────────┘
                            ←  model complexity  →
```

| Failure mode | Symptom | Fix |
|---|---|---|
| **Underfitting** | High error on BOTH training and test data | Bigger / more expressive model, more features, less regularization |
| **Overfitting** | Low training error, high test error | More data, regularization, cross-validation, simpler model, dropout, early stopping |

#### Cross-validation — measuring generalization honestly

Splitting once into train/test gives one noisy estimate. **k-fold cross-validation** instead:

```
Data = [ A | B | C | D | E ]   (5 folds)

Fold 1: train on B,C,D,E   test on A
Fold 2: train on A,C,D,E   test on B
Fold 3: train on A,B,D,E   test on C
Fold 4: train on A,B,C,E   test on D
Fold 5: train on A,B,C,D   test on E

Final score = mean of 5 test scores
```

Every example gets used for both training (4 times) and testing (once). Tighter, less noisy estimate of generalization error than a single split.

**Red-team angle:** **shadow model attacks** for membership inference (module 11) build many "shadow" models on different data splits — same mechanic as cross-validation, but the goal is to learn what overfitting *looks like* so you can detect it remotely.

#### Regularization — preventing overfitting by penalizing complexity

Add a penalty term to the loss so the optimizer prefers smaller weights:

$$
\mathcal{L}_{\text{regularized}}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \lambda \cdot R(\theta)
$$

where $\lambda$ is the regularization strength and $R(\theta)$ is the penalty.

| Type | Penalty $R(\theta)$ | Effect | Connection |
|---|---|---|---|
| **L1** (Lasso) | $\lVert \theta \rVert_1 = \sum_j \lvert \theta_j \rvert$ | Drives many weights to **exactly zero** → sparse models, automatic feature selection | Same $L_1$ norm as in Section 2; same idea as ElasticNet attack (module 10) — sparsity-inducing |
| **L2** (Ridge / weight decay) | $\lVert \theta \rVert_2^2 = \sum_j \theta_j^2$ | Shrinks all weights smoothly toward zero (none reach zero) | Same $L_2$ norm; the default regularizer in deep nets via "weight decay" |

So the L1/L2 norms you saw in Section 2 do double duty: they're used to (a) **measure** vector size in attacks, and (b) **regularize** training to fight overfitting. Same math, different role.

#### Red-team takeaways

- **Inference is the broader concept; "inference attack" is a term of art** for *attacks against the inference phase* — module 11 membership inference, module 07 model extraction.
- **Overfitting is the source of multiple attack surfaces:** memorized training points → membership inference; over-confident wrong answers → easier evasion; capacity to memorize triggers → trojan attacks survive.
- **Precision/recall trade-off shows up as detector tuning.** A spam filter operator (or malware classifier) sets a threshold trading FP for FN. Attackers exploit whichever side of the trade-off the defender chose.
- **Regularization (L1/L2) is one defense against overfitting-derived attacks**, but it doesn't solve them — it just makes them harder. Differential privacy (module 11 DP-SGD) is the principled defense.
- **Cross-validation methodology = shadow model methodology.** Same trick, repurposed: defenders use it to honestly estimate generalization; attackers use it to learn what generalization-failures look like remotely.

**Takeaways:**
- Two problem types: classification (categorical) vs regression (continuous).
- Core loop: training data → features + labels → model → evaluation → iterate.
- Four standard metrics: accuracy, precision, recall, F1 — F1 is the harmonic balance.
- Generalization is the goal; overfitting and underfitting are the failure modes.
- Cross-validation = honest generalization estimate; regularization (L1/L2) = complexity penalty.

---

### 4. Linear Regression

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A supervised algorithm that fits a **straight line** (or hyperplane in higher dimensions) through the training data, predicting a **continuous** target $y$ from one or more predictors $x$. Simplest possible learning algorithm — and the conceptual base of everything that follows.

#### Where it sits

| Question | Answer |
|---|---|
| Supervised? | Yes |
| Output type | Continuous (regression, not classification) |
| Decision boundary | A line / plane / hyperplane |
| Has a closed-form solution? | **Yes (uniquely)** — see OLS below |
| Need iteration / gradient descent? | No (but you can use it; for huge datasets, iterative is faster) |

#### Simple linear regression — one predictor

$$
y = m x + c
$$

| Symbol | Meaning |
|---|---|
| $y$ | Predicted target |
| $x$ | Predictor (input feature) |
| $m$ | Slope — "how much $y$ changes per unit change in $x$" |
| $c$ | Intercept — value of $y$ when $x = 0$ |

Same equation you saw in algebra class. Training = "find the $m$ and $c$ that make this line fit the cloud of $(x, y)$ training points as closely as possible."

#### Multiple linear regression — many predictors

$$
y = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n
$$

| Symbol | Meaning |
|---|---|
| $b_0$ | Intercept (the constant) |
| $b_1, \dots, b_n$ | **Coefficients** — one per feature |
| $x_1, \dots, x_n$ | Features |

Each coefficient $b_j$ tells you "how much $y$ changes per unit change in feature $x_j$, holding the others constant." In neural-net terminology these coefficients are called **weights**.

In matrix form (you'll see this constantly):

$$
\hat{y} = X \beta
$$

where $X$ is the data matrix (rows = samples, columns = features), $\beta$ is the vector of coefficients, $\hat{y}$ is the predicted-values vector. This `X β` is exactly the matrix-vector multiplication from Section 2.

#### Ordinary Least Squares (OLS) — finding the best line

We need to define "best fit." OLS picks coefficients that minimize the **sum of squared residuals**:

$$
\text{RSS}(\beta) = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{N} (y_i - X_i \beta)^2
$$

Visualizing the residuals (vertical distance from data point to fitted line):

```
        ↑
      y │            data point ●
        │                       ╎ ← residual = (y - ŷ)
        │  fitted line ─ ─ ─ ─ ╎ ─ ─ ─ ─ ─ ─ ─
        │                                    ●
        │                                    ╎
        │                                    ╎
        │           ●                        ╎
        │           ╎                        
        │           ╎                        
        └────────────────────────────────────→
                                             x
```

**Why squared?**
- All residuals become positive (no cancellation).
- Larger errors get penalized disproportionately (a residual of 4 contributes 16, but two residuals of 2 contribute only 4+4=8).
- The squared-error function is differentiable and convex → unique minimum, easy to solve.

#### Closed-form solution — the **normal equations**

For OLS, the optimal coefficients have a *direct algebraic* solution (no iteration needed):

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

This is **rare**. Linear regression with squared loss is essentially the only practical ML algorithm with a closed-form optimum. Every other algorithm in this path (logistic regression, SVMs, neural nets) requires **iterative optimization** (gradient descent variants). Linear regression is the textbook case where you can see "the optimum exists, here's the formula" — every later algorithm is "the optimum exists somewhere, search for it."

#### Assumptions

OLS predictions are only reliable if the data satisfies four assumptions:

| Assumption | Meaning | What breaks if violated |
|---|---|---|
| **Linearity** | The true relationship is actually linear | Curved patterns get systematically mis-fit; need polynomial features or a non-linear model |
| **Independence** | Each observation is independent of the others | Time-series correlation breaks standard error estimates; need ARIMA / autoregressive models |
| **Homoscedasticity** | Residuals have **constant variance** across the range of $x$ | If residuals fan out (more error at large $x$), confidence intervals are wrong |
| **Normality of errors** | Residuals are normally distributed | Affects significance tests + confidence intervals (predictions themselves are still fine) |

Mnemonic: **L.I.N.E.** — Linearity, Independence, Normality, Equal variance (homoscedasticity).

#### Red-team angles

- **Linear models are reverse-engineerable.** Coefficients $\beta$ leak directly which features the model relies on. **Module 07's model reverse engineering** exploits this — and it's much harder against deep nets, but linear regression is the trivially-easy case. If a system uses linear regression and exposes predictions, you can recover $\beta$ via algebra.
- **The squared loss appears in adversarial-attack objectives too.** When an attacker minimizes $\lVert \delta \rVert^2$ subject to a misclassification constraint (DeepFool, ElasticNet — modules 09–10), they're solving a regularized least-squares problem. Same math, weaponized.
- **OLS is hyper-sensitive to outliers** because residuals are squared — one bad point can swing the line. **Data poisoning attacks** (module 06) often lean on this: a few crafted training points can disproportionately distort the model.
- **Coefficient = weight.** When module 04 talks about "model weights" and module 07 talks about "weight extraction", those are the multi-feature analogues of the $b_j$ coefficients here.

**Takeaways:**
- $y = mx + c$ (simple) → $y = X \beta$ (multiple) → $\hat{\beta} = (X^T X)^{-1} X^T y$ (OLS closed form).
- "RSS" = sum of squared residuals = the loss being minimized.
- Linear regression is the **only** algorithm in the path with a closed-form optimum; everything else needs iterative gradient descent.
- L.I.N.E. assumptions: Linearity, Independence, Normality, Equal variance.
- Coefficients = weights. Reverse engineering them = the simplest case of model extraction.

---

### 5. Logistic Regression

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

**Misnamed.** Logistic regression is a **classifier**, not a regressor. The name comes from the *logistic function* (sigmoid) it uses internally. It's binary classification's workhorse and the conceptual bridge from linear regression to neural networks.

#### The pipeline

```
features ──► linear combination ──► sigmoid ──► probability ──► threshold ──► class
   x      z = β₀ + β₁x₁ + … + βₙxₙ    σ(z)         p ∈ [0,1]      p ≥ 0.5?     0 or 1
```

**Logistic regression = linear regression + sigmoid.** Same $X \beta$ from §4, then squashed through $\sigma$ to land in $[0, 1]$ so we can read it as a probability.

#### The sigmoid function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

S-shaped curve (where the name comes from):

```
 σ(z)
  1 ┤                           ╭────────────────
    │                       ╭───╯
    │                    ╭──╯
  0.5┤                ╭──╯
    │             ╭──╯
    │         ╭──╯
    │     ╭──╯
  0 ┤────╯
    └─────────────────────────────────────────→ z
        −6      −2    0    2       6
```

Properties that make sigmoid the right choice for binary classification:

| Property | Why it matters |
|---|---|
| Maps $(-\infty, +\infty) \to (0, 1)$ | Output reads as a probability |
| Smooth, differentiable everywhere | Gradient descent works (unlike a hard step function) |
| $\sigma(0) = 0.5$ | Natural classification threshold |
| $\sigma(z) + \sigma(-z) = 1$ | Symmetric around 0 — flipping sign of $z$ flips the predicted class |

The full prediction:

$$
p = \sigma(X \beta) = \frac{1}{1 + e^{-X\beta}}
$$

#### Logits and log-odds (terminology you'll meet constantly)

The raw $z = X \beta$ value (before sigmoid) is called the **logit**. It's the *inverse* of the sigmoid:

$$
z = \log\!\left(\frac{p}{1 - p}\right)
$$

That ratio $\frac{p}{1-p}$ is the **odds** (e.g. "3-to-1 odds" = probability $0.75$). Its log = **log-odds**. Logistic regression assumes log-odds are linear in the features:

$$
\log\!\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n
$$

In deep learning, the term **"logits"** means "the raw network output before softmax/sigmoid" — same concept, generalized. When module 09 talks about "gradient with respect to the logits", it's the gradient of $z$ here.

#### Decision boundary — the central concept

The model classifies $x$ as positive when $p(x) \geq 0.5$, which is equivalent to $z = X\beta \geq 0$. So the boundary is exactly:

$$
X \beta = 0
$$

That equation defines a **hyperplane** in feature space:

| Feature space dimension | "Hyperplane" is actually a... |
|---|---|
| 1D (one feature) | Single point on the number line |
| 2D (two features) | A line splitting the plane |
| 3D (three features) | A flat plane splitting the room |
| $n$D (many features) | An $(n-1)$-dimensional flat subspace |

ASCII view in 2D:

```
 x₂
  │
  │   ●  ●     ●
  │   ●●        ╲
  │  ● ●         ╲ decision boundary: Xβ = 0
  │     ●         ╲
  │                ╲    □ □ □
  │                 ╲  □ □ □ □
  │     class 0      ╲  □  □
  │                   ╲   class 1
  │                    ╲
  └─────────────────────────────→ x₁
```

Everything on one side: $z > 0$, predicted positive ($p > 0.5$). Other side: predicted negative. The line *itself* is where $p = 0.5$ exactly — the model's most uncertain region.

#### Threshold tuning — precision/recall control knob

Default threshold is 0.5, but you can move it:

| Threshold | Effect |
|---|---|
| **Lower** (e.g. 0.3) | More instances cross into "positive" → ↑ recall, ↓ precision (catches more, more false alarms) |
| **Higher** (e.g. 0.8) | Fewer instances classified positive → ↑ precision, ↓ recall (more confident, misses more) |

Same precision/recall trade-off from §3 — now you can see exactly how it's controlled. Spam filter: high threshold (don't drop legit mail). Cancer screening: low threshold (don't miss a case).

#### Assumptions

| Assumption | Meaning |
|---|---|
| **Binary outcome** | Standard logistic regression handles 2 classes. (Multi-class extension: softmax / multinomial logistic regression.) |
| **Linearity of log-odds** | The relationship between features and log-odds is linear (relaxed compared to linear regression — only the *odds* need to be linear, not the probabilities) |
| **Low multicollinearity** | Features shouldn't be highly correlated with each other; otherwise coefficient interpretation breaks down |
| **Large sample size** | Coefficient estimates are noisy with small data |

#### Red-team angles — this is **the** attack-target section

- **Decision boundaries ARE the attack target.** Every evasion attack in modules 08–10 (FGSM, I-FGSM, DeepFool, ElasticNet, JSMA) is mathematically: "find the smallest perturbation $\delta$ such that $x + \delta$ ends up on the *other side* of the decision boundary." For logistic regression specifically, the boundary is a single hyperplane → DeepFool can solve this in **one closed-form step** (project the input onto the hyperplane). That's why HTB introduces DeepFool right after FGSM in module 09.
- **Linear decision boundaries are the easiest to attack.** The attack direction is just the normal vector $\beta$ of the hyperplane — no iteration needed. Neural networks have *piecewise-linear* boundaries (one per ReLU activation region), so attacks generalize but cost more compute.
- **The sigmoid's flat tails create gradient masking opportunities.** When $z$ is very large positive or very negative, $\sigma'(z) \approx 0$. Adversaries-aware defenders sometimes try to push inputs into the flat region to hide gradients (and adversarial-attack-aware attackers counter with logit-targeting methods that bypass the saturated region).
- **Logits are the canonical attack input.** Most modern adversarial attacks compute gradients of *logits* not probabilities — bypasses sigmoid saturation and gives stronger gradient signal. When module 09 says "we attack the pre-softmax outputs," it means logits.
- **Threshold tuning = defender's last knob.** Moving the threshold changes attack difficulty: a defender who lowers the threshold (more aggressive blocking) forces attackers to push inputs further across the boundary. But it also raises false positives — same precision/recall trade-off, now framed adversarially.
- **Module 02's spam classifier** uses Naive Bayes (next sections), but the spam-detection example here is the same problem. Module 04 (prompt injection) uses logistic-regression-style classifiers as content filters; bypassing those is exactly a "cross the decision boundary" task.

**Takeaways:**
- Logistic regression = linear regression $X\beta$ + sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$, then a threshold.
- "Logit" = raw $z$ before sigmoid = log-odds. Used in every NN context.
- Decision boundary = hyperplane where $X\beta = 0$ ($p = 0.5$). This is what evasion attacks target.
- Threshold (default 0.5) is the precision/recall control knob.
- Linear boundaries are easiest to attack; DeepFool exploits this exact case in one step.

---

### 6. Decision Trees

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A different beast from regression-based methods. Decision trees ask a sequence of yes/no questions about features and follow branches to a leaf that holds the prediction. **No equation, no gradient — just nested if-else rules.** Works for both classification and regression.

#### Anatomy of a tree

| Node type | Role |
|---|---|
| **Root** | Starting point; entire dataset enters here |
| **Internal nodes** | Each tests one feature ("Outlook = Sunny?") and routes the sample down a branch |
| **Leaves** | Terminal nodes that output the prediction (class label or numeric value) |

ASCII view (the "Play Tennis" example, hand-drawn):

```
                       Outlook?
                  ┌───────┼───────┐
              Sunny    Overcast   Rainy
                │         │        │
            Humidity?    YES     Wind?
            ┌───┴───┐           ┌───┴───┐
          High    Normal      Strong    Weak
            │       │           │        │
           NO      YES         NO       YES
```

Reading: a sunny + high-humidity day → leaf says NO. Overcast → always YES, no further questions.

#### Building the tree — pick the split that "purifies" the data fastest

At each node, the algorithm scans every feature and chooses the one that **best separates** the classes in the resulting subsets. "Best" = whichever split most reduces *impurity* (synonyms: disorder, uncertainty). Three common impurity measures:

##### Gini impurity

Probability of misclassifying a randomly drawn sample if you label it according to the class distribution:

$$
\text{Gini}(S) = 1 - \sum_{i} p_i^2
$$

where $p_i$ is the proportion of samples in class $i$. Range: $0$ (perfectly pure — one class only) to $0.5$ (max for binary, equal split).

Worked example: 30 of class A, 20 of class B → $p_A = 0.6, p_B = 0.4$:

$$
\text{Gini} = 1 - (0.6^2 + 0.4^2) = 1 - 0.52 = 0.48
$$

##### Entropy

Information-theoretic disorder, in bits:

$$
H(S) = -\sum_{i} p_i \log_2 p_i
$$

Range: $0$ (pure) to $\log_2(k)$ for $k$ classes. For binary: max is $1.0$ (50/50 split).

Same example: $p_A = 0.6, p_B = 0.4$:

$$
H = -(0.6 \log_2 0.6 + 0.4 \log_2 0.4) = -(0.6 \cdot {-0.737} + 0.4 \cdot {-1.322}) \approx 0.971
$$

(High disorder, close to the 1.0 max for binary — class proportions are nearly balanced.)

##### Information Gain

The reduction in entropy you get from making a split on feature $A$:

$$
\text{IG}(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} \cdot H(S_v)
$$

Picking the feature that **maximizes** information gain at each node = the standard greedy tree-building algorithm (ID3, C4.5).

Worked example: dataset of 50 samples (30A, 20B), feature $F$ takes two values:

| Branch | Counts | $p_A$ | $p_B$ | Entropy | Weight |
|---|---|---|---|---|---|
| $F=1$ | 30 (20A, 10B) | 0.667 | 0.333 | 0.918 | 30/50 = 0.6 |
| $F=2$ | 20 (10A, 10B) | 0.5 | 0.5 | 1.000 | 20/50 = 0.4 |

Weighted post-split entropy: $0.6 \cdot 0.918 + 0.4 \cdot 1.000 = 0.951$.

Information Gain: $0.971 - 0.951 = 0.020$ — a small reduction. The algorithm would compare this against IG for every other feature and pick the largest.

##### Gini vs. Entropy — which one?

In practice they agree on most splits. Gini is faster (no log). Entropy is theoretically grounded in information theory. Most libraries (scikit-learn, XGBoost) use Gini by default; switch to entropy if you want the information-theoretic interpretation. The choice rarely changes the resulting tree meaningfully.

#### Stopping criteria — when to stop splitting

Without limits, a tree will keep splitting until every leaf has only one sample → perfectly memorizes the training set → terrible generalization (textbook overfitting). Standard stopping conditions:

| Condition | Hyperparameter (sklearn) | Why |
|---|---|---|
| **Maximum depth** | `max_depth` | Bound the number of consecutive questions. Prevents the tree from growing too complex. |
| **Min samples per leaf / split** | `min_samples_leaf`, `min_samples_split` | Don't make a split when too few samples remain — the resulting decision is unreliable. |
| **Pure node** | (automatic) | All samples in the node belong to one class — no further split possible. |
| **Min impurity decrease** | `min_impurity_decrease` | Stop if the best split doesn't reduce impurity by at least some threshold. |

#### Decision boundary — axis-aligned, not arbitrary

Each split tests one feature against one threshold (`x_3 ≤ 0.7?`). So the resulting decision boundary is a series of **axis-aligned hyperplane pieces** — staircases in 2D, blocks in higher dim:

```
 x₂
  │  □ □  ●            ┌── tree boundary
  │ □ □ │ ● ●          │
  │ □ □ ┼──────────    │
  │ □ □ │● ● ●         ▼
  │     │
  │  ───┴───────       (vertical and horizontal cuts only,
  │  ●  │              no diagonal lines like logistic regression)
  │  ●● │ □ □
  └───────────────→
                   x₁
```

This is fundamentally different from logistic regression's diagonal hyperplane.

#### Assumptions — minimal

This is one of decision trees' biggest selling points:

| Linear regression assumed... | Decision trees... |
|---|---|
| Linearity between features and target | No assumption — handles non-linear relationships natively |
| Normality of residuals | Doesn't care about distributions |
| Sensitive to outliers (squared loss) | Robust — splits are based on order, not distance |
| Comparable feature scales help | Scale-invariant — no normalization needed |

You can throw raw, mixed-scale, non-normal, outlier-laden tabular data at a tree and it'll usually produce something useful with zero preprocessing.

#### Red-team angles — different attack surface

- **Gradient-based attacks (FGSM, DeepFool, JSMA) DON'T directly apply to trees.** Trees are non-differentiable (axis-aligned step functions) — there's no gradient $\nabla_x f$ to follow. Module 09's attacks target neural networks specifically. Trees have their own attack family.
- **Decision-boundary evasion is still possible — but combinatorial, not gradient-based.** An attacker who knows the tree can find the smallest feature change that lands the input in a different leaf by walking the tree backwards. For categorical features, "evade" = flip a few bits.
- **Trees are exceptionally easy to extract.** Each query reveals a path from root to leaf. With $\sim O(\text{tree size})$ targeted queries, an attacker can fully reconstruct the tree's structure and decision rules. Module 07's model reverse engineering — trees are the worst-case-for-defender model class for extraction attacks.
- **Once extracted, trees are 100% interpretable.** An attacker reads the rules directly: "if `byte_entropy > 7.2 AND has_packer = TRUE → MALICIOUS`". They now know exactly which feature thresholds to keep their malware *under*. Compare to a deep net where extracting the model still leaves attackers needing gradient access to find evasions.
- **Module 02 uses Random Forests** (an ensemble of decision trees) for **NSL-KDD network anomaly detection** — same vocabulary applies, plus the ensemble adds noise that makes pure extraction harder but doesn't fundamentally change the attack family.
- **Overfitted trees memorize training points.** A tree without depth limits creates one leaf per training sample → membership inference becomes trivial: query the tree, see if your guess matches a leaf's class with 100% confidence (likely a training point) vs. lower confidence (likely unseen).
- **Tree information gain uses $\log_2$** — same information-theoretic machinery as DP-SGD's privacy-budget accounting (module 11). Different application, same math.

**Takeaways:**
- Trees = nested if-else; root → internal questions → leaf prediction.
- Build by greedy splitting on the feature with highest information gain (or lowest Gini).
- Three impurity measures: **Gini** (fast, default), **Entropy** (info-theoretic), **Information Gain** (entropy reduction).
- Stopping: max depth, min samples per leaf, pure nodes — all hedge against overfitting.
- Decision boundary = axis-aligned staircase, not arbitrary hyperplane.
- Minimal assumptions; works on raw tabular data.
- Attack surface differs from neural nets: no gradient attacks, but trivially extractable + 100% interpretable once extracted.

---

### 7. Naive Bayes

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A probabilistic classifier built on **Bayes' theorem** plus one strong simplifying assumption (features are conditionally independent given the class — the "naive" part). Despite the assumption being almost never strictly true, it works astonishingly well in practice — especially for **text classification** (spam, sentiment, topic).

#### Bayes' theorem — the foundation

$$
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}
$$

| Symbol | Name | Read as |
|---|---|---|
| $P(A \mid B)$ | **Posterior** | "Probability of $A$ given that $B$ happened" |
| $P(B \mid A)$ | **Likelihood** | "Probability of seeing $B$ if $A$ is true" |
| $P(A)$ | **Prior** | "What we believed about $A$ before seeing $B$" |
| $P(B)$ | **Evidence** (marginal probability) | "Total probability of seeing $B$, summed over all causes" |

The shape: posterior = (likelihood × prior) / evidence. Updating beliefs in light of new data.

#### Worked example — disease test (the base-rate fallacy)

Setup:
- Disease prevalence: $P(A) = 0.01$ (1% of population)
- Test sensitivity: $P(B \mid A) = 0.95$ (true positive rate)
- Test false positive rate: $P(B \mid \neg A) = 0.05$

You test positive. **What's the probability you actually have the disease?**

Step 1 — total probability of testing positive (law of total probability):

$$
P(B) = P(B \mid A) P(A) + P(B \mid \neg A) P(\neg A) = (0.95)(0.01) + (0.05)(0.99) = 0.059
$$

Step 2 — apply Bayes:

$$
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)} = \frac{(0.95)(0.01)}{0.059} \approx 0.161
$$

**~16%, not 95%.** The 95%-accurate test only gives you a 16% posterior because the disease is rare. This counterintuitive result is the **base-rate fallacy** — humans (and naive intuitions about ML) systematically over-trust likelihood and ignore the prior. It shows up directly in adversarial contexts: a 95%-accurate spam detector at a 0.1% true-spam base rate produces *mostly false positives* among its "spam" verdicts.

#### Naive Bayes for classification

Given an input $x$ with features $(x_1, x_2, \dots, x_n)$, we want the most-likely class $c$:

$$
P(c \mid x_1, \dots, x_n) = \frac{P(x_1, \dots, x_n \mid c) \, P(c)}{P(x_1, \dots, x_n)}
$$

The denominator is the same for every class, so we can ignore it for ranking. The numerator's joint likelihood $P(x_1, \dots, x_n \mid c)$ is computationally hopeless for many features — until we make the **naive independence assumption**:

$$
P(x_1, \dots, x_n \mid c) \approx \prod_{i=1}^{n} P(x_i \mid c)
$$

i.e. *given the class, every feature is independent of every other feature.* This collapses the joint into a product of per-feature likelihoods, which we can estimate from training data trivially. Final prediction:

$$
\hat{c} = \underset{c}{\arg\max} \; P(c) \prod_{i=1}^{n} P(x_i \mid c)
$$

In practice we sum log-probabilities (avoid underflow from multiplying many small numbers):

$$
\hat{c} = \underset{c}{\arg\max} \; \left[ \log P(c) + \sum_{i=1}^{n} \log P(x_i \mid c) \right]
$$

#### Walkthrough — spam classifier (this is exactly Module 02)

| Step | What happens |
|---|---|
| 1. Compute priors | $P(\text{spam}), P(\text{ham})$ from training data (e.g. 0.2 vs 0.8) |
| 2. Compute likelihoods | For each word $w$: $P(w \mid \text{spam})$ and $P(w \mid \text{ham})$ — count word occurrences in each class, divide by class size |
| 3. Score new email | Multiply prior × per-word likelihoods for each class (assumes word independence — the naive part) |
| 4. Predict | Class with the larger posterior score wins |

The "naive" assumption: that the word "free" appearing in an email is independent of "money" appearing, *given* that we already know the class. Obviously false (those words co-occur in spam), but the classifier still works — because the *ranking* of class scores is often robust even when absolute probabilities are wrong.

#### Three variants — pick by feature type

| Variant | Feature type | Likelihood model | When to use |
|---|---|---|---|
| **Gaussian NB** | Continuous (real-valued) | Each $P(x_i \mid c) \sim \mathcal{N}(\mu_{i,c}, \sigma^2_{i,c})$ — fit a Gaussian per feature per class | Sensor readings, age, income |
| **Multinomial NB** | Counts / discrete (e.g. word frequencies) | Likelihood from word-count proportions in each class | **Text classification, spam (the Module 02 default)** |
| **Bernoulli NB** | Binary (present/absent) | Each feature is a Bernoulli draw with class-specific probability | Document classification with binary "word in doc?" features, presence/absence indicators |

The HTB Module 02 spam exercise uses **Multinomial NB** because email features are word counts.

#### Assumptions

| Assumption | Reality | Impact |
|---|---|---|
| **Conditional independence** of features given class | Almost always violated | Surprisingly tolerable — class *ranking* often survives even when absolute probabilities are off |
| **Distributional fit** (Gaussian / multinomial / Bernoulli) | Approximate | Pick the variant matching your features |
| **Sufficient training data** | Need enough samples to estimate $P(x_i \mid c)$ stably | Otherwise zero counts cause $\log 0 = -\infty$ — fixed with **Laplace smoothing** (add 1 to every count before normalizing) |

#### Red-team angles — this section's attacks are central to the path

- **Module 02's Spam Classification exercise** is a Multinomial Naive Bayes spam filter. You'll build it, then later attack it.
- **Module 08 (AI Evasion Foundations) is built around attacking Naive Bayes spam filters with the GoodWords attack.** The trick: append words with high $P(w \mid \text{ham})$ to a spam email. Each appended "good word" multiplies the ham-class score, eventually flipping the prediction. This is **the** canonical example of evading a probabilistic classifier — and it works *because* of the naive independence assumption (words are scored individually, so one decisive bad word doesn't outweigh many added good ones).
- **The independence assumption IS the attack surface.** A defender who uses bigrams ("free money") instead of unigrams ("free", "money" separately) breaks the independence assumption advantageously and makes GoodWords harder. Attackers respond by inserting noise tokens between bigram parts.
- **Bayes' theorem powers membership inference attacks** (Module 11). The shadow-model attack reframes "was sample $x$ in the training set?" as: $P(\text{member} \mid \text{model output}) = \frac{P(\text{output} \mid \text{member}) P(\text{member})}{P(\text{output})}$. Same structure as the disease-test example — just with "member" instead of "diseased."
- **Priors can be poisoned.** If an attacker can influence training data composition (Module 06), they can shift $P(\text{class})$ — e.g. flooding the training set with spam relabeled as ham reduces $P(\text{spam})$, biasing the prior toward declaring everything ham at inference time.
- **Base-rate fallacy bites detection systems hard.** A high-accuracy classifier deployed against rare events (true malware, true APT activity) generates mostly false positives — same math as the disease example. Attackers exploit this by counting on alert fatigue.
- **Laplace smoothing leaves a fingerprint.** Models trained without smoothing crash on unseen tokens (giving probability 0); models with smoothing assign them a small but nonzero probability. An attacker probing with novel tokens can sometimes infer training set vocabulary.

**Takeaways:**
- Bayes' theorem: posterior = (likelihood × prior) / evidence. Updates beliefs given evidence.
- Base-rate fallacy: a 95% accurate test on a 1% disease gives only ~16% posterior — priors dominate when likelihoods are close.
- Naive Bayes assumes features are conditionally independent given the class — false but useful.
- Final classifier: $\hat{c} = \arg\max_c P(c) \prod_i P(x_i \mid c)$.
- Three variants by feature type: Gaussian (continuous), Multinomial (counts — **Module 02 spam**), Bernoulli (binary).
- The independence assumption is exactly what GoodWords (Module 08) exploits.

---

### 8. Support Vector Machines (SVMs)

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A classifier (and regressor) that finds a decision boundary by **maximizing the margin** — the gap between the boundary and the nearest training point on each side. Conceptually different from everything before: not probabilistic (Naive Bayes), not rule-based (trees), not loss-fitting (logistic). The geometry of *separation* is the whole game.

#### The big idea — margin maximization

Many possible hyperplanes can separate two classes. Which is "best"? SVMs pick the one with **maximum margin** — the widest possible gap between the boundary and the closest training points on either side.

ASCII view in 2D:

```
 x₂
  │ ● ●                margin width
  │  ●                 ←─────────→
  │ ●─────────────────────────────  upper margin: w·x + b = +1
  │   ●     ▲
  │         │ support vectors
  │     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    decision boundary: w·x + b = 0
  │     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  │              ▼
  │              support vectors
  │  □─────────────────────────────  lower margin: w·x + b = −1
  │ □  □
  │  □
  └────────────────────────────────→ x₁
```

The data points sitting *on* the margin lines are called **support vectors**. They alone define the boundary — every other training point could be deleted without changing the model. That's why it's called "support vector" machine.

**Why maximize the margin?** A wider margin tolerates more noise in new data → better generalization. It's also a kind of robustness: an attacker needs a bigger perturbation to push an input across a wider margin. (More on this in red-team angles.)

#### Linear SVM — the math

The hyperplane is parameterized exactly like logistic regression's decision boundary:

$$
\mathbf{w} \cdot \mathbf{x} + b = 0
$$

| Symbol | Meaning |
|---|---|
| $\mathbf{w}$ | Weight vector — perpendicular to the hyperplane |
| $\mathbf{x}$ | Input feature vector |
| $b$ | Bias / intercept — shifts the hyperplane away from origin |

Distance from a point $\mathbf{x}$ to the hyperplane:

$$
\text{dist}(\mathbf{x}) = \frac{\lvert \mathbf{w} \cdot \mathbf{x} + b \rvert}{\lVert \mathbf{w} \rVert}
$$

So if we fix the closest points to satisfy $\lvert \mathbf{w} \cdot \mathbf{x} + b \rvert = 1$ (a normalization choice), the **margin width** between the two parallel margin planes is $\frac{2}{\lVert \mathbf{w} \rVert}$. Maximizing the margin = minimizing $\lVert \mathbf{w} \rVert$.

#### The optimization problem

With class labels $y_i \in \{-1, +1\}$, the **hard-margin** SVM solves:

$$
\begin{aligned}
\text{minimize} \quad & \frac{1}{2} \lVert \mathbf{w} \rVert^2 \\
\text{subject to} \quad & y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \text{for all } i
\end{aligned}
$$

Reading the constraint: for every training point, the score $\mathbf{w} \cdot \mathbf{x}_i + b$ should have the same sign as the label $y_i$ AND be at least 1 in magnitude → the point is on the correct side of the margin, not just the boundary.

The objective $\frac{1}{2} \lVert \mathbf{w} \rVert^2$ is **convex**, so there's a unique global optimum. Solved with quadratic programming.

In practice, real data isn't perfectly separable, so the **soft-margin** SVM adds slack variables $\xi_i \geq 0$ that let some points violate the margin (with a penalty $C$ for each violation). This is what `sklearn.svm.SVC` actually trains.

#### Non-linear SVMs — the kernel trick

What if the data isn't linearly separable in its original space? Lift it into a higher-dimensional space where it *is* separable.

```
Original 2D space (not linearly separable):
                                        Lifted 3D space (separable):
  ●   ○   ●                                       ●
○   ●   ○   ●                                ●        ●
  ●   ○   ●           ──── kernel ────►      ─────────────  ← linear hyperplane here
○   ●   ○   ●                                  ○      ○
  ●   ○   ●                                       ○
                                                
(no straight line works)               (a flat plane separates them)
```

The "trick" is that you don't have to actually compute the high-dimensional coordinates. You only need the **dot product** between mapped points, and a **kernel function** $K(\mathbf{x}_i, \mathbf{x}_j)$ computes that dot product directly in the original space:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)
$$

where $\phi$ is the implicit mapping. Even when $\phi$ maps to *infinite* dimensions, $K$ stays computable.

Common kernels:

| Kernel | Formula | Implicit mapping | Use when |
|---|---|---|---|
| **Linear** | $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j$ | Identity (no lift) | Data is already linearly separable; high-dim sparse text features |
| **Polynomial** | $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + c)^d$ | Polynomial features up to degree $d$ | Mild non-linearity; need interaction terms |
| **RBF (Gaussian)** | $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2)$ | Infinite-dimensional Gaussian basis | Default for non-linear data — most flexible, most popular |
| **Sigmoid** | $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\alpha \mathbf{x}_i \cdot \mathbf{x}_j + c)$ | Sigmoid-shaped boundaries | Rare; behaves like a 1-layer neural net |

**RBF is the workhorse.** $\gamma$ controls how "local" the kernel is — small $\gamma$ → smooth wide boundaries, large $\gamma$ → tight overfit-prone boundaries.

#### Assumptions

| Assumption | What it means |
|---|---|
| **No distributional assumption** | Doesn't require Gaussian features or normal residuals (unlike linear regression) |
| **Handles high dimensionality** | Works well even when $d > N$ (more features than samples); great for text |
| **Robust to outliers** | Margin-based, not loss-based across all points; soft-margin's $C$ controls outlier tolerance |

#### Red-team angles — margin = robustness, support vectors = leaks

- **Linear SVM = same boundary geometry as logistic regression.** $\mathbf{w} \cdot \mathbf{x} + b = 0$ is the same hyperplane. Gradient direction for evasion is just $\mathbf{w}$. **DeepFool (Module 09) handles this in one closed-form step** — same attack as for logistic.
- **The margin IS the adversarial robustness.** To flip a prediction, an attacker must perturb $\mathbf{x}$ enough to cross the boundary — minimum perturbation = $\frac{1}{\lVert \mathbf{w} \rVert}$ for points outside the margin. Wider margin → bigger required perturbation. This is *the* connection between classical SVM theory and modern adversarial defense research ("margin-based certified robustness").
- **Support vectors are training points → membership inference jackpot.** Only the support vectors define the model. If an attacker can identify which inputs are support vectors (e.g. by perturbing inputs and watching which produce decision-boundary changes), they've identified specific training samples. This is a documented MIA against SVMs and a precursor to Module 11's shadow-model attacks.
- **Kernel trick = embedding spaces.** The idea of "lift inputs into a higher-dim space where they're easier to work with" is the conceptual ancestor of every embedding layer in deep learning. When Module 04 talks about LLM embedding spaces and Module 09 attacks gradients in feature space, the mental model is the same.
- **RBF kernels can defeat naive evasion attacks** — the boundary is non-linear and locally curved, so a one-shot linear perturbation (like FGSM) often doesn't transfer cleanly. But iterative attacks (I-FGSM) handle this fine.
- **Soft-margin's $C$ parameter is exploitable.** Large $C$ = small margin = more sensitive to outliers = easier to poison the boundary with a few mislabeled points (Module 06 label-flipping). Small $C$ = wide margin = harder to attack but possibly underfit.
- **SVMs are popular for malware classification** with engineered byte/PE features. Attacks on PE-feature SVMs (Wagner & Soto, 2002 onward) are an entire subfield. Module 02 uses Random Forests instead, but the techniques transfer.

**Takeaways:**
- SVM finds the maximum-margin hyperplane. Boundary defined by **support vectors** — the closest training points.
- Linear SVM: minimize $\frac{1}{2} \lVert \mathbf{w} \rVert^2$ subject to $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$. Margin width = $\frac{2}{\lVert \mathbf{w} \rVert}$.
- Non-linear SVMs use the **kernel trick** to compute high-dim dot products without high-dim coordinates. RBF is the default.
- Margin maximization is the classical analog of modern adversarial-robustness research.
- Support vectors leak training data → MIA-friendly.

---

### 9. Unsupervised Learning Algorithms

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A different paradigm: **no labels.** The algorithm is shown only the inputs $x$, never the answers $y$. Its job is to discover *structure* in the data — which points belong together, which directions matter, which observations are weird.

#### Three problem types

| Type | Goal | Examples | Where in path |
|---|---|---|---|
| **Clustering** | Group similar points (no predefined classes) | Customer segmentation, document topics | §10 K-Means |
| **Dimensionality reduction** | Compress features while preserving info | Image compression, feature engineering for downstream models | §11 PCA |
| **Anomaly detection** | Spot points unlike the rest | Fraud detection, intrusion detection, malware behavior | §12 + **Module 02 network anomaly detection** |

#### Similarity measures — the foundation

Most unsupervised algorithms reduce to "how close are these two points?" Three measures dominate:

##### Euclidean distance (L2)

Straight-line distance — the default for most clustering algorithms.

$$
d_{\text{Euclidean}}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2} = \lVert \mathbf{x} - \mathbf{y} \rVert_2
$$

Same $L_2$ norm from §2. Sensitive to feature scale (a feature ranging 0–1000 will dominate one ranging 0–1 if you don't scale them).

##### Manhattan distance (L1)

Sum of absolute coordinate differences — like walking city blocks instead of cutting diagonally.

$$
d_{\text{Manhattan}}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} \lvert x_i - y_i \rvert = \lVert \mathbf{x} - \mathbf{y} \rVert_1
$$

Same $L_1$ norm from §2. More robust to outliers than Euclidean (no squaring → outliers don't dominate).

##### Cosine similarity

Measures the **angle** between vectors, ignoring their magnitudes.

$$
\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\lVert \mathbf{x} \rVert \lVert \mathbf{y} \rVert}
$$

Range: $-1$ (opposite directions) to $+1$ (same direction). $0$ = orthogonal (unrelated).

**This is THE similarity measure for embedding spaces** — LLM word embeddings, sentence embeddings, image embeddings, RAG document retrieval. When you hear "vector database" or "semantic search," cosine similarity is the underlying metric. Critical for understanding Module 04 / 05 prompt injection vectors that work by manipulating retrieval.

#### Cluster validity — "are these clusters even real?"

Two complementary metrics evaluate cluster quality:

| Metric | What it measures | Want |
|---|---|---|
| **Cohesion** | How tightly packed points are *within* a cluster | High |
| **Separation** | How far apart different clusters are | High |

Combined into single-number scores:

| Score | Range | Interpretation |
|---|---|---|
| **Silhouette score** | $[-1, +1]$ | Per-point: $+1$ = clearly in its cluster, $0$ = on a boundary, $-1$ = likely in wrong cluster. Average over all points = overall quality. |
| **Davies-Bouldin index** | $[0, \infty)$ | Lower = better. Average ratio of within-cluster scatter to between-cluster distance. |

These also help pick **how many clusters** to use (the $k$ in K-Means) — try several $k$ values, pick the one with the best silhouette.

#### Dimensionality and the curse

**Dimensionality** = number of features. **Intrinsic dimensionality** = the effective complexity of the data, which can be much lower than the feature count (e.g. 1000 pixel features encoding a face that lives on a roughly-50-dimensional manifold).

The **curse of dimensionality** — counterintuitive things that happen as $d$ grows:

1. **Distances flatten.** In high dimensions, *all* points become roughly equidistant. The ratio $\frac{\max d - \min d}{\min d} \to 0$ as $d \to \infty$. → Distance-based methods (K-Means, KNN, anomaly detectors using distance thresholds) become noisier.
2. **Volume concentrates at the boundary.** Most of a high-dimensional sphere's volume sits near its surface — counterintuitive but true (and exploitable).
3. **Sparsity.** A fixed number of samples covers an exponentially-shrinking fraction of the input space as $d$ grows.

This is exactly what dimensionality reduction (§11 PCA) is for: project down to the data's intrinsic dimensionality where distances and density estimates become meaningful again.

#### Anomaly vs. outlier — different vocabulary

| Term | Connotation | Implies |
|---|---|---|
| **Outlier** | Statistical deviation — "far from the bulk" | Could be noise, measurement error, or genuine extreme value |
| **Anomaly** | Domain-meaningful deviation — "doesn't fit the model of normal" | Usually implies *something interesting* is going on (fraud, intrusion, malfunction) |

Same data point can be both, neither, or one without the other. In security writing: "outlier" usually means a statistical concept; "anomaly" usually implies a security-relevant event worth investigating.

#### Feature scaling — required for distance-based methods

Distance-based algorithms (every clustering and anomaly method) treat all features as if they're on the same scale. If they aren't, the largest-magnitude feature dominates. Two standard techniques:

##### Min-Max scaling

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

Maps every feature to $[0, 1]$. Simple, but sensitive to outliers (one extreme value compresses everything else into a narrow range).

##### Z-score standardization

$$
x' = \frac{x - \mu}{\sigma}
$$

Centers each feature at 0 with unit variance. More robust to outliers and the standard preprocessing for PCA, K-Means, SVMs.

**You always do one of these before running an unsupervised algorithm on multi-scale features.** Forgetting is the #1 source of "why are my clusters garbage?" results.

#### Red-team angles

- **Module 02's Network Anomaly Detection exercise is unsupervised anomaly detection on NSL-KDD.** The same vocabulary applies: distance-to-normal threshold, curse of dimensionality concerns at 41 features, feature scaling required.
- **Anomaly detectors are the defense — bypassing them is a red-team primitive.** "Living off the land" attacks (using legitimate tools so the behavior stays close to baseline) are exactly anomaly evasion. Adversarial-ML literature has principled versions: craft inputs that minimize the anomaly score while still achieving the malicious goal.
- **Curse of dimensionality is exploitable.** When defenders run anomaly detection on high-dim feature spaces (raw network flows, full process trees), distances become uninformative → noisy detectors → easier to hide. Attackers add noise dimensions; defenders apply PCA first to combat this.
- **Cosine similarity is the metric for LLM/RAG attacks.** When module 04/05 deals with prompt injection via retrieved context, the retrieval is cosine-similarity over embeddings. Crafting an "injectable document" = crafting a vector that lands close-to-query in cosine space while carrying malicious instructions in its text.
- **Feature scaling parameters (min, max, μ, σ) are persistence-relevant.** They're computed once at training time and applied at inference. If an attacker can probe the scaler (e.g. by submitting extreme values and observing rejections), they learn the deployment-time normalization range — useful for crafting adversarial inputs that survive preprocessing.
- **Clustering can leak training data.** Tight clusters with few members reveal that those members were probably training points. K-Means specifically returns *centroids* — averages of training points — which can leak aggregated information about training data composition.
- **Silhouette score on adversarial inputs reveals attacks.** Adversarial examples often have low silhouette scores (they're between clusters, not in any) — defenders sometimes use this as a detection signal. Attackers respond by adding constraints to keep cluster membership clean while still flipping the prediction.

**Takeaways:**
- Three unsupervised problems: clustering, dimensionality reduction, anomaly detection. Module 02 uses #3 (anomaly detection on NSL-KDD).
- Three similarity measures: Euclidean ($L_2$, default), Manhattan ($L_1$, outlier-robust), Cosine (the embedding-space metric — critical for LLM/RAG context).
- Cluster validity: cohesion + separation → silhouette score, Davies-Bouldin index.
- Curse of dimensionality flattens distances → distance-based methods fail in high dim → need dimensionality reduction.
- Anomaly = domain-meaningful deviation; outlier = statistical deviation.
- Always feature-scale (Min-Max or Z-score) before distance-based unsupervised methods.

---

### 10. K-Means Clustering

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

The default clustering algorithm. Partitions $N$ points into $K$ non-overlapping clusters by **iteratively** alternating between assigning points to the nearest centroid and recomputing centroids as cluster means. Minimizes total within-cluster variance.

#### The algorithm — four steps, repeat until convergence

```
1. INITIALIZE   Pick K random data points as initial centroids μ₁, μ₂, …, μ_K

2. ASSIGN       Each point x → cluster k* where  k* = argmin_k ||x − μ_k||²
                (Voronoi partition of the feature space around the centroids)

3. UPDATE       Each centroid μ_k ← mean of points now assigned to cluster k

4. REPEAT 2-3   Until centroids stop moving (or max-iters reached)
```

Each iteration is guaranteed to decrease (or keep equal) the within-cluster sum of squares:

$$
\text{WCSS} = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \lVert \mathbf{x} - \boldsymbol{\mu}_k \rVert^2
$$

The algorithm converges to a *local* minimum of WCSS — not necessarily the global one. Different random initializations can land at different solutions.

##### Visualization across iterations (2D, K=2)

```
 Iter 0 (random init)         Iter 1 (after assign+update)        Iter 2 (converged)
                              
   ●   ●   ★                       ●   ●                              ●   ●
     ●     ●                          ●     ●                            ●     ●
   ●         ●                      ●  ★      ●                       ●  ★      ●
       ●                                ●                                    ●
                                                                       
       ●                                  ●                                  ●
   ●     ★                              ●                                  ●
     ●     ●                       ★      ●                            ●  ★  ●
                                       ●                                    ●

   ★ = centroid
```

Centroids drift toward cluster means until they stabilize.

##### K-Means++ (the modern initializer)

Pure random initialization is fragile — bad starts → bad local minima. **K-Means++** picks initial centroids that are *spread out* (each new centroid chosen with probability proportional to squared distance from the nearest existing centroid). Default in `sklearn.cluster.KMeans`. Almost always strictly better than random init; worth knowing the name.

#### Distance metric — Euclidean by default

K-Means uses **Euclidean distance** ($L_2$) by default — and the math kind of requires it for the centroid-update step to make sense (the mean minimizes sum of squared $L_2$ distances). For other metrics, you'd use related algorithms (K-Medoids for $L_1$, spherical K-Means for cosine).

#### Choosing K — the central problem

K isn't learned; you pick it. Two standard heuristics:

##### Elbow method

Run K-Means for $K = 1, 2, 3, \dots$ and plot WCSS vs. K.

```
WCSS
  ↑
  │ ●
  │  
  │  ●
  │
  │   ●
  │       ← elbow at K=3
  │    ●────●─────●─────●
  │
  └──────────────────────→ K
    1   2   3   4   5   6
```

WCSS always decreases with more clusters (more granular = tighter). The "elbow" — where the rate of decrease sharply slows — heuristically marks "good enough K". Beyond it, you're paying complexity for little gain (and risking overfitting).

##### Silhouette analysis

For each point, compute:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

where $a(i)$ = mean distance from $i$ to other points in its own cluster, $b(i)$ = mean distance from $i$ to points in the *nearest other* cluster. Range:

| $s$ value | Meaning |
|---|---|
| $\approx +1$ | Point clearly belongs to its cluster |
| $\approx 0$ | Point is on a boundary between clusters |
| $\approx -1$ | Point likely in the wrong cluster |

Average silhouette over all points → a single quality score per $K$. Pick the $K$ with the highest average silhouette.

The two methods often disagree mildly. Elbow is a quick visual; silhouette is more rigorous. Combine with **domain knowledge** (e.g. "we have 4 marketing personas, so K=4 makes sense regardless of what the silhouette says").

#### Assumptions

| Assumption | Failure mode |
|---|---|
| **Spherical, similar-sized clusters** | Elongated, irregular, or heavily-imbalanced clusters get badly partitioned (use DBSCAN or Gaussian Mixture Models instead) |
| **Standardized features** | Without scaling (Min-Max or Z-score from §9), the largest-magnitude feature dominates the distance computation |
| **Outlier-free** | A single far-away outlier becomes its own cluster or pulls a centroid wildly off-center |
| **Pre-chosen K** | Wrong K → meaningless or misleading clusters |

#### Red-team angles

- **Centroids leak training data composition.** Each centroid $\boldsymbol{\mu}_k$ is the *mean* of training points assigned to cluster $k$. With a small or homogeneous cluster, the centroid is statistically close to actual training points → indirect membership inference. Module 11's MIA techniques generalize from supervised models, but clustering exposes this directly.
- **Cluster-based anomaly detection is bypassable by "normal-shaping" inputs.** When defenders train K-Means on normal traffic and flag points far from any centroid as anomalous, attackers craft malicious inputs that stay close to a known centroid (low anomaly score) while still completing the malicious action. This is a recurring tactic in adversarial ML for IDS evasion.
- **Initialization sensitivity = inconsistent boundaries across deployments.** Two K-Means models trained on the same data with different random seeds can produce different cluster boundaries. For defenders, this means anomaly detection thresholds aren't reproducible. For attackers, it means an attack tuned against one deployment may not transfer to a re-trained one (good for defenders if they re-train often, bad for attackers).
- **Spherical-cluster assumption is exploitable.** Real attack-traffic distributions are often elongated or curved (e.g. low-and-slow scans). K-Means forces them into spherical clusters, leaving "natural" gaps where attack patterns can hide between centroids.
- **Feature scaling probing.** Same as §9: if the deployment uses Z-score scaling (most do), an attacker submitting extreme-value probes can infer the (μ, σ) parameters → can craft inputs that survive normalization.
- **Cluster IDs as features for downstream classifiers** is a common pipeline. If an attacker can flip an input's cluster assignment, they change a feature for the next model in the chain — chained-model attacks (Module 07's "insecure integrated components" pattern).
- **K is a tuning knob with security implications.** Small K = broad clusters = lots of "normal" → easier to evade. Large K = fine-grained clusters = many tiny normal regions → harder to evade but more false positives. Same precision/recall trade-off appearing yet again.

**Takeaways:**
- 4-step iterative algorithm: init → assign → update → repeat. Minimizes WCSS = $\sum_k \sum_{x \in C_k} \lVert x - \mu_k \rVert^2$.
- Converges to a **local** optimum; K-Means++ initialization mitigates bad starts (sklearn default).
- Picking K: Elbow method (visual) + Silhouette analysis (quantitative) + domain knowledge.
- Assumes spherical, similar-sized, scaled, outlier-free clusters — many failure modes for security data.
- Centroids = means of training points → leak information; cluster-based anomaly detection is bypassable by normal-shaping.

---

### 11. Principal Component Analysis (PCA)

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

The default **dimensionality reduction** technique. Finds a new coordinate system aligned with the **directions of maximum variance** in your data, then lets you keep only the top few axes — discarding the rest as low-information noise. Compresses high-dim data into a faithful low-dim summary.

#### The intuition — rotate the axes to match the data

Imagine 2D data shaped like an elongated cigar. The original $(x_1, x_2)$ axes are arbitrary; PCA finds a rotation where the *first* axis runs along the cigar's length (max variance) and the *second* axis runs perpendicular (smaller variance). Keep only the first axis → you've compressed 2D → 1D while losing very little information.

```
  x₂
  │              ●●●     ← PC1 (largest variance — long axis of the cigar)
  │           ●●●●●●●● ↗
  │        ●●●●●●●●●●●
  │     ●●●●●●●●●●●●
  │  ●●●●●●●●●●●
  │       ↖ PC2 (perpendicular, small variance — width of the cigar)
  │
  └─────────────────────→ x₁

PCA finds these orthogonal directions automatically.
Drop PC2 → 2D data becomes 1D along PC1's axis.
```

In higher dimensions: same idea — find $d$ orthogonal directions ranked by variance, keep the top $k$, project the data onto those.

#### The algorithm — six steps

```
1. STANDARDIZE      For each feature: z = (x - μ) / σ          (Z-score, from §9)

2. COVARIANCE       Compute the d×d covariance matrix C of the standardized data

3. EIGENDECOMPOSE   Solve  C v = λ v   for all eigenvectors v_i and eigenvalues λ_i

4. SORT             Order eigenvectors by descending eigenvalue (largest λ first)

5. SELECT           Keep the top k eigenvectors → matrix V (d × k)

6. PROJECT          Y = X · V    (transforms N×d original data into N×k reduced form)
```

#### The eigenvalue equation in PCA

This is the core formula:

$$
C \mathbf{v} = \lambda \mathbf{v}
$$

| Symbol | Meaning |
|---|---|
| $C$ | $d \times d$ covariance matrix of standardized features |
| $\mathbf{v}$ | An eigenvector — a direction in feature space |
| $\lambda$ | The corresponding eigenvalue — variance captured along $\mathbf{v}$ |

A covariance matrix has $d$ eigenvectors (orthogonal to each other), each with its own eigenvalue. Sort them by $\lambda$ descending → the eigenvector with the largest $\lambda$ is **PC1**, the next is **PC2**, etc.

##### Quick recap from §2 — what eigenvectors mean geometrically

For a transformation matrix $A$, an eigenvector $\mathbf{v}$ is a direction that $A$ only **stretches** (by factor $\lambda$), without rotating:

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

Rubber-band example from HTB: $A = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$, $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$:

$$
A \mathbf{v} = \begin{bmatrix} 2 \\ 0 \end{bmatrix} = 2 \mathbf{v}
$$

The vector $[1, 0]$ is an eigenvector of $A$ with eigenvalue $2$ — $A$ stretches it by 2 along the x-axis without rotating it. In PCA, the same idea: principal components are the directions that the *covariance* matrix preserves up to scaling.

#### Solving the eigenvalue equation in practice

| Method | What it does | When |
|---|---|---|
| **Eigendecomposition** | Direct: factor $C = V \Lambda V^{-1}$ | Conceptually clean; less numerically stable |
| **Singular Value Decomposition (SVD)** | Factor the data matrix directly: $X = U \Sigma V^T$. Columns of $V$ = principal components; diagonal of $\Sigma^2 / (N-1)$ = eigenvalues | What `sklearn.decomposition.PCA` actually uses — more numerically stable, doesn't require forming $C$ explicitly |

SVD is everywhere in modern ML (low-rank attention approximations, model compression, recommender systems). PCA via SVD is the canonical example.

#### Projecting the data — final transformation

Once you've picked the top $k$ eigenvectors as columns of $V$ (a $d \times k$ matrix):

$$
Y = X V
$$

| Symbol | Shape | Meaning |
|---|---|---|
| $X$ | $N \times d$ | Original standardized data |
| $V$ | $d \times k$ | Selected eigenvectors |
| $Y$ | $N \times k$ | Projected data in the lower-dim space |

Now downstream models receive $k$-dimensional inputs instead of $d$-dimensional — faster training, less risk of curse of dimensionality (§9).

#### Choosing the number of components $k$

Plot **cumulative explained variance ratio** vs. $k$:

```
cumulative
 explained
 variance
   ↑
 1.0┤                                   ───────────────────────
    │                            ─ ─ ─ ─
 0.95┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─        ← target threshold
    │                       ───
    │                  ────             "knee" at k ≈ 4
 0.5┤             ───
    │       ────
    │  ────
   0┤─
    └──────────────────────────────────→ k
     1   2   3   4   5   6   7   8   9
```

Standard rule: pick the smallest $k$ such that cumulative $\sum \lambda_i / \sum_{\text{all}} \lambda_i \geq 0.95$ (or 0.99 for tighter preservation). Translates to "keep enough components to explain 95% of the variance, drop the rest."

#### Assumptions

| Assumption | Failure mode |
|---|---|
| **Linearity** | PCA finds *linear* combinations of features. Curved manifolds need kernel PCA, t-SNE, UMAP, or autoencoders. |
| **Correlated features** | If features are uncorrelated, the covariance matrix is diagonal — PCA finds nothing useful. |
| **Standardization required** | Feature scales matter; without Z-score, large-magnitude features dominate the covariance and steal the principal components. |

#### Red-team angles — PCA is **both** a defense and an attack surface

- **Adversarial perturbations love the low-variance directions PCA discards.** Real data lives near a low-rank manifold; adversarial perturbations often need to push *off* this manifold — exactly into the directions PCA throws away. So a defender who applies PCA preprocessing thinks they're stripping noise, but they may also be stripping the attacker's signal — making certain attacks harder. This is the basis of **PCA-based adversarial defenses** (defensive distillation, randomized smoothing precursors).
- **Counter-attack: adversaries craft perturbations that lie *within* the high-variance subspace** — i.e., perturbations that survive PCA projection. These on-manifold attacks are central to modern adversarial-ML research and are much harder to defend against than off-manifold attacks.
- **Principal components leak training-data structure.** Each PC is a linear combination of features fitted to the training set's covariance. With enough query access, an attacker can reconstruct the principal components → infer which features the model considers most variable → learn what the training data looked like in aggregate. Module 07's model reverse engineering exploits this.
- **Eigenfaces and similar PCA-based facial recognition systems** are explicitly attackable: knowing the eigenfaces (which are public for many academic models like Yale Face Database) lets an attacker craft adversarial faces that project to a target identity's coordinates in the eigenspace.
- **PCA-based anomaly detection** projects new data into PC space and flags points with high reconstruction error (i.e., they don't fit the principal components well). Bypass: craft adversarial inputs whose components in the kept PCs match normal-data statistics → low reconstruction error → low anomaly score, while the malicious payload hides in feature combinations that don't show up in the discarded PCs.
- **Embedding-space PCA in LLMs/RAG.** When systems compress LLM token/sentence embeddings via PCA before storage (a common cost-saving), attackers can craft prompts whose post-PCA representation collides with target documents → retrieval-injection attacks. Modules 04/05 territory.
- **The covariance matrix itself is sensitive training data.** In federated/distributed training, sharing $C$ between parties leaks aggregated information about each party's data. There's a whole subfield of "privacy-preserving PCA" that adds noise to $C$ before computing eigenvectors.
- **SVD = the math behind weight extraction.** Module 07's model reverse engineering against linear/low-rank models often reduces to SVD on observed input/output pairs.

**Takeaways:**
- PCA = find directions of max variance via eigendecomposition (or SVD) of the covariance matrix; keep top $k$.
- Core equation: $C \mathbf{v} = \lambda \mathbf{v}$. Eigenvectors = principal components, eigenvalues = variance captured.
- 6-step recipe: standardize → covariance → eigendecompose → sort → select → project.
- Pick $k$ by cumulative explained variance ratio (typically ≥ 95%).
- Assumes linearity, correlated features, standardized scales.
- Defense AND attack surface — adversarial perturbations exploit the low-variance directions PCA discards (off-manifold attacks); on-manifold attacks survive PCA preprocessing.

---

### 12. Anomaly Detection

**Status:** - [ ]  |  **Type:** Theory

---

### 13. Reinforcement Learning Algorithms

**Status:** - [ ]  |  **Type:** Theory

---

### 14. Q-Learning

**Status:** - [ ]  |  **Type:** Theory

---

### 15. SARSA (State-Action-Reward-State-Action)

**Status:** - [ ]  |  **Type:** Theory

---

### 16. Introduction to Deep Learning

**Status:** - [ ]  |  **Type:** Theory

---

### 17. Perceptrons

**Status:** - [ ]  |  **Type:** Theory

---

### 18. Neural Networks

**Status:** - [ ]  |  **Type:** Theory

---

### 19. Convolutional Neural Networks

**Status:** - [ ]  |  **Type:** Theory

---

### 20. Recurrent Neural Networks

**Status:** - [ ]  |  **Type:** Theory

---

### 21. Introduction to Generative AI

**Status:** - [ ]  |  **Type:** Theory

---

### 22. Large Language Models

**Status:** - [ ]  |  **Type:** Theory

---

### 23. Diffusion Models

**Status:** - [ ]  |  **Type:** Theory

---

### 24. Skills Assessment

**Status:** - [ ]  |  **Type:** Interactive

See [[Skills-Assessment]].

---

## Skills Assessment

See [[Skills-Assessment]].

## References

- HTB module URL: <paste when starting next session>
- Related vault notes: [[00-Meta/HTB AI Red Team Path]]
