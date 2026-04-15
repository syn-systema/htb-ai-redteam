---
module_number: 01
module_name: "Fundamentals of AI"
status: in-progress
difficulty: "Medium"
tier: ""
estimated_time: ""
sections_total: 24
sections_done: 4
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

**Status:** - [ ]  |  **Type:** Theory

---

### 6. Decision Trees

**Status:** - [ ]  |  **Type:** Theory

---

### 7. Naive Bayes

**Status:** - [ ]  |  **Type:** Theory

---

### 8. Support Vector Machines (SVMs)

**Status:** - [ ]  |  **Type:** Theory

---

### 9. Unsupervised Learning Algorithms

**Status:** - [ ]  |  **Type:** Theory

---

### 10. K-Means Clustering

**Status:** - [ ]  |  **Type:** Theory

---

### 11. Principal Component Analysis (PCA)

**Status:** - [ ]  |  **Type:** Theory

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
