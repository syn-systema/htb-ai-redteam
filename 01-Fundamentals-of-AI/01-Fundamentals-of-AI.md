---
module_number: 01
module_name: "Fundamentals of AI"
status: in-progress
difficulty: "Medium"
tier: ""
estimated_time: ""
sections_total: 24
sections_done: 21
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

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

Identifies data points that deviate significantly from "normal" patterns. The defensive ML primitive: every IDS, fraud detector, malware behavior monitor, and SOC alert pipeline is doing some form of anomaly detection. **For red teamers, this is the section about everything you're trying to stay under the threshold of.**

#### Three types of anomalies — three attack strategies

| Type | Definition | Security example | Attacker counter |
|---|---|---|---|
| **Point** | Single point looks weird in isolation | Sudden 10× spike in outbound bandwidth from one host | Throttle the action to look normal point-by-point |
| **Contextual** | Looks weird *given context* but not absolutely | A login at 3 a.m. for a 9-to-5 employee | Time the action to a legitimate context window |
| **Collective** | A group of points is weird together, individuals look fine | 50 different IPs each making 1 login attempt within a minute (low-and-slow brute force) | Distribute the attack across many actors so no individual is flagged |

The three attack strategies on the right side ARE the canonical anomaly-evasion playbook. Real attackers combine them.

#### Three approaches — three defense families

| Approach | How it defines "normal" | Example methods |
|---|---|---|
| **Statistical** | Assumes data follows a known distribution (often Gaussian); anomalies = deviations from it | Z-score, modified Z-score, boxplot/IQR, 3-sigma rule |
| **Clustering-based** | Normal points cluster together; anomalies are far from clusters or in tiny ones | K-Means + distance-to-centroid threshold, DBSCAN noise points |
| **ML-based** | Learn a model of "normal" data, score new points against it | One-Class SVM, Isolation Forest, Local Outlier Factor (LOF), autoencoders |

The three ML-based methods deserve their own treatment.

#### One-Class SVM — "draw a fence around normal"

Recall §8: SVMs find a maximum-margin hyperplane between two classes. **One-Class SVM** trains on *only the normal class* and finds a hyperplane (or curved boundary, with a kernel) that *encloses* the normal data tightly — anything outside is an anomaly.

```
 x₂                       ╭──────────╮
  │     normal data ●●●●  │  ●●●●●  │   ← boundary learned by One-Class SVM
  │                  ●●●● │ ●●●●●●  │       (with RBF kernel: a curved enclosure)
  │                       │  ●●●    │
  │                       ╰──────────╯
  │
  │   ✕ anomaly (outside)
  │
  └────────────────────────────────────→ x₁
```

Same kernel trick from §8 (linear, polynomial, RBF, sigmoid) applies. RBF is again the default for non-linear normal regions.

**Bypass:** craft inputs that fall *inside* the boundary while still carrying the malicious payload. The decision boundary is the attack target — same geometry as supervised SVM evasion, just with different training labels.

#### Isolation Forest — "anomalies isolate quickly"

A clever inversion of the usual "model normal, score anomalies" approach. Instead of modeling normal, **isolate** every point by recursive random splitting and observe how *deep* in the tree each point ends up.

The intuition: anomalies are *few* and *different* → they get isolated faster (shorter path lengths in the random tree). Normal points are *many* and *similar* → harder to isolate (longer paths).

##### The algorithm

1. Build many isolation trees, each from random subsamples.
2. At each node: pick a random feature, pick a random split value within that feature's range, partition the data.
3. Recurse until each point is alone in its leaf.
4. Compute the average path length $E(h(x))$ across all trees.
5. Compute the anomaly score:

$$
s(x) = 2^{-\frac{E(h(x))}{c(n)}}
$$

| Symbol | Meaning |
|---|---|
| $E(h(x))$ | Average path length to isolate $x$ across all trees |
| $c(n)$ | Average path length of unsuccessful BST search with $n$ samples — normalization factor |
| $n$ | Sample size |

| Score | Interpretation |
|---|---|
| Close to **1** | Likely anomaly (short path → easy to isolate) |
| Close to **0.5** | Probably normal |
| Close to **0** | Definitely normal (long path → blends with the crowd) |

**Why it works for security data:** linear in time, scales to high dimensions, doesn't need distance metrics → bypasses the curse of dimensionality (§9). Sklearn's `IsolationForest` is a common production choice.

**Bypass:** make your malicious sample *not stand out on any single feature* — if every feature is within the bulk of normal values, no random split can isolate you quickly. This is why "blend in" tactics work.

#### Local Outlier Factor (LOF) — "anomalies are in low-density neighborhoods"

A density-based approach. For each point $p$:
1. Find its $k$ nearest neighbors.
2. Compute its **local reachability density** — roughly "how dense is the neighborhood around $p$?".
3. Compare $p$'s density to the average density of its neighbors.

If $p$'s density is *much lower* than its neighbors' (i.e., $p$ is sitting in a sparse region while its neighbors are in a dense region), it's an outlier.

##### The math

**Local reachability density (LRD)** of point $p$:

$$
\text{lrd}(p) = \frac{1}{\frac{1}{k}\sum_{o \in N_k(p)} \text{reach\_dist}(p, o)}
$$

where $\text{reach\_dist}(p, o) = \max(\text{k-distance}(o), \text{dist}(p, o))$. (The "reachability distance" smooths things — points in dense regions can't be reached arbitrarily close to one another.)

**LOF score** for $p$:

$$
\text{LOF}(p) = \frac{\frac{1}{k}\sum_{o \in N_k(p)} \text{lrd}(o)}{\text{lrd}(p)}
$$

| LOF value | Interpretation |
|---|---|
| ≈ 1 | Point has same density as neighbors → normal |
| > 1 | Point is in a *sparser* region than its neighbors → outlier |
| ≫ 1 (e.g. > 1.5–2) | Strong outlier |

**Why use LOF over Isolation Forest or One-Class SVM?** LOF handles datasets with **varying density** — clusters of different tightness are common in real data, and LOF doesn't assume a single global density.

**Bypass:** insert your malicious sample into a *dense* part of feature space — surround it with normal-looking points (real or synthetic) so its local density matches its neighbors. This is sometimes called a **mimicry attack**.

#### Assumptions — how each method can fail

| Assumption | Method | Failure mode |
|---|---|---|
| Normal follows a known distribution (e.g. Gaussian) | Statistical (z-score) | Real data has heavy tails, multi-modality → many false positives or false negatives |
| Normal data clusters tightly | Clustering-based | Normal data with multiple sub-populations → some "normal" sub-clusters get flagged |
| Anomalies are rare and few | Isolation Forest, all of them | If anomalies are a sizable fraction (say > 20%), they form their own dense regions and stop looking anomalous |
| Local density matters | LOF | Slow on large datasets; needs careful $k$ selection |

#### Red-team angles — this is the playbook

- **Module 02's NSL-KDD Network Anomaly Detection** uses a **Random Forest** (supervised, since NSL-KDD is labeled) but the underlying problem framing is identical: classify "normal" vs "attack" traffic. The anomaly-detection vocabulary here transfers directly.
- **The three anomaly types map directly to attack tradecraft:**
  - **Point evasion** = throttling + packet-size obfuscation. Don't make any single observation look weird.
  - **Contextual evasion** = "live off the land" + timing. Use legitimate tools, act during business hours, blend with baseline.
  - **Collective evasion** = botnet distribution + low-and-slow. Distribute the campaign so no single source crosses any threshold.
- **Each detection algorithm has a specific bypass:**

| Detector | Bypass technique |
|---|---|
| Z-score / statistical | Keep all feature values within ±3σ of the per-feature mean |
| One-Class SVM | Stay inside the learned boundary — same geometry as supervised SVM evasion |
| Isolation Forest | Don't stand out on any *single* feature; ensure every feature's value is in the bulk of normal range |
| LOF | Mimicry — surround your malicious activity with synthetic normal-looking activity to raise local density |
| Clustering-based (K-Means + threshold) | Stay close to the nearest centroid (the "normal-shaping" attack from §10) |

- **Adversarial ML against anomaly detectors is a documented subfield.** Papers like "Adversarial Examples Against Anomaly Detection" (and many follow-ups) formalize the bypasses above as constrained optimization problems: minimize anomaly score subject to the malicious objective being achieved.
- **Poisoning the "normal" training set** (Module 06's data attacks) is the meta-bypass. If an attacker can inject malicious-looking traffic into the *training* data labeled as normal, the detector will never flag that pattern at inference time. This is why anomaly detectors trained on production data are fundamentally fragile.
- **Detector ensembles raise the bar.** Real defenders stack One-Class SVM + Isolation Forest + LOF + statistical thresholds and require multiple to fire. Attacker must bypass *all* simultaneously → harder optimization problem (though not impossible — multi-detector evasion has its own literature).
- **Concept drift = a free attack vector.** Normal traffic distributions change over time (new services, new users, holiday patterns); detectors retrained on drifted data slowly become less sensitive. Attackers exploit drift windows to introduce new techniques.
- **Path length in Isolation Forest is information-leaky.** Querying the model and observing scores reveals which features the model considers "isolating" → attackers can probe to learn the feature distribution.

**Takeaways:**
- Three anomaly types: Point (single-point), Contextual (situational), Collective (group). Each maps to a specific attack strategy.
- Three detection approaches: Statistical, Clustering-based, ML-based.
- Three ML-based methods to know: **One-Class SVM** (draw a fence around normal), **Isolation Forest** (anomalies isolate quickly, score = $2^{-E(h(x))/c(n)}$), **LOF** (anomalies have lower density than neighbors).
- Each detector has a specific bypass — internalize the table above; this is the core anomaly-evasion playbook.
- Module 02 builds a supervised classifier on NSL-KDD, but the framing is identical.
- Poisoning the training "normal" set (Module 06) is the meta-bypass; ensemble detection raises the bar.

---

### 13. Reinforcement Learning Algorithms

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A completely different paradigm from supervised and unsupervised learning. There is no labeled dataset. There is no fixed dataset at all. Instead: an **agent** interacts with an **environment**, takes actions, observes the consequences, and receives **rewards** that signal whether the actions were good or bad. Over time, the agent learns a **policy** — a strategy for choosing actions — that maximizes long-run cumulative reward.

#### The interaction loop — the whole paradigm in one picture

```
            ┌─────────────────┐
            │                 │
            │      AGENT      │
            │   (policy π)    │
            │                 │
            └──┬──────────────┘
               │            ▲
       action  │            │  next state s_{t+1}
       a_t     │            │  reward     r_{t+1}
               ▼            │
            ┌─────────────────┐
            │                 │
            │   ENVIRONMENT   │
            │                 │
            └─────────────────┘

At each timestep t:
  1. Agent observes state s_t
  2. Agent picks action a_t ~ π(s_t)
  3. Environment transitions to s_{t+1}, returns reward r_{t+1}
  4. Repeat
```

This loop runs for many timesteps (an *episode*, or forever in continuous tasks). The agent's job is to learn a policy $\pi$ that makes good action choices.

#### Core vocabulary — memorize these

| Term | Symbol | Meaning |
|---|---|---|
| **Agent** | — | The learner and decision-maker (the "thing that does stuff") |
| **Environment** | — | Everything outside the agent that responds to actions |
| **State** | $s \in S$ | The current situation the agent observes |
| **Action** | $a \in A$ | A choice the agent makes |
| **Reward** | $r \in \mathbb{R}$ | Scalar feedback after each action — positive = good, negative = punishment |
| **Policy** | $\pi(a \mid s)$ | The agent's strategy: probability of picking action $a$ in state $s$ |
| **Value function** | $V(s)$ or $Q(s, a)$ | Expected long-run reward from a state (or state-action pair) |
| **Discount factor** | $\gamma \in [0, 1]$ | How much to weight future rewards vs. immediate ones |
| **Episode** | — | One run from start to terminal state (e.g. one game of chess) |

#### Two flavors of policy

| Type | Description |
|---|---|
| **Deterministic** | $\pi(s) = a$ — same state always produces the same action |
| **Stochastic** | $\pi(a \mid s) = $ probability distribution over actions — same state can produce different actions |

#### Two flavors of RL

| Approach | Idea | Analogy |
|---|---|---|
| **Model-based** | Agent learns (or is given) a model of the environment — predicts next state + reward for each action — then plans | Having a map of the maze before navigating |
| **Model-free** | Agent learns a policy or value function directly from experience, without modeling environment dynamics | Navigating the maze blind, only learning from reward signals |

Q-Learning (§14) and SARSA (§15) are both **model-free**. Most modern RL applications are model-free; model-based methods are common in robotics where simulators provide a model.

#### The math — cumulative reward and value

The agent's goal is to maximize **cumulative discounted reward** (also called the *return*) starting from time $t$:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

The discount factor $\gamma$ controls the time horizon:

| $\gamma$ | Behavior |
|---|---|
| $0$ | Myopic — only immediate reward matters |
| $0.9$ – $0.99$ | Standard — values future rewards but with diminishing weight |
| $1$ | All future rewards weighted equally — only safe for finite-horizon (episodic) tasks |

##### State-value function $V^\pi(s)$

Expected return from being in state $s$ and following policy $\pi$ thereafter:

$$
V^\pi(s) = \mathbb{E}_\pi \!\left[ G_t \mid S_t = s \right]
$$

"How good is it to be in state $s$, assuming I'll keep using policy $\pi$?"

##### Action-value function $Q^\pi(s, a)$

Expected return from taking action $a$ in state $s$, then following policy $\pi$ thereafter:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \!\left[ G_t \mid S_t = s, A_t = a \right]
$$

"How good is it to take action $a$ in state $s$, then keep using $\pi$?"

The **Q-function** is the central object in Q-Learning (§14) and SARSA (§15). The optimal policy can always be derived from $Q^*$:

$$
\pi^*(s) = \arg\max_a Q^*(s, a)
$$

Just always pick the action with the highest Q-value.

#### Episodic vs. continuous tasks

| Type | Description | Examples |
|---|---|---|
| **Episodic** | Has a clear terminal state; the world resets between episodes | Game of chess, maze navigation, single trading day |
| **Continuous** | Runs indefinitely; no natural reset | Robot arm control, ongoing recommendation system, chatbot in a long session |

For continuous tasks, $\gamma < 1$ is mathematically required (otherwise infinite-sum returns are undefined).

#### Red-team angles — RL is becoming central to AI security

- **RLHF (Reinforcement Learning from Human Feedback) is how every modern aligned LLM is trained.** GPT, Claude, Gemini, Llama-Instruct — all trained to be "helpful and harmless" via RL on human preference data. The reward function is a learned model of what humans rate as a good answer. **Modules 04–05 (Prompt Injection, LLM Output Attacks) are fundamentally attacks against RLHF-trained policies** — finding inputs where the trained policy fails to follow its alignment.
- **Reward hacking / specification gaming.** When the reward signal doesn't perfectly capture the intended goal, agents find shortcuts. Famous examples: a boat-racing RL agent that learned to circle in place collecting power-ups instead of finishing the race; LLMs that learned to "appear helpful" on the rated metric without actually being helpful. Jailbreaks that say "respond as my deceased grandmother who used to read me napalm recipes" exploit this — the reward model didn't penalize the role-play frame.
- **Adversarial perturbations to state observations.** For RL agents using vision/sensors, a tiny perturbation to the input image can flip the chosen action — same FGSM/DeepFool machinery from modules 09–10, applied to the RL state input. Self-driving car evasion via adversarial stop-sign stickers is the canonical example.
- **Policy poisoning via training-experience injection.** If an attacker can inject training experiences (state, action, reward) into the agent's replay buffer, they can shape the learned policy. Practical for distributed/federated RL setups.
- **Reward poisoning of RLHF.** If attackers can submit fake human feedback (e.g. by spinning up sock puppets that consistently rate certain harmful outputs as "good"), they steer the reward model — and the policy follows. Defended via Sybil resistance + outlier detection on raters.
- **Model-based RL leaks the world model.** If an attacker can extract the agent's learned environment model, they know exactly what the agent expects and can craft state transitions the agent didn't anticipate.
- **The discount factor is exploitable.** Aggressive discounting (small $\gamma$) makes agents myopic — exploitable by attacks that delay their payoff (the agent ignores long-term consequences). Large $\gamma$ makes agents value-of-information-driven — exploitable via "reward over the horizon" spoofing.
- **Multi-agent RL** (e.g. autonomous trading, multi-agent simulation) introduces game-theoretic attacks: one adversarial agent can manipulate others' learning by playing strategically against their value estimates.

**Takeaways:**
- RL = agent + environment + states + actions + rewards + policy + value functions.
- Goal: maximize cumulative discounted reward $G_t = \sum \gamma^k r_{t+k+1}$.
- Two value functions: $V^\pi(s)$ (state-value), $Q^\pi(s, a)$ (action-value, central to Q-Learning).
- Optimal policy from optimal Q: $\pi^*(s) = \arg\max_a Q^*(s, a)$.
- Two paradigms: model-based (learn dynamics + plan) vs model-free (learn policy/value directly from experience).
- **RLHF** is THE training method for modern aligned LLMs — Modules 04–05 are attacks against RLHF-trained policies.
- Reward hacking, adversarial state perturbations, and reward poisoning are the three major attack families against RL systems.

---

### 14. Q-Learning

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A **model-free** RL algorithm that learns the optimal **Q-function** $Q^*(s, a)$ — the expected long-run reward of taking action $a$ in state $s$ then acting optimally forever after. Once $Q^*$ is learned, the optimal policy is just $\pi^*(s) = \arg\max_a Q^*(s, a)$.

"Model-free" means the agent never builds an explicit model of how the environment works — it just learns Q-values from experience.

#### The Q-table — store one Q-value per (state, action) pair

For small discrete state and action spaces, the Q-function is literally a lookup table:

| State | Up | Down | Left | Right |
|---|---|---|---|---|
| $S_1$ | -1.0 | 0.0 | -0.5 | 0.2 |
| $S_2$ | 0.0 | 1.0 | 0.0 | -0.3 |
| $S_3$ | 0.5 | -0.5 | 1.0 | 0.0 |
| $S_4$ | -0.2 | 0.0 | -0.3 | 1.0 |

Each cell = "expected long-run reward if I take this action from this state." After training, picking the row's max-value column is the greedy policy.

**The fundamental scalability problem:** the table grows as $|S| \times |A|$. For chess (~$10^{40}$ states), Go ($10^{170}$), or any high-dim sensor input (millions of pixels), a literal table is impossible. Solution: **Deep Q-Networks (DQN)** — replace the table with a neural net that maps $(s) \to Q(s, \cdot)$ for all actions. Same update rule, scaled up.

#### The Q-Learning update rule (the Bellman equation)

The heart of the algorithm. After observing transition $s \xrightarrow{a} s'$ with reward $r$:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

| Symbol | Meaning |
|---|---|
| $Q(s, a)$ | Current estimate (gets updated) |
| $\alpha$ | Learning rate ($0 < \alpha \leq 1$) — how much to trust the new evidence |
| $r$ | Immediate reward observed |
| $\gamma$ | Discount factor from §13 |
| $\max_{a'} Q(s', a')$ | The best Q-value reachable from the next state — the "what could I do next?" lookahead |
| $r + \gamma \max_{a'} Q(s', a')$ | The **TD target** — the new estimate of what $Q(s, a)$ "should" be |
| $[\text{TD target} - Q(s, a)]$ | The **TD error** — surprise: how wrong was our old estimate? |

Reading: *"nudge $Q(s, a)$ toward the new better estimate by a fraction $\alpha$ of the surprise."*

##### Worked example — the grid robot

Setup:
- Robot in $S_1$, takes action `Right`, lands in $S_2$, receives reward $r = 0.5$.
- $\alpha = 0.1$, $\gamma = 0.9$.
- Current $Q(S_1, \text{Right}) = 0.2$.
- Best Q-value from $S_2$: $\max_{a'} Q(S_2, a') = \max(0.0, 1.0, 0.0, -0.3) = 1.0$.

Plug in:

$$
Q(S_1, \text{Right}) \leftarrow 0.2 + 0.1 \cdot [0.5 + 0.9 \cdot 1.0 - 0.2] = 0.2 + 0.1 \cdot 1.2 = 0.32
$$

After this single update, $Q(S_1, \text{Right})$ moved from $0.2$ to $0.32$ — incorporating the actual reward seen + the value of where the agent landed.

#### The algorithm — six steps, repeated

```
1. INITIALIZE     Q(s, a) = 0  for all (s, a)        (or random small values)

2. CHOOSE ACTION  In state s, pick a using exploration strategy
                  (e.g. ε-greedy — see below)

3. TAKE & OBSERVE Execute a → observe r and next state s'

4. UPDATE Q       Q(s, a) ← Q(s, a) + α[r + γ·max Q(s', a') − Q(s, a)]

5. UPDATE STATE   s ← s'

6. REPEAT 2-5     Until Q-values converge or max episodes reached
```

#### Exploration vs. exploitation — and why both matter

The agent faces a fundamental dilemma every step:

| Choice | Risk |
|---|---|
| **Exploit** — pick the current best-known action | Miss out on better actions you've never tried |
| **Explore** — try a random/different action | Waste time on actions that turn out to be bad |

If the agent only exploits, it gets stuck on the first decent strategy it finds (a local optimum). If it only explores, it never converges. You need *both*, balanced over time.

##### ε-greedy strategy — the standard solution

With probability $\varepsilon$, take a **random** action. With probability $1 - \varepsilon$, take the **greedy** (current best) action:

$$
a = \begin{cases}
\text{random action} & \text{with probability } \varepsilon \\
\arg\max_{a'} Q(s, a') & \text{with probability } 1 - \varepsilon
\end{cases}
$$

| $\varepsilon$ | Behavior |
|---|---|
| 1.0 | Pure exploration (uniformly random) |
| 0.9 | Almost always explore — early training |
| 0.1 | Mostly exploit, occasional exploration — mid training |
| 0.0 | Pure greedy exploitation — final policy at deployment |

**Practical pattern:** start with high $\varepsilon$ (~0.9), **decay** it over training to a small value (~0.05). This is "explore aggressively when you know nothing, exploit confidently once you know enough." Other strategies exist (Boltzmann / softmax exploration, UCB) but ε-greedy is the default.

#### Assumptions

| Assumption | Meaning | Failure mode |
|---|---|---|
| **Markov property** | Next state and reward depend ONLY on current $(s, a)$ — not on history | If history matters (e.g. in a partially observable environment), Q-values become inconsistent → use POMDP methods or recurrent policies |
| **Stationary environment** | Transition probabilities and reward function don't change over time | Concept drift breaks learned Q-values; agent must re-train |

#### Off-policy vs on-policy — the key distinction (preview of §15)

Q-Learning is **off-policy**: the update uses $\max_{a'} Q(s', a')$ — the best possible action from $s'$ — regardless of what action the agent will actually take next. The agent learns about the optimal policy while following an exploratory one.

This sets up the contrast with **SARSA** (next section), which uses the action the agent *actually takes* in $s'$ → on-policy.

#### Red-team angles

- **Tabular Q-learning is mostly extinct in production; Deep Q-Networks (DQN) replace the table with a neural net.** This means **every neural-network attack from modules 06–10 applies to RL agents** — adversarial state perturbations (FGSM-style), training-data poisoning, model extraction. Stop-sign attacks against self-driving RL agents are exactly DQN evasion.
- **Q-table extraction is trivial.** Query the agent in many states (or observe its actions in many states), record the action it picked → infer the argmax of Q for each. With enough queries, you reconstruct the policy. For DQN, similar but use model extraction attacks to recover network weights.
- **The Markov property assumption is exploitable by introducing history dependence.** If you can craft scenarios where the "right" action depends on past states the agent has forgotten, the Q-values will be systematically wrong → predictable failure modes.
- **The stationary-environment assumption breaks in adversarial settings by design.** Any active adversary modifying the environment IS non-stationarity. Defender has to detect drift and trigger re-training; attacker exploits the lag between drift and re-train.
- **Reward hacking is the central RL attack family.** The agent maximizes whatever the reward function says — not the designer's intent. Examples scale from "boat circles to collect power-ups instead of finishing race" to "LLM produces convincing-looking but factually wrong answers because RLHF reward model rewarded confident tone." Module 04/05 jailbreaks often exploit reward-hacking gaps in RLHF training.
- **ε-greedy is exploitable both ways.** A defender deploying with $\varepsilon > 0$ is stochastic → harder to predict, but each step has ε chance of taking a random possibly-bad action (an attacker can wait for one). A defender at $\varepsilon = 0$ is deterministic → predictable from observed behavior.
- **Reward poisoning during training corrupts the policy.** If an attacker can inject experiences with crafted $(s, a, r, s')$ tuples into the agent's replay buffer (real-world systems with experience replay), they can shape Q-values arbitrarily. Practical for federated / distributed RL.
- **Specifically for RLHF:** the reward model is itself a neural network trained on human preference data. If you can poison the preference data (Sybil attacks with fake preference labels) or extract the reward model (Module 07 territory), you can craft inputs that exploit gaps in the reward function the policy was trained against.

**Takeaways:**
- Q-Learning learns $Q(s, a)$ — expected long-run reward from $(s, a)$ — via the Bellman update: $Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$.
- Optimal policy: $\pi^*(s) = \arg\max_a Q^*(s, a)$.
- Tabular Q-table doesn't scale; **Deep Q-Networks (DQN)** are the production version → all NN attacks apply.
- ε-greedy balances exploration vs exploitation; standard pattern is decaying $\varepsilon$ over training.
- Off-policy: learns about the optimal policy regardless of the action actually chosen (contrasts with SARSA next).
- Assumes Markov property + stationary environment — both exploitable in adversarial settings.

---

### 15. SARSA (State-Action-Reward-State-Action)

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A model-free RL algorithm that's almost identical to Q-Learning — same Q-table, same Bellman-style update — except for **one small change** that fundamentally alters the agent's behavior. Where Q-Learning is **off-policy** and aggressive, SARSA is **on-policy** and conservative.

The name comes from the 5-tuple it operates on: **S**tate, **A**ction, **R**eward, next **S**tate, next **A**ction.

#### The single-character difference — the update rule

Compare side-by-side:

| Algorithm | Update rule |
|---|---|
| **Q-Learning** | $Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \, \mathbf{\max_{a'} Q(s', a')} - Q(s, a)]$ |
| **SARSA** | $Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \, \mathbf{Q(s', a')} - Q(s, a)]$ |

Q-Learning uses $\max_{a'} Q(s', a')$ — the *best possible* action from $s'$, regardless of what the agent will actually do.
SARSA uses $Q(s', a')$ — the value of the action the agent **actually picks** in $s'$ using its current (potentially exploratory) policy.

That single change has big consequences:
- Q-Learning learns "what's the value of acting optimally from now on" — assumes future-perfect behavior.
- SARSA learns "what's the value of acting *the way I'm currently acting*" — including exploration mistakes.

#### The algorithm — six steps (note the extra a' choice)

```
1. INITIALIZE      Q(s, a) = 0  for all (s, a)

2. CHOOSE a        In state s, pick a using ε-greedy

3. TAKE & OBSERVE  Execute a → observe r, next state s'

4. CHOOSE a'       In s', pick a' using the SAME ε-greedy policy   ← the SARSA twist
                   (Q-Learning skipped this step — it just used max)

5. UPDATE Q        Q(s, a) ← Q(s, a) + α[r + γ·Q(s', a') − Q(s, a)]

6. UPDATE STATE    s ← s',  a ← a'    (carry the chosen next-action forward)

7. REPEAT 2-6      Until convergence
```

Note step 4: SARSA *commits to the next action before updating*. Q-Learning didn't need to — it used the hypothetical optimal action.

#### On-policy vs off-policy — the core distinction

| | Off-policy (Q-Learning) | On-policy (SARSA) |
|---|---|---|
| **Policy being learned** | The optimal policy $\pi^*$ | The current behavior policy (including exploration) |
| **Update target** | $\max_{a'} Q(s', a')$ — assumes future-greedy | $Q(s', a')$ for the action actually chosen |
| **Behavior during learning** | Aggressive — happy to explore risky states because it learns about the optimum | Cautious — internalizes the cost of its own exploration |
| **Convergence** | To $Q^*$ regardless of behavior policy (as long as all (s,a) visited often enough) | To Q-value of whatever policy is being used (will track it as policy changes) |

##### The cliff-walking intuition

The classic teaching example. Imagine a grid:

```
S . . . . . . . . G       S = start, G = goal
C C C C C C C C C C       C = cliff (huge negative reward, episode ends)
```

Two policies the agent could learn:
- **Optimal (along the cliff edge)** — shortest path, but one slip into the cliff = disaster.
- **Safe (one row up from the cliff)** — slightly longer, but cliff isn't reachable in one wrong step.

With ε-greedy exploration during training:
- **Q-Learning** learns the optimal-along-the-edge policy. It uses $\max_{a'}$ in the update, so it knows the edge path is technically best — and it doesn't internalize the cost of *occasionally* random-walking off the cliff.
- **SARSA** learns the safe one-row-up policy. It uses $Q(s', a')$ for the action actually taken (which is occasionally the random ε-greedy off-cliff step) — so it sees the real cost of being adjacent to the cliff and learns to avoid it.

**Same algorithm structure, opposite behaviors.** This is the canonical "when to use SARSA vs Q-Learning" example.

#### Exploration strategies — same as Q-Learning

##### ε-greedy

Same formula as §14:

$$
a = \begin{cases}
\text{random} & \text{w.p. } \varepsilon \\
\arg\max_{a'} Q(s, a') & \text{w.p. } 1 - \varepsilon
\end{cases}
$$

##### Softmax (Boltzmann) — smoother exploration

Instead of "best action OR uniform random," softmax assigns each action a probability proportional to $e^{Q/\tau}$:

$$
P(a \mid s) = \frac{e^{Q(s, a) / \tau}}{\sum_{a'} e^{Q(s, a') / \tau}}
$$

| $\tau$ (temperature) | Behavior |
|---|---|
| Large | All actions roughly equally likely (uniform exploration) |
| Small | Concentrated on the highest-Q action (greedy) |
| → 0 | Pure greedy, identical to ε=0 |

This is the same softmax function from logistic-regression / multi-class classification — same math, used here as an exploration mechanism. **In LLMs, the temperature parameter is exactly this $\tau$** controlling output token sampling.

#### Convergence and tuning

SARSA converges to the optimal policy under conditions:

1. Learning rate $\alpha$ decays appropriately ($\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$ — Robbins-Monro conditions).
2. All state-action pairs are visited infinitely often (the exploration-frequency condition).
3. Stationary environment (Markov property).

Tuning knobs:

| Parameter | Effect of "high" | Effect of "low" |
|---|---|---|
| **Learning rate $\alpha$** | Faster updates, less stable | Slower learning, more stable |
| **Discount factor $\gamma$** | Far-sighted (values long-term reward) | Myopic (only short-term) |
| **Exploration $\varepsilon$ or $\tau$** | More exploration, slower convergence | Faster but risks local optima |

Standard pattern: decay $\alpha$ and $\varepsilon$ over time, hold $\gamma$ fixed.

#### Assumptions

Same as Q-Learning:
- **Markov property** — next state depends only on $(s, a)$, not history.
- **Stationary environment** — dynamics don't change over time.

#### When to use SARSA vs Q-Learning

| Use SARSA when... | Use Q-Learning when... |
|---|---|
| Online learning where exploration mistakes are costly (medical, autonomous vehicles, real money) | Offline learning or simulation where mistakes are free |
| You need a safe, robust policy that internalizes ε-greedy noise | You want the absolute optimal policy and exploration noise won't be present at deployment |
| Stability and predictability matter more than max performance | Squeezing out the last few % of performance matters |

#### Red-team angles

- **Modern RLHF (PPO and friends) is conceptually closer to SARSA than Q-Learning.** PPO is on-policy: it learns from data generated by the current policy, with a "trust region" / clipping mechanism that keeps updates conservative. The aligned-LLM behavior produced by PPO inherits SARSA-like cautiousness: the model is trained to "stay close to safe behavior even when exploring." Jailbreaks (Modules 04–05) exploit gaps in this conservative training distribution — adversarial prompts push the model into states it wasn't trained on cautiously.
- **SARSA agents are MORE predictable than Q-Learning agents.** Conservative on-policy learners tend toward safe-but-stereotyped policies → behavior is easier for an attacker to model + exploit. Q-Learning agents may be more random in mid-training but eventually deterministic.
- **The on-policy property is exploitable via "exploration spoofing."** If an attacker can manipulate what the agent perceives during ε-greedy exploration steps (e.g., adversarial perturbation of state during the random action), the SARSA agent learns that the *neighborhood* of certain states is dangerous → permanently biases the policy away from those (legitimate) states. Effectively, you can use the agent's own learning process to teach it to avoid features you care about.
- **Softmax temperature is an attack surface in LLMs.** When sampling tokens with temperature $\tau$, low $\tau$ gives consistent predictable outputs (easier to attack with crafted prompts that target argmax tokens), high $\tau$ gives diverse output (harder to predict, but each generation has higher chance of occasionally producing harmful content). Many jailbreak techniques specifically exploit high-temperature sampling.
- **Cliff-walking dynamics in security-critical RL.** Real systems (autonomous vehicles, surgical robots) prefer SARSA-style policies for safety reasons — but this means the deployed policy is suboptimal *and* avoids "edge" states. An attacker who can craft scenarios that look like "edges" can cause the agent to take long detours or refuse to act, leading to availability/DOS-style failures.
- **Same Markov + stationary assumptions = same exploitable surfaces** as Q-Learning. Inject history-dependent dynamics or non-stationarity → predictable policy failures.

**Takeaways:**
- SARSA = Q-Learning with one change: use $Q(s', a')$ for the actually-chosen $a'$ instead of $\max_{a'} Q(s', a')$.
- Off-policy (Q-Learning, aggressive, learns optimal) vs On-policy (SARSA, conservative, learns "current behavior").
- Cliff-walking: SARSA learns the safe path one row above the cliff; Q-Learning learns the optimal-but-risky edge path.
- Same exploration strategies: ε-greedy, softmax (with temperature $\tau$ — same parameter as LLM token sampling).
- Modern RLHF (PPO) is philosophically SARSA-like — on-policy + conservative — and aligned-LLM jailbreaks exploit gaps in that conservative training.

---

### 16. Introduction to Deep Learning

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A subset of ML that uses **artificial neural networks with many layers** ("deep") to learn directly from raw data. Distinguishing feature vs. classical ML: deep learning **automatically learns hierarchical features** instead of relying on hand-crafted ones. Lower layers learn primitives (edges, syllables); higher layers learn compositions (faces, sentences).

This is the foundational section for understanding everything attacked in Modules 06–11.

#### Why "deep" — the hierarchy of features

```
Image input → [edges, textures] → [shapes, parts] → [objects] → "cat"
                ↑ Layer 1            ↑ Layer 2-3      ↑ Layer 4    ↑ Output

Each layer transforms the previous layer's output into a more abstract representation.
The depth (number of layers) is what enables this compositional learning.
```

**Why this matters for security:** in classical ML you attacked the *features* (e.g. craft an email that looks ham). In DL you can attack the *learned representations* directly — perturbations that look like noise to a human but trigger specific neuron activations the model was trained to associate with target classes.

#### Anatomy of a neural network

A neural net is a stack of **layers**, each made of **neurons** connected to the previous layer by **weighted edges**.

```
   Input          Hidden           Hidden           Output
   layer          layer 1          layer 2          layer

    x₁ ●───┐     ┌─● ───┐         ┌─● ───┐         ┌─● → ŷ₁
           ╲    ╱       ╲        ╱       ╲        ╱
    x₂ ●───┼───●─────────●───────●─────────●───────●  → ŷ₂
           ╱    ╲       ╱        ╲       ╱        ╲
    x₃ ●───┘     └─● ───┘         └─● ───┘         └─● → ŷ₃

  features    learned features    learned features   predictions
              (low-level)          (high-level)
```

| Layer type | Role |
|---|---|
| **Input layer** | Receives raw features (one neuron per feature). No computation, just pass-through. |
| **Hidden layers** | The actual learning happens here. Each neuron computes $\sigma(\mathbf{w} \cdot \mathbf{x} + b)$ — same shape as logistic regression from §5. "Deep" = many hidden layers. |
| **Output layer** | Produces final predictions. Number + activation depends on task: 1 sigmoid for binary, $K$-way softmax for multi-class, linear for regression. |

Each neuron's computation:

$$
z = \mathbf{w} \cdot \mathbf{x} + b, \qquad a = \phi(z)
$$

where $\phi$ is the activation function. Stack these per-neuron computations into matrix form per layer:

$$
\mathbf{a}^{(\ell)} = \phi(W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)})
$$

That's literally the entire forward pass — repeat once per layer.

#### Activation functions — where the non-linearity comes in

Without activation functions, stacking linear layers just produces another linear function (a stack of matrix multiplies = one big matrix multiply). Activations introduce non-linearity → the network can model curved decision boundaries.

| Activation | Formula | Range | When used |
|---|---|---|---|
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $(0, 1)$ | Binary classification output; rare in hidden layers (gradient saturation) |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Older RNNs; centered version of sigmoid |
| **ReLU** | $\text{ReLU}(z) = \max(0, z)$ | $[0, \infty)$ | **The default for hidden layers in modern DL** |
| **Softmax** | $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | $(0, 1)$, sums to 1 | Multi-class classification output (probabilities over classes) |

##### Why ReLU dominates

```
ReLU(z) = max(0, z)

         │
       3 │           ╱
       2 │         ╱
       1 │       ╱
       0 ┤─────●────────────→ z
            -2 -1  1  2  3
```

Cheap (one comparison + max), gradient is exactly 1 for $z > 0$ (no saturation), and the resulting network is **piecewise linear** — every input lives in some "linear region" defined by which ReLUs are active. This piecewise-linearity is *what makes gradient-based adversarial attacks work* — the gradient $\nabla_x f$ is well-defined and informative within each region.

#### The forward pass — input → prediction

```
x → [layer 1: z = Wx + b, a = ReLU(z)] →
  → [layer 2: z = Wa + b, a = ReLU(z)] →
  → [output:  z = Wa + b, a = softmax(z)] →  ŷ
```

Just matrix multiplies + activations, repeated. For a typical net with 5–100 layers and millions of weights, this is hundreds of millions of multiply-adds — but a GPU does it in milliseconds.

#### The loss function — measuring how wrong we are

The **loss** $\mathcal{L}(\hat{y}, y)$ is a scalar that measures the gap between prediction $\hat{y}$ and target $y$. Training = adjust weights to minimize average loss over the training set.

| Task | Loss | Formula |
|---|---|---|
| **Regression** | Mean Squared Error | $\mathcal{L} = \frac{1}{N} \sum_i (\hat{y}_i - y_i)^2$ |
| **Binary classification** | Binary Cross-Entropy | $\mathcal{L} = -\frac{1}{N} \sum_i [y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)]$ |
| **Multi-class classification** | Categorical Cross-Entropy | $\mathcal{L} = -\frac{1}{N} \sum_i \sum_c y_{i,c} \log \hat{y}_{i,c}$ |

**Cross-entropy** is the classification workhorse. Notice the $\log$ — same MathJax/LaTeX-rendered math from §2 — and the negative sign. The negative log of a probability is small when the probability is large (right answer with confidence) and large when the probability is small (wrong answer or low confidence).

#### Backpropagation — how we compute the gradient

Backprop is the algorithm that computes $\nabla_W \mathcal{L}$ — the gradient of the loss with respect to every weight in the network. It works by applying the **chain rule** of calculus, propagating gradients from the output layer backward to the input.

High-level:

```
1. Forward pass: compute predictions ŷ and the loss L.

2. Backward pass: starting at the output, compute ∂L/∂W for each layer
   working backward, using the chain rule to combine gradients across layers.

3. Weight update: each weight is nudged opposite to its gradient:
       W ← W − α · ∂L/∂W
   where α is the learning rate.

4. Repeat for the next minibatch of training examples.
```

You don't typically implement backprop yourself — frameworks (PyTorch, TensorFlow, JAX) do it automatically via **autograd** (automatic differentiation). But conceptually: every weight ends up with a gradient saying "increase me to reduce loss" or "decrease me to reduce loss", and the optimizer steps in that direction.

**This gradient is the central object adversarial attacks weaponize.** FGSM, I-FGSM, DeepFool, JSMA all compute gradients — but with respect to the **input** $x$, not the weights $W$ — and use that gradient to craft perturbations.

#### Optimizers — different ways to apply the gradient

| Optimizer | Idea | When |
|---|---|---|
| **SGD** | $W \leftarrow W - \alpha \nabla_W \mathcal{L}$ — simple gradient step | Baseline, well-understood; sometimes best after careful LR tuning |
| **SGD + Momentum** | Carries velocity across updates: $v \leftarrow \mu v + \nabla_W \mathcal{L}$, $W \leftarrow W - \alpha v$ | Smooths SGD's noisy gradient; standard for image classification |
| **Adam** | Adaptive per-parameter learning rates using running estimates of first and second moments of the gradient | **Default for most modern DL** — fast convergence, low tuning |
| **RMSprop** | Adam's predecessor — adaptive per-parameter learning rates only | RNN training, where it remained popular for a while |

#### Hyperparameters — set before training, control everything

| Hyperparameter | Effect |
|---|---|
| **Learning rate** $\alpha$ | Step size. Too high → divergence; too low → slow + stuck in local minima |
| **Number of layers (depth)** | More layers = more representational power, more risk of vanishing/exploding gradients |
| **Neurons per layer (width)** | More = more capacity, more risk of overfitting |
| **Batch size** | How many examples per gradient step; affects gradient noise + memory |
| **Number of epochs** | Passes through the dataset; too many → overfit, too few → underfit |
| **Regularization strength** $\lambda$ | L1/L2 penalty (from §3) controlling overfitting |
| **Dropout rate** | Randomly zero some neurons during training as a regularizer |

Tuning these well is half the practical art of DL. Tools: grid search, random search, Bayesian optimization, learning-rate schedulers.

#### Red-team angles — this section is the foundation of every later attack module

- **Backpropagation creates the gradient → adversarial attacks weaponize it.** FGSM (Module 09) computes $\nabla_x \mathcal{L}$ — gradient of loss w.r.t. **input** instead of w.r.t. weights — and steps the input in that direction to maximize loss → misclassification. The same backprop machinery used to train the model is what an attacker uses to break it.
- **ReLU's piecewise linearity is the geometric basis of evasion attacks.** Within each "linear region" (defined by which ReLUs are active for a given input), the network is exactly linear. So a small perturbation in the input causes a *predictable linear change* in the output — exactly the regime where gradient-based attacks succeed. Networks built entirely from non-piecewise-linear activations (sigmoid-only, etc.) are harder to attack with single-step methods, but iterative attacks still work.
- **Cross-entropy loss IS what FGSM follows.** The standard FGSM formula is: $x_{\text{adv}} = x + \varepsilon \cdot \text{sign}(\nabla_x \mathcal{L}_{\text{CE}}(f(x), y))$. The cross-entropy loss is the function being maximized; "moving in the gradient direction" means moving the input toward higher loss for the correct class.
- **Targeted attacks invert the loss.** Untargeted: maximize loss for correct class (push to *anywhere* wrong). Targeted: minimize loss for the *attacker-chosen* class (push specifically to a chosen wrong class). Module 09 covers both.
- **Adam's adaptive learning rates affect attack difficulty.** Adversarial training (training on adversarial examples to defend) interacts with Adam's adaptivity in non-obvious ways — sometimes the optimizer's own state can be poisoned via crafted training inputs.
- **Hyperparameters affect attack surfaces.**
  - Smaller learning rates → smoother loss landscapes → easier to find adversarial examples (gradients more reliable).
  - More layers → richer feature hierarchy → more places adversarial perturbations can exploit (but also harder to engineer).
  - Higher dropout → randomized inference → harder for single-step attacks (model output varies between calls), but iterative attacks still succeed via expectation-over-random-passes.
- **Output activation determines attack target.** Softmax outputs are probabilities; attacks against a classifier with sigmoid output are trivially easier (one-dimensional decision boundary). Most production classifiers use softmax → multi-dimensional attack target.
- **The "learns features automatically" property is a double-edged sword.** Defenders can't enumerate which features the model relies on, so they can't defend each one explicitly. Attackers can probe the model to discover spurious learned features (e.g. "this image classifier secretly relies on the green grass background to detect cows") and exploit them.

**Takeaways:**
- Deep learning = many-layer neural networks that learn hierarchical features automatically.
- Per-neuron computation: $z = \mathbf{w} \cdot \mathbf{x} + b$, $a = \phi(z)$. Per-layer: $\mathbf{a}^{(\ell)} = \phi(W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)})$.
- Activations: ReLU dominates hidden layers (cheap, piecewise linear); sigmoid/softmax for outputs.
- Loss functions: MSE for regression, cross-entropy for classification.
- **Backpropagation** computes $\nabla_W \mathcal{L}$ via the chain rule — the gradient is the central object both training and attacks rely on.
- Optimizers: SGD/Momentum/Adam/RMSprop. Adam is the default.
- Hyperparameters: LR, depth, width, batch size, epochs, regularization, dropout.
- **Adversarial attacks (FGSM, DeepFool, JSMA) reuse the same backprop gradient — but with respect to the input instead of weights — to craft perturbations that maximize loss.**

---

### 17. Perceptrons

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

The **atomic unit** of neural networks. A perceptron models a single artificial neuron: take weighted inputs, sum them, add a bias, apply an activation, output one number. Every modern deep network is just thousands-to-billions of these stacked in layers.

Recognition moment: **a perceptron is mathematically identical to logistic regression (§5)** — just with a step activation instead of a sigmoid. Once this clicks, the entire NN vocabulary makes sense.

#### Anatomy

```
   x₁ ●──── w₁ ───┐
                  │
   x₂ ●──── w₂ ───┤
                  ├─── Σ ──── (+ b) ──── f(·) ──── y
   x₃ ●──── w₃ ───┤
                  │
   ⋮              │
                  │
   xₙ ●──── wₙ ───┘
```

| Component | Symbol | Role |
|---|---|---|
| **Inputs** | $x_1, x_2, \dots, x_n$ | Feature values |
| **Weights** | $w_1, w_2, \dots, w_n$ | Learned importance of each input |
| **Summation** | $\sum w_i x_i$ | Weighted sum of inputs |
| **Bias** | $b$ | Shift the activation threshold (lets the neuron fire even with all-zero inputs) |
| **Activation** | $f$ | Non-linear function applied to the weighted sum + bias |
| **Output** | $y$ | Scalar result — typically binary (0/1) for a classic perceptron |

The whole computation in one line:

$$
y = f\!\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w} \cdot \mathbf{x} + b)
$$

**Notice:** $\mathbf{w} \cdot \mathbf{x} + b$ is *exactly* the logistic-regression and SVM score. The only difference between these three algorithms is what $f$ is:

| Algorithm | Activation $f$ |
|---|---|
| **Perceptron (classic)** | Step function: $f(z) = 1$ if $z > 0$, else $0$ |
| **Logistic regression (§5)** | Sigmoid: $f(z) = \frac{1}{1 + e^{-z}}$ |
| **Linear SVM (§8)** | Sign function: $f(z) = \text{sign}(z)$ (with margin constraint during training) |

They're all **linear classifiers** wearing different masks.

#### Worked example — HTB's "play tennis" decision

Setup — a perceptron with four inputs, weights, and bias:

| Feature | Value | Weight |
|---|---|---|
| Outlook (Sunny=0, Overcast=1, Rainy=2) | 0 | $w_1 = 0.3$ |
| Temperature (Hot=0, Mild=1, Cool=2) | 1 | $w_2 = 0.2$ |
| Humidity (High=0, Normal=1) | 0 | $w_3 = -0.4$ |
| Wind (Weak=0, Strong=1) | 0 | $w_4 = -0.2$ |
| Bias | — | $b = 0.1$ |

Step 1 — weighted sum:

$$
\mathbf{w} \cdot \mathbf{x} = (0.3)(0) + (0.2)(1) + (-0.4)(0) + (-0.2)(0) = 0.2
$$

Step 2 — add bias:

$$
z = 0.2 + 0.1 = 0.3
$$

Step 3 — apply step activation:

$$
y = f(0.3) = 1 \quad (\text{since } 0.3 > 0) \quad \Rightarrow \text{Play Tennis}
$$

In code:

```python
def step_activation(z):
    return 1 if z > 0 else 0

# Features: sunny, mild, high humidity, weak wind
x = [0, 1, 0, 0]
w = [0.3, 0.2, -0.4, -0.2]
b = 0.1

z = sum(wi * xi for wi, xi in zip(w, x)) + b
y = step_activation(z)
print(y)   # → 1  (Play Tennis)
```

That's the complete forward pass of a perceptron. A 100-layer deep neural net is this same computation repeated millions of times.

#### The critical limitation — linear separability

A single perceptron can only learn **linearly separable** boundaries — the same limitation as logistic regression and linear SVM. The boundary is:

$$
\mathbf{w} \cdot \mathbf{x} + b = 0
$$

which is a hyperplane. Any decision that requires a non-linear boundary is impossible for a single perceptron.

##### The XOR problem — the classic counterexample

XOR: output 1 if exactly one input is 1, else 0.

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Plotted in 2D:

```
  x₂
  │
  1 ●────────────○
  │              │
  │              │
  │              │
  0 ○────────────●
    0            1   x₁

  ● = class 1 (XOR outputs 1)
  ○ = class 0 (XOR outputs 0)
```

**No single straight line separates the ●s from the ○s.** Try it — whichever line you draw, two of the four points end up on the wrong side. XOR is not linearly separable → no perceptron can solve it.

##### Historical consequence — the first AI winter

Minsky & Papert's 1969 book *Perceptrons* proved the XOR limitation and argued the perceptron framework was fundamentally limited. This (along with other factors) triggered the first "AI winter" — a period of reduced funding and interest in neural networks that lasted roughly until the mid-1980s.

What revived the field was the combination of:
1. **Multi-layer perceptrons (MLPs)** — stacking perceptrons across hidden layers solves XOR and far more.
2. **Backpropagation** (popularized 1986) — made training those multi-layer networks computationally tractable.

Every modern deep network is the direct descendant of this revival — MLP + backprop + better activations + more compute.

#### Training a perceptron — the perceptron learning rule

Rosenblatt's original rule (1958, pre-backprop): if the perceptron makes a mistake, nudge weights toward the correct answer:

$$
w_i \leftarrow w_i + \alpha (y_{\text{true}} - y_{\text{pred}}) \, x_i
$$

where $\alpha$ is the learning rate. Simple, guaranteed to converge if the data is linearly separable (the **perceptron convergence theorem**), and the conceptual seed of gradient descent. The more general method that supplanted it is gradient descent on a differentiable loss, which required replacing the step activation with sigmoid/ReLU (step isn't differentiable).

#### Red-team angles

- **A perceptron is the simplest possible gradient-attack target.** The decision boundary $\mathbf{w} \cdot \mathbf{x} + b = 0$ is known to the attacker once they have the weights. The smallest perturbation to flip the output is directly along the direction $\mathbf{w}$, scaled by just enough to cross zero: $\delta = -\frac{\mathbf{w} \cdot \mathbf{x} + b}{\lVert \mathbf{w} \rVert^2} \mathbf{w}$. This is the **DeepFool** attack (Module 09) in its simplest form — generalized to multi-layer networks by iterating on the piecewise-linear regions of the network.
- **Step activation breaks differentiability.** That's why modern networks don't use it — backprop needs derivatives. But in the minority of deployments that use non-differentiable components (quantized models, binarized networks), gradient-based attacks are harder → *gradient-free* attacks (genetic algorithms, zeroth-order optimization) become relevant. Some ML defenses intentionally add non-differentiable components as "gradient masking" — but those defenses have been repeatedly broken by adaptive attacks.
- **Weight extraction against linear classifiers is trivial** with enough query access. $(n+1)$ linearly independent queries suffice to recover the $n$ weights + bias exactly via linear algebra. Module 07's model reverse engineering applies here as the easiest case.
- **The XOR limitation motivated the entire deep-learning revolution — and the attack surfaces that come with it.** A world of linear classifiers would have been easy to defend (coefficient inspection, interpretability). Deep networks' non-linear compositional capacity is exactly what makes them both powerful and attackable.
- **The perceptron learning rule IS an adversarial-training primitive.** If an attacker can control $(y_{\text{true}}, x)$ pairs the model sees during online training, they can steer the weights arbitrarily via the same update rule the defender is using to improve the model. Online-learning systems are especially exposed.

**Takeaways:**
- Perceptron = single artificial neuron = logistic regression with a step activation. Forward pass: $y = f(\mathbf{w} \cdot \mathbf{x} + b)$.
- Every neuron in every modern deep network is a perceptron with a different activation (typically ReLU).
- Single-layer perceptrons can only learn **linearly separable** boundaries — XOR is the famous counterexample.
- Multi-layer perceptrons (MLPs) + backpropagation solved XOR and launched modern DL — covered next in §18.
- Decision boundary $\mathbf{w} \cdot \mathbf{x} + b = 0$ is the simplest gradient-attack target; DeepFool's closed-form solution for linear models is one line of algebra.

---

### 18. Neural Networks

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

**Multi-Layer Perceptrons (MLPs)** — stack perceptrons into layers, connect them fully, and training figures out which combinations of inputs matter. This architecture solves problems single perceptrons can't (starting with XOR), and it's the backbone of every modern deep-learning model you'll attack.

Most content here was covered in §16 (activations, backprop, gradient descent) and §17 (the per-neuron computation). This section's job: the *multi-layer* aspects.

#### Architecture — three types of layers

```
   Input         Hidden layer 1       Hidden layer 2      Output
   layer         (features-of-        (features-of-       layer
                  inputs)              features)

    x₁ ●─────────●─────────────────────●──────────────────● → ŷ₁
       ╲╱                             ╲╱
       ╱╲                             ╱╲
    x₂ ●─────────●─────────────────────●──────────────────● → ŷ₂
       ╲╱                             ╲╱
       ╱╲                             ╱╲
    x₃ ●─────────●─────────────────────●──────────────────● → ŷ₃

       ↑         ↑         ↑          ↑         ↑         ↑
       raw      low-level learned    high-level learned   final
       features features              features             predictions
```

**Fully-connected** (or "dense"): every neuron in one layer connects to every neuron in the next. This is the default for MLPs — other architectures like CNNs use restricted connectivity (§19) and RNNs add temporal connections (§20).

| Layer | Role | Notes |
|---|---|---|
| **Input** | Pass-through of feature values | One neuron per feature; no computation |
| **Hidden** | Learn intermediate representations | One or more layers; "deep" = many of these |
| **Output** | Produce final prediction | Size + activation chosen to match the task |

#### Output-layer sizing — task-dependent

| Task | Output neurons | Activation |
|---|---|---|
| Binary classification | 1 | Sigmoid (probability of positive class) |
| Multi-class classification | $K$ (one per class) | Softmax (probability distribution) |
| Regression | 1 | Linear (no activation) |
| Multi-output regression | $K$ (one per target) | Linear |

This is a frequent source of attack-surface detail: for binary, the logit $z$ is a single scalar; for multi-class, the attacker has $K$ logits to work with. Targeted adversarial attacks exploit the multi-class structure.

#### Per-neuron computation (recap from §17)

Every neuron in every layer does:

$$
z = \mathbf{w} \cdot \mathbf{a}^{(\ell-1)} + b, \qquad a = \phi(z)
$$

where $\mathbf{a}^{(\ell-1)}$ is the activation vector from the previous layer. Stack per-layer:

$$
\mathbf{a}^{(\ell)} = \phi(W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)})
$$

The matrix $W^{(\ell)}$ has one row per neuron in layer $\ell$, one column per neuron in layer $\ell-1$. Size = (neurons in $\ell$) × (neurons in $\ell-1$).

#### How multi-layer solves XOR

From §17: a single perceptron can't separate XOR's diagonal classes. An MLP with just **one hidden layer of 2 neurons** solves it — the hidden layer transforms the inputs into a new feature space where the classes *are* linearly separable.

Conceptually:

```
x₁ ●─┐                 h₁ ●   (learns: "x₁ AND NOT x₂" — fires on (1,0))
     ├────hidden─────┤                                    
x₂ ●─┘                 h₂ ●   (learns: "NOT x₁ AND x₂" — fires on (0,1))

Then output = h₁ OR h₂ = 1 when exactly one of x₁, x₂ is 1 = XOR
```

The hidden layer's job is to warp the input space so the output layer's linear decision boundary suffices. **Every deep network works this way — each layer is a learned coordinate transformation that makes the next layer's job easier.** This is the hierarchical-feature intuition from §16 made concrete.

#### The Universal Approximation Theorem

A theorem from 1989 (Cybenko, then Hornik): **an MLP with a single hidden layer of enough neurons, using a non-linear activation, can approximate any continuous function on a bounded domain to arbitrary accuracy.**

In plain English: MLPs are **universal function approximators**. If there's a function mapping your inputs to your outputs, an MLP can in principle represent it.

Caveats:
- "Enough neurons" can be exponentially many — narrow-but-deep networks are usually more parameter-efficient than wide-but-shallow ones.
- "Can represent" ≠ "can learn from data" — you still need enough training data + a good optimizer to find those weights.

**Security implication:** because MLPs can represent any function, an attacker who can poison training data enough (Module 06) can in principle make the model learn *any* input-output mapping — including arbitrary backdoors (trojan attacks).

#### Activation functions — brief recap from §16

| Activation | Range | Typical use | Attack-relevant property |
|---|---|---|---|
| **ReLU** | $[0, \infty)$ | Hidden layers (default) | Piecewise linear → exploitable by single-step gradient attacks |
| **Sigmoid** | $(0, 1)$ | Binary output | Saturation → gradient masking issue |
| **Tanh** | $(-1, 1)$ | RNN hidden states (historical) | Similar saturation issue as sigmoid |
| **Softmax** | $(0, 1)$, sums to 1 | Multi-class output | Gradient w.r.t. logits has closed form — the standard attack target |

#### Training — backpropagation + gradient descent

Recap from §16:

```
For each minibatch of training examples:

  1. FORWARD PASS:  compute predictions ŷ by feeding x through all layers.

  2. COMPUTE LOSS:  L = loss_function(ŷ, y)   (e.g. cross-entropy)

  3. BACKWARD PASS: compute ∂L/∂W for every weight via chain rule,
                    propagating gradients layer-by-layer from output → input.

  4. UPDATE WEIGHTS: W ← W − α · ∂L/∂W      (gradient descent step)

Repeat over many epochs until convergence.
```

##### The gradient descent update in detail

$$
W^{(\ell)} \leftarrow W^{(\ell)} - \alpha \frac{\partial \mathcal{L}}{\partial W^{(\ell)}}
$$

$$
b^{(\ell)} \leftarrow b^{(\ell)} - \alpha \frac{\partial \mathcal{L}}{\partial b^{(\ell)}}
$$

One step per minibatch. Over thousands of minibatches, the weights converge to a local minimum of the loss. Adam (from §16) replaces vanilla SGD with adaptive per-parameter learning rates — same shape, smarter step sizes.

#### Variations of gradient descent (vocabulary)

| Variant | Batch size | Trade-off |
|---|---|---|
| **Batch GD** | All data at once | Stable gradient, slow per step, needs lots of memory |
| **Stochastic GD (SGD)** | One sample at a time | Noisy updates, fast per step, escapes local minima via noise |
| **Mini-batch GD** | Typical: 32–256 samples | Best practical trade-off — **the standard for DL** |

#### Red-team angles

- **MLPs are piecewise linear (when using ReLU).** Each input $x$ lies in one "linear region" defined by which ReLUs are active — within that region, the net acts like a linear function $\mathbf{w}_{\text{local}} \cdot \mathbf{x} + b_{\text{local}}$. This is why **gradient-based attacks (FGSM, etc.) work**: the gradient $\nabla_x f$ correctly predicts how small perturbations change the output. Attacks essentially use the local linear approximation.
- **Adversarial examples transfer across MLP architectures.** An adversarial perturbation crafted against one model often fools other models trained on the same data — because different networks learn correlated decision surfaces. This enables **black-box attacks**: train a surrogate MLP, craft attacks against it, transfer to the target. Module 09 covers this explicitly.
- **Universal Approximation = arbitrary-backdoor capacity.** Because MLPs can approximate any function, a sufficiently-powerful attacker with enough training-data influence can insert arbitrary input-output mappings into the trained model. Trojan attacks (Module 06) exploit this: the model learns the *normal* task correctly + a *hidden* triggered-input-to-target-output mapping, and the universal-approximation property says there's always enough capacity for both.
- **Depth/width interact with attack success.**
  - **Wider hidden layers** → more linear regions → more diverse attack directions → easier to find adversarial examples.
  - **Deeper networks** → more complex decision boundaries → harder single-step attacks, but **better adversarial-example transfer** (deep features are more universal).
- **Model extraction scales with parameter count.** An attacker querying the model to recover weights needs roughly $\Omega(\text{number of parameters})$ queries for exact extraction. Modern LLMs have billions of parameters — practical exact extraction is infeasible, but *functional* extraction (training a surrogate that behaves similarly) is much cheaper and is what Module 07 actually targets.
- **Memorization enables membership inference.** MLPs have enough capacity to memorize individual training points, especially when overfit. Module 11's membership inference attacks exploit this: a point the model has "seen" during training gets a different loss/confidence signature than an unseen point.
- **Weight matrices have exploitable structure.** Singular value decomposition of $W^{(\ell)}$ reveals which input directions the layer is most/least sensitive to. Module 07 extraction attacks on linear layers reduce to SVD (from §11).
- **Gradient of loss w.r.t. input** ($\nabla_x \mathcal{L}$, different from the training gradient $\nabla_W \mathcal{L}$) is the backbone of all gradient-based evasion. Same backprop machinery, chain rule stops at the input instead of continuing into the weights.

**Takeaways:**
- MLP = stacked perceptrons with non-linear activations → solves non-linearly separable problems (XOR and beyond).
- Three layer types: input, hidden (one or more), output. Size of output layer matches task (1 for binary, $K$ for multi-class).
- **Universal Approximation Theorem:** an MLP with enough hidden neurons can approximate any continuous function.
- Training = backprop (compute $\nabla_W \mathcal{L}$ via chain rule) + gradient descent (step weights opposite to gradient).
- Mini-batch SGD / Adam are the practical workhorses.
- **Adversarial attacks reuse the same chain rule with $\nabla_x \mathcal{L}$ instead of $\nabla_W \mathcal{L}$** — the crucial flip that turns training code into attack code.

---

### 19. Convolutional Neural Networks

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-14

A specialized architecture for **grid-like data** (images, video, sometimes text). CNNs exploit two key properties of such data — **locality** (nearby pixels are related) and **stationarity** (a feature means the same thing anywhere in the image) — to be far more parameter-efficient than fully-connected MLPs.

CNNs are the architecture you'll attack across most of modules 06, 09, and 10. Classic canonical examples: ImageNet classification, stop-sign recognition in self-driving cars, malware classification when bytes are visualized as 2D images (Module 02's Malware Classification).

#### Three layer types

```
Input image → [Conv → ReLU → Pool] × N → Flatten → [FC → ReLU] × M → Softmax → class

─────────────────────────────────────   ──────   ────────────────────────
        feature extraction                |              classification
   (learned hierarchical filters)    flatten the 2D         (MLP from §18)
                                     features to 1D
```

| Layer | Role | What's new vs MLP |
|---|---|---|
| **Convolutional** | Apply learnable filters that slide across input to detect features | Replaces fully-connected layer — fewer weights, local receptive fields, weight sharing |
| **Pooling** | Downsample feature maps — max or average over small windows | Reduces spatial dimensions, adds translation invariance |
| **Fully Connected** | Classic MLP layers at the end for final reasoning/classification | Same as §18 |

#### The convolution operation

A **filter** (also called **kernel**) is a small matrix of learned weights — typically 3×3 or 5×5. It slides across the input, and at each position computes a single output value: the **elementwise multiplication** of the filter and the patch it's over, summed.

```
  Input (5×5)              Filter (3×3)           Feature map (3×3)
  ┌──┬──┬──┬──┬──┐         ┌──┬──┬──┐
  │ 1│ 0│ 2│ 1│ 3│         │ 1│ 0│-1│
  ├──┼──┼──┼──┼──┤         ├──┼──┼──┤
  │ 2│ 1│ 0│ 1│ 2│   ★     │ 1│ 0│-1│    =      ┌──┬──┬──┐
  ├──┼──┼──┼──┼──┤  conv   ├──┼──┼──┤           │ ?│ ?│ ?│
  │ 0│ 2│ 1│ 3│ 0│         │ 1│ 0│-1│           ├──┼──┼──┤
  ├──┼──┼──┼──┼──┤         └──┴──┴──┘           │ ?│ ?│ ?│
  │ 1│ 2│ 0│ 2│ 1│         (vertical-edge       ├──┼──┼──┤
  ├──┼──┼──┼──┼──┤          detector —          │ ?│ ?│ ?│
  │ 3│ 0│ 1│ 2│ 1│          Sobel-like)         └──┴──┴──┘
  └──┴──┴──┴──┴──┘
```

Formally, the 2D convolution of input $I$ with filter $K$ at position $(i, j)$:

$$
(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)
$$

Key properties:

| Property | What it is | Why it matters |
|---|---|---|
| **Weight sharing** | The same filter applies at every position | A "vertical edge" anywhere in the image activates the same filter — massive parameter reduction vs a fully-connected layer, and built-in translation invariance |
| **Local connectivity (receptive field)** | Each output cell depends only on a small local patch of input | Captures local structure efficiently; dramatically reduces weights compared to FC layers |
| **Multiple filters per layer** | A typical conv layer has 32–512 filters | Each filter learns a different feature (horizontal edges, vertical edges, diagonal edges, color blobs, etc.); stacked outputs form a multi-channel feature map |

##### Hyperparameters specific to conv layers

| Hyperparameter | Meaning |
|---|---|
| **Kernel size** | Filter dimensions (3×3, 5×5, 7×7). Smaller = sharper locality; larger = more context per step |
| **Stride** | How far the filter moves between applications. Stride 1 = overlapping patches; stride 2 = halves output size |
| **Padding** | Zeros added around input edges to control output size. "Same" padding preserves spatial dimensions; "valid" lets output shrink |
| **Number of filters** | Output channels = number of different features this layer learns |

#### Pooling — downsample the feature maps

After a conv layer, a **pooling layer** shrinks each feature map's spatial dimensions by taking a summary statistic over small windows.

```
  Max pooling (2×2 window, stride 2):

   Input 4×4             Output 2×2
   ┌──┬──┬──┬──┐
   │ 1│ 3│ 2│ 4│         ┌──┬──┐
   ├──┼──┼──┼──┤         │ 3│ 4│       max(1,3,2,1) = 3
   │ 2│ 1│ 1│ 0│    →    ├──┼──┤       max(2,4,1,0) = 4
   ├──┼──┼──┼──┤         │ 5│ 8│       max(0,5,3,2) = 5
   │ 0│ 5│ 3│ 2│         └──┴──┘       max(1,4,7,8) = 8
   ├──┼──┼──┼──┤
   │ 1│ 4│ 7│ 8│
   └──┴──┴──┴──┘
```

| Pooling type | Operation | When |
|---|---|---|
| **Max pooling** | Take the maximum over the window | Most common — preserves strong activations, discards weaker ones |
| **Average pooling** | Take the mean over the window | Smoother summarization; used in "global average pooling" at final layers |

Pooling achieves three things: (1) reduces computation for subsequent layers, (2) adds a small degree of translation invariance (the exact pixel doesn't matter, just "something strong happened here"), (3) reduces overfitting by summarizing.

#### Hierarchical feature learning — the CNN story

Stack conv-pool blocks deep, and the features progress from primitive to abstract:

```
Layer 1: edges, corners, simple textures
          ↓
Layer 2: textures + edges combine → patterns, simple shapes
          ↓
Layer 3: patterns combine → object parts (eyes, wheels, characters)
          ↓
Layer 4: object parts combine → whole objects (faces, cars, digits)
          ↓
        FC layers → final classification
```

This is the hierarchical feature learning from §16, now architecturally baked in: conv layers at different depths specialize on features of different abstraction levels.

#### A canonical image-classifier CNN

```
Input: 224×224 RGB image (shape: 3 × 224 × 224)
  ↓
Conv(64 filters, 3×3) → ReLU → Pool(2×2)    # → 64 × 112 × 112
  ↓
Conv(128 filters, 3×3) → ReLU → Pool(2×2)   # → 128 × 56 × 56
  ↓
Conv(256 filters, 3×3) → ReLU → Pool(2×2)   # → 256 × 28 × 28
  ↓
Conv(512 filters, 3×3) → ReLU → Pool(2×2)   # → 512 × 14 × 14
  ↓
Flatten                                     # → 512 × 196 = 100352
  ↓
FC(4096) → ReLU → Dropout
  ↓
FC(1000) → Softmax                          # → probabilities over 1000 classes
```

This is roughly the shape of VGG16 (a classic CNN). ResNet, DenseNet, EfficientNet, and modern variants add skip connections and other tricks — but the core conv-pool-FC pattern remains.

#### The four key assumptions — and how attackers exploit each

| Assumption | What it means | How attackers exploit it |
|---|---|---|
| **Grid-like structure** | Data has 2D/3D spatial layout | Exploit the grid structure to craft perturbations along it (pixel-level attacks) |
| **Spatial hierarchy** | Features compose from simple (low) to abstract (high) | Target specific hierarchy levels — e.g. flip only the edge-level features, leave high-level unchanged |
| **Feature locality** | Relevant correlations are in small neighborhoods | **Adversarial patches** — a small localized region can hijack the prediction (physical-world stop-sign attacks) |
| **Feature stationarity** | Same feature = same meaning everywhere (weight sharing) | **Universal adversarial perturbations** — one perturbation pattern that fools the CNN regardless of where you put it in the image |

#### Red-team angles — this is THE architecture for Modules 06, 09, 10

- **Every classic adversarial-example paper is about attacking CNNs.** Szegedy 2013, Goodfellow FGSM 2014, DeepFool 2016, Carlini-Wagner 2017, JSMA, ElasticNet — all evaluated primarily on CNN image classifiers. Module 09 implements FGSM against CNNs; Module 10 implements JSMA/ElasticNet against CNNs.
- **Pixel-level attacks work because CNNs preserve pixel structure.** A small $L_\infty$ perturbation at every pixel (FGSM) aggregates into coherent feature-level signals because the receptive fields of early layers see overlapping neighborhoods. The gradient $\nabla_x \mathcal{L}$ has spatial structure, and attackers follow it.
- **Adversarial patches exploit feature locality + stationarity together.** A small (e.g. 50×50) crafted patch placed *anywhere* in a larger image can make a CNN classify the whole image as a chosen target class. Because the CNN applies the same filters everywhere, the patch's local activations propagate up regardless of placement. Eykholt et al. 2017's "Robust Physical-World Attacks" on stop signs is the canonical example — physical stickers that flip a CNN's classification.
- **Universal adversarial perturbations** (Moosavi-Dezfooli 2017) — a single perturbation image that fools the target CNN when added to *almost any* input. Exploits stationarity at scale.
- **Trojan attacks (Module 06) are designed around CNN structure.** Inject training examples with a small trigger pattern (e.g. a 3×3 pixel sticker in the corner) labeled with an attacker-chosen class. The CNN learns a filter chain that responds to the trigger → at deployment, any input with that pattern triggers the backdoor classification. The weight-sharing + local-receptive-field properties are exactly why this works: the trojan trigger can be placed anywhere.
- **Tensor steganography (Module 06)** hides data in CNN weight tensors. Because CNNs have huge weight matrices with some numerical redundancy, attackers can modulate low-order bits of weights to encode hidden payloads without noticeably affecting classification. Covered in detail in Module 06 §5.
- **Feature visualization (DeepDream, Olah et al.) reveals what each filter detects.** Attackers use this to understand which filters to target — craft inputs that maximally activate a specific filter chain leading to the desired output. Google "activation atlases" for the research literature.
- **JSMA (Module 10) is specifically designed for CNN pixel attacks.** It iteratively modifies one or two pixels at a time based on gradient saliency, producing extremely sparse adversarial examples. Works so well on CNNs because of the local-receptive-field structure: targeting the right pixels has focused effects on specific filter activations.
- **Dropout does not help against adversarial examples.** CNNs with aggressive dropout are still trivially attacked — dropout randomizes training but the learned features are still susceptible to gradient-based perturbations. This is documented extensively; don't let anyone tell you dropout is a defense.
- **Malware-as-image classifiers** (Module 02's Malware Classification exercise uses ResNet50) apply the same CNN machinery — so every CNN attack transfers. Adversarial perturbations of malware-image representations can flip malicious → benign classifications.

**Takeaways:**
- CNN = stacked conv + pool blocks for hierarchical feature extraction from grid data, followed by FC layers for classification.
- Convolution operation: filter slides across input, elementwise multiply + sum at each position → feature map. $(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) K(m, n)$.
- Weight sharing + local connectivity = fewer parameters + translation invariance.
- Hierarchy: early layers = primitives (edges), deep layers = objects.
- Four assumptions: grid structure, spatial hierarchy, locality, stationarity — **each corresponds to a specific adversarial attack family**.
- **THE architecture for Modules 06, 09, 10 attacks.** Pixel-level adversarial examples, adversarial patches, universal perturbations, Trojan backdoors, tensor steganography — all target CNN structure.

---

### 20. Recurrent Neural Networks

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-15

An architecture for **sequential data** — where order matters. Unlike CNNs (which assume spatial structure) or MLPs (which assume independent features), RNNs assume a **temporal or sequential** relationship between inputs. Each step's output depends on previous steps via a **hidden state** that carries context forward.

Most modern text processing uses **transformers** (§22), not RNNs. But understanding RNNs matters because:
1. The vanishing-gradient problem here is exactly *why* transformers won.
2. RNNs remain in production for speech recognition, network-traffic IDS, system-call sequence analysis, and some malware behavior models.
3. The stateful "carry context forward" intuition carries over to how LLMs generate text token-by-token at inference time.

#### The recurrent loop

A single RNN cell applied repeatedly across timesteps:

```
                     ┌─────────────────────────┐
                     │   RNN cell              │
                     │                         │
                     │   h_t = tanh(W_h h_{t−1}│
                     │          + W_x x_t + b) │
                     │                         │
     h_{t−1} ────────┼─────────────────────────┼───── h_t (→ next step)
                     │                         │
     x_t ────────────┼─────────────────────────┼───── y_t (optional output)
                     └─────────────────────────┘
```

At each timestep $t$:
- Input $x_t$ (e.g. a word, a waveform sample)
- Previous hidden state $h_{t-1}$ (the "memory" of what happened so far)
- Compute new hidden state $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$
- Optionally emit output $y_t$

The same weights $(W_h, W_x)$ are used at every timestep — like CNNs' weight sharing across space, RNNs share weights across time.

##### Unrolled view

The same cell applied across a sequence:

```
        h₀ ──▶ [cell] ──▶ h₁ ──▶ [cell] ──▶ h₂ ──▶ [cell] ──▶ h₃
                 ▲                ▲                ▲
                 │                │                │
                x₁               x₂               x₃
                ("The")          ("cat")          ("sat")
                 │                │                │
                 ▼                ▼                ▼
                y₁               y₂               y₃
```

By the time the RNN finishes processing "The cat sat on the mat", $h_6$ contains a compressed representation of the entire sentence.

#### Forward pass math

For a vanilla RNN:

$$
h_t = \tanh(W_h \, h_{t-1} + W_x \, x_t + b_h)
$$

$$
y_t = \text{softmax}(W_y \, h_t + b_y)
$$

**Training uses Backpropagation Through Time (BPTT)** — unroll the network across timesteps, then apply standard backprop. Each weight update accumulates gradients from every timestep.

#### The vanishing (and exploding) gradient problem

The big flaw of vanilla RNNs. During BPTT, gradients must propagate backward through every timestep, each propagation involving a multiplication by the recurrent weight matrix $W_h$:

$$
\frac{\partial \mathcal{L}}{\partial h_t} \propto W_h^{T - t} \cdot (\text{later gradients})
$$

After many timesteps ($T - t$ large):

| If eigenvalues of $W_h$ are... | Then... | Result |
|---|---|---|
| **< 1** | Gradients shrink exponentially (multiplied by small numbers many times) | **Vanishing gradient** — early timesteps effectively receive zero signal; long-range dependencies cannot be learned |
| **> 1** | Gradients grow exponentially | **Exploding gradient** — training diverges or becomes unstable (often fixed with gradient clipping) |

**Consequence:** a vanilla RNN can barely remember anything more than ~10 timesteps back. For a language model trying to understand "The book *that I bought yesterday which was written by a Nobel laureate who lived in Paris in the 1920s* is..." — the long dependency between "book" and "is" gets lost.

#### LSTM — Long Short-Term Memory

Hochreiter & Schmidhuber 1997's solution. LSTM replaces the vanilla RNN cell with a more elaborate structure: a **memory cell** $c_t$ that carries information across time with *additive* updates (not multiplicative), plus three **gates** that learn to control what gets stored, forgotten, and output.

```
                    ┌───────────────────────────────────────────┐
                    │  LSTM cell                                │
                    │                                           │
    c_{t−1} ────────┤────► c_t (cell state — long-term memory)  │
                    │                                           │
                    │  ┌──────┐    ┌──────┐    ┌──────┐         │
                    │  │forget│    │input │    │output│         │
    h_{t−1} ────────┤─▶│ gate │───▶│ gate │───▶│ gate │─────────┤─▶ h_t
                    │  │ f_t  │    │ i_t  │    │ o_t  │         │
                    │  └──────┘    └──────┘    └──────┘         │
                    │                                           │
    x_t ────────────┤──────────────────▲────────────────────────┤
                    └─────────────────────────────────────────────┘
```

| Gate | Symbol | Controls |
|---|---|---|
| **Forget** | $f_t$ | How much of the previous cell state $c_{t-1}$ to keep |
| **Input** | $i_t$ | How much of the new candidate info to add to the cell |
| **Output** | $o_t$ | How much of the cell state to expose as the hidden state $h_t$ |

Each gate is a sigmoid that outputs values in $[0, 1]$ — effectively a per-dimension on/off switch learned from data. The gating structure lets the cell state $c_t$ flow across many timesteps **with minimal distortion**, dodging the vanishing-gradient problem.

#### GRU — Gated Recurrent Unit

Cho et al. 2014's simpler alternative. Fewer parameters, often comparable performance to LSTM.

| Gate | Controls |
|---|---|
| **Update** $z_t$ | How much of the previous hidden state to keep vs replace with new info |
| **Reset** $r_t$ | How much of the previous hidden state to combine with the current input |

GRU merges LSTM's separate cell state and hidden state into a single $h_t$, and combines the forget + input gates into a single update gate. Two gates instead of three → fewer parameters → faster training. In practice, LSTM and GRU perform comparably; pick by compute budget.

#### Bidirectional RNNs

Standard RNNs process left-to-right, so $h_t$ only knows about the past. But for tasks where the full sequence is available at once (e.g. sentence classification, not live transcription), you can run two RNNs simultaneously:

- Forward RNN: $\overrightarrow{h_t}$ depends on $x_1, x_2, \dots, x_t$
- Backward RNN: $\overleftarrow{h_t}$ depends on $x_T, x_{T-1}, \dots, x_t$
- Combined: $h_t = [\overrightarrow{h_t} \,;\, \overleftarrow{h_t}]$ (concatenation)

Now $h_t$ encodes context from both directions. **BiLSTMs** (bidirectional LSTMs) were the dominant NLP architecture until BERT (2018) and the transformer explosion.

#### Why transformers replaced RNNs

Three big reasons, all relevant to Module 22 (LLMs):

1. **Parallelization.** RNNs must process timesteps sequentially (each $h_t$ depends on $h_{t-1}$). Transformers process all positions in parallel via attention → much faster on GPUs.
2. **Long-range dependencies.** Transformers let any position directly attend to any other position (constant path length). RNNs must propagate information step-by-step ($O(T)$ path length) → even LSTMs struggle with very long sequences.
3. **Scaling.** Transformers scale better to billions of parameters and trillions of training tokens than RNNs ever did.

But RNNs haven't disappeared — they remain competitive for **streaming / latency-critical** applications (speech recognition where you must process audio as it arrives; real-time network traffic analysis) and are often *more* parameter-efficient on smaller tasks.

#### Red-team angles

- **RNN adversarial examples exist but are less studied than CNN ones.** Text-domain adversarial attacks against RNN sentiment classifiers exist (word swaps, synonym replacement, character perturbations) — see HotFlip, TextFooler. The gradient machinery is the same as CNNs'; the constraint is that perturbations must remain valid tokens (discrete) instead of continuous pixels.
- **Sequence poisoning / trigger injection.** Trojan attacks on text classifiers often embed a specific token sequence (e.g. a rare phrase) as a trigger. Because RNNs' hidden states are shaped by every input they see, inserting a crafted trigger at any position can steer later predictions. Direct analog to CNN trojan patches.
- **Hidden-state manipulation.** If an attacker has access to intermediate hidden states (in federated / distributed systems, or via model-extraction proxies), they can craft *prefix sequences* that push the hidden state into attacker-chosen regions before the target input arrives. The resulting classification is hijacked.
- **LSTMs in malware behavior analysis are a well-documented attack target.** Systems that model system-call sequences (sequence of `execve`, `open`, `connect`, etc.) with LSTMs can be evaded by inserting **benign filler calls** between malicious operations — diluting the attack signature in the hidden state. Research: "Malware Evasion Attacks Against API-Call-Based Detection."
- **Streaming ASR (speech recognition) evasion.** RNN-based speech recognizers can be fooled by crafted audio perturbations (Carlini & Wagner 2018's "Audio Adversarial Examples") that sound like one phrase to humans but transcribe to a different, attacker-chosen phrase.
- **Bidirectional RNNs are harder to attack than unidirectional ones.** An adversarial perturbation must succeed against both forward and backward passes simultaneously — effectively a constrained optimization with two objectives. Single-direction attacks often break under bidirectional processing.
- **RNN inference at deployment is stateful.** For models that process live streams, the hidden state from the previous request persists. An attacker who can influence prior requests can poison the hidden state going into the target request. This is a real concern for stateful conversational systems.
- **The "token-by-token generation" pattern used by LLMs at inference** is conceptually an RNN even though the model is a transformer. Each generated token becomes the next input; the process is sequential. Jailbreak techniques that work by gradually steering the conversation across multiple turns are exploiting this stateful inference dynamic — explicitly an RNN-like state accumulation in the KV cache.

**Takeaways:**
- RNN = NN with recurrent connections; each step's output depends on prior via hidden state $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$.
- Vanilla RNNs suffer vanishing/exploding gradients → can't learn long-range dependencies.
- **LSTM** (3 gates: forget/input/output + cell state) and **GRU** (2 gates: update/reset) fix this via gating + additive state updates.
- **Bidirectional RNNs** process forward + backward, concatenate hidden states — dominant NLP architecture before BERT.
- Transformers (§22) mostly replaced RNNs for text: parallelization, long-range dependencies, scaling — but RNNs persist for streaming + low-latency applications.
- Adversarial attacks on RNNs include sequence poisoning, hidden-state manipulation, benign-filler evasion of sequence-based malware detectors, and audio adversarial examples against speech recognizers.

---

### 21. Introduction to Generative AI

**Status:** - [x]  |  **Type:** Theory  |  **Completed:** 2026-04-15

A completely different framing of ML. Where discriminative models (everything through §20) learn a conditional distribution $P(y \mid x)$ — "given input $x$, what's the label $y$?" — **generative models learn the data distribution itself** $P(x)$, or in conditional form $P(x \mid c)$ (e.g. "images of cats" given $c = $ "cat"). Once you have $P(x)$, you can **sample** from it → generate new content.

This framing change matters for red teaming: **you now attack the data distribution the model represents, not just its decision boundary.**

#### Discriminative vs generative — the core distinction

| | Discriminative | Generative |
|---|---|---|
| Learns | $P(y \mid x)$ — conditional probability of label given input | $P(x)$ or $P(x \mid c)$ — distribution of the data itself |
| Output | Class labels, regression values | New data samples (images, text, audio, code) |
| Example | "Is this image a cat?" (Modules 09–10 targets) | "Generate an image of a cat" (Stable Diffusion, Midjourney) |
| Attack surface | Adversarial examples flip predictions | Attacks include: extract training data, craft prompts for harmful outputs, evade provenance detectors |

#### The pipeline — train, generate, evaluate

```
1. TRAIN       Fit the model to a large dataset. It learns the statistical
                 patterns that make the data "look right."

2. GENERATE    Start from noise or a prompt. Iteratively refine / decode
                 to produce output that matches the learned distribution.

3. EVALUATE    Measure quality (realism), diversity (coverage of modes), and
                 sometimes prompt-adherence (did the output match the request?).
```

Step 3 is harder than for discriminative models — there's no single "correct answer" to compare against. Standard metrics are partial proxies (more below).

#### The four families of generative models

| Family | Core mechanism | Famous examples | Where in path |
|---|---|---|---|
| **GANs** (Generative Adversarial Networks) | Two networks: a **generator** makes samples, a **discriminator** tries to distinguish fake from real. They train adversarially — generator tries to fool discriminator, discriminator gets better at spotting fakes | StyleGAN, BigGAN | Deepfakes; classical image synthesis |
| **VAEs** (Variational Autoencoders) | An encoder maps data to a probabilistic latent space; a decoder samples and reconstructs | β-VAE, NVAE | Controlled generation with explicit latent structure |
| **Autoregressive models** | Generate one element at a time, each conditioned on previous elements. $P(x) = \prod_t P(x_t \mid x_{<t})$ | **GPT, Llama, Claude — every modern LLM is autoregressive** | §22 LLMs → Modules 04, 05 |
| **Diffusion models** | Add noise progressively to data, then train a network to reverse the noise | Stable Diffusion, DALL-E, Imagen | §23 diffusion models |

The three attacks-of-note architectures for this path are **autoregressive (LLMs)** and **diffusion (images)**. GANs are mostly displaced in image generation by diffusion; VAEs are foundational but less deployed standalone.

#### GAN training — the adversarial game

```
             ┌─────────────┐          real / fake?
             │             │              ▲
             │  Generator  │─────┐        │
             │    G(z)     │     │        │
             └─────────────┘     │   ┌─────────────┐
                     ▲           └──▶│Discriminator│
                     │                │   D(x)      │
               random ●                └─────────────┘
               noise z ──                    ▲
                                             │
                                     real data x (training set)

   G tries to produce x = G(z) that D classifies as "real".
   D tries to correctly distinguish real x vs G(z).
   They train together. At equilibrium, G's outputs are indistinguishable from real.
```

The minimax objective:

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]
$$

GANs are notoriously unstable to train and subject to **mode collapse** (see below).

#### Key concepts

##### Latent space

A compressed representation of the data, typically a low-dimensional vector space $z \in \mathbb{R}^d$ where similar data points map to nearby latent vectors.

```
  z₂
  │   ● cats                            ● dogs
  │    ● ● ●                             ● ●
  │   ● ● ● ●                           ●  ● ●
  │       ●                              ●
  │                         ← interpolate along this line →
  │                         get hybrid cat/dog images
  │                           ● birds
  │                          ● ● ●
  │                           ●  ●
  │                            ●
  └─────────────────────────────────── z₁
```

Properties:
- **Clustering** — similar concepts cluster in latent space.
- **Interpolation** — smoothly walking between latent points produces smooth transitions in output space.
- **Semantic directions** — specific axes in latent space sometimes correspond to meaningful attributes (e.g. "age" direction in face latents, "smile" direction).

**Security relevance:** a latent space is a high-value extraction target. Recovering the latent space of a deployed generative model gives you near-total control: you can find the latent vector for any target output, and navigate to produce specific attacker-chosen content.

##### Sampling

Generating output = drawing from the learned distribution. Key parameters:

| Parameter | Effect |
|---|---|
| **Random seed** | Determines the starting point of sampling → reproducibility |
| **Temperature** (§15's $\tau$) | Controls randomness of sampling in autoregressive models |
| **Top-k / top-p / nucleus sampling** | Restrict sampling to most-likely tokens to avoid low-quality outputs |
| **Guidance scale** (diffusion) | How strongly the sample should follow the conditioning prompt |

##### Mode collapse

A GAN pathology where the generator learns to produce only a narrow slice of the true data distribution. If the training set has 1000 distinct output categories but the generator only ever outputs 3 variants, it has "collapsed" onto those 3 modes.

Intuition: if the generator finds a single output that reliably fools the discriminator, it has no incentive to try other outputs. Gradient signal disappears.

**Mode collapse as an attack:** poisoning training data (Module 06) to induce mode collapse is a documented attack — a model that collapses onto narrow outputs fails its task while looking ostensibly "trained."

##### Overfitting in generative models

Generative overfitting is worse than discriminative overfitting because it reveals **specific training data**. An overfit LLM can regurgitate training examples verbatim (names, addresses, copyrighted text, API keys). An overfit image model produces outputs nearly identical to training images.

This is a direct pathway to **training-data extraction attacks** — see red-team angles.

#### Evaluation metrics — quality and diversity

Single-number metrics for generative models are imperfect, but standard:

| Metric | What it measures | Higher is better? |
|---|---|---|
| **Inception Score (IS)** | Quality + diversity of generated images, using an Inception classifier's predictions | Higher |
| **Fréchet Inception Distance (FID)** | Distance between distributions of generated vs real images in Inception feature space | **Lower** |
| **BLEU (text)** | N-gram overlap between generated and reference text | Higher (0–1 scale) |
| **Perplexity (text)** | How "surprised" a language model is by the text | **Lower** (on held-out data) |
| **Human eval** | Subjective quality ratings | Higher |

Each has known failure modes — BLEU rewards memorization; FID is biased toward ImageNet-like content; human eval is expensive and variable. Modern evaluation typically combines several.

#### Red-team angles — generative AI has its own distinct attack surface

- **Training-data extraction is the signature privacy attack against generative models.** Carlini et al. 2021 demonstrated that GPT-2 could be prompted to regurgitate verbatim training data including names, email addresses, phone numbers, and full code snippets. An LLM's memorization is indistinguishable from retrieval when the input matches a memorized context. **This is the generative analog of membership inference (Module 11), but strictly more powerful** — you don't just learn "was this in training," you retrieve the actual training content.
- **Latent space reconstruction = near-total model control.** If an attacker can recover a generative model's latent space (via encoder extraction or repeated sampling), they can find the latent vector for any desired output. For face-generation models, this means producing arbitrary faces; for code models, specific malicious code; for image models, specific identifiable persons.
- **Deepfakes and provenance attacks.** Every generative image/video model is a deepfake engine. Detection systems (DeepFake detection CNNs) are themselves classifiers → adversarial attacks on them to make fake content undetectable are a subfield. Provenance watermarking (embedding invisible "this was generated" markers) is being adversarially attacked and patched in an ongoing arms race.
- **Prompt injection (Modules 04–05) weaponizes autoregressive generation.** LLMs generate tokens conditioned on the input prompt + prior output tokens. Adversarial prompts hijack this conditioning — the model "completes" the attacker's prompt instead of following its system instructions. The autoregressive decomposition $P(x) = \prod_t P(x_t \mid x_{<t})$ is exactly what makes this attack possible: once you've steered the early tokens, later tokens follow.
- **Mode collapse as a poisoning objective.** Attackers with training-data influence can induce mode collapse by injecting duplicate or near-duplicate examples — the model learns "this is what the defender wants" and loses diversity, making it unable to handle legitimate varied inputs.
- **Evaluation-metric gaming.** Generative models optimized to maximize BLEU or FID without caring about actual quality produce outputs that score well but are semantically broken. Attackers can craft inputs that score well on any single metric while failing human assessment. This matters when defenders use metric thresholds as guardrails ("only deploy models with BLEU > X").
- **GAN-generated adversarial examples.** Samangouei et al. 2018's **Defense-GAN** and its attack-adaptive counterparts: train a GAN on clean data, use it to generate adversarial examples that look like natural data but fool discriminative classifiers. Advanced technique that combines generative + discriminative attack surfaces.
- **Copyright and licensing exfiltration.** Models trained on copyrighted or proprietary data (code, text, images) will occasionally regurgitate verbatim. Tools like `gitcop` scan generated code for literal matches against training corpora. From an attacker's perspective, this is intelligence collection — query a commercial model to learn its training data composition.
- **Diffusion model-specific attack: prompt-attribute decoupling.** Attackers can craft prompts that superficially ask for benign content but exploit subtle latent correlations to produce harmful content (e.g. prompts that don't contain the target identity's name but reliably produce that person's face due to training-set correlations).

**Takeaways:**
- Generative models learn $P(x)$ (the data distribution) instead of $P(y \mid x)$; sample from it to generate new content.
- Four families: GANs (adversarial training), VAEs (latent-space encoder+decoder), **autoregressive** (next-element prediction — the architecture of all LLMs), **diffusion** (noise → denoise).
- Key concepts: latent space, sampling, mode collapse, overfitting.
- Evaluation: IS, FID (images), BLEU, perplexity (text), human eval — each imperfect.
- **Training-data extraction is the signature privacy attack against generative models** — more powerful than membership inference.
- Autoregressive decomposition $P(x) = \prod_t P(x_t \mid x_{<t})$ is exactly what makes **prompt injection (Modules 04–05) possible** — steer early tokens, later tokens follow.

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
