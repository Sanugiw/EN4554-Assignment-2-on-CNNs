# EN4554 Assignment 2: CNNs on CIFAR-10 (PyTorch)

**Name:** `Wickramasinghe S.D.`
**Index number:** `220700T`
**Course:** EN4554, Assignment 2 (CNNs), set by Dr. Ranga Rodrigo

All work is in a single notebook, `220700T_a02.ipynb`. 

## Contents

| Task | What it covers |
|------|----------------|
| **1. Backpropagation** | Hand-written Linear, ReLU, Sigmoid and softmax cross-entropy with explicit forward/backward (no autograd), assembled into an MLP. Gradient checks against autograd and centred finite differences. Training on flattened CIFAR-10. Per-layer gradient norms of a 10-layer MLP (sigmoid vs ReLU). |
| **2. Optimizers** | `torch.optim.Optimizer` subclasses for SGD with momentum and Adam (with bias correction), verified against `torch.optim`. Trajectories on f(x, y) = 0.5(x² + 50y²). Learning-rate sensitivity of SGD, SGD+momentum and Adam on a small CNN. |
| **3. Regularization and residual connections** | Baseline, weight decay, dropout, flip+crop, cutout, mixup and all combined, on 5k training images (cutout and mixup implemented from scratch). Plain vs residual CIFAR networks of depth 8, 20 and 56 (6n+2). |
| **4. Best CNN (optional)** | Residual CNN with at most 500k parameters and 20 epochs, tuned on the validation split, with an ablation and a single test-set evaluation. |

## Setup used

- Dataset: CIFAR-10 via `torchvision.datasets.CIFAR10`, split into 45k train / 5k validation (fixed seed), per-channel normalization with training statistics.
- The test set is used **once**, in Task 4.
- Seeds are fixed with `seed_all(0)`.
- Framework: PyTorch. Designed to run on a Google Colab T4 GPU.

## How to run

**On Colab (recommended)**

1. Upload `your_index_a02.ipynb` at colab.research.google.com.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run all cells from top to bottom. CIFAR-10 downloads automatically (about 170 MB).

**Locally**

```bash
pip install torch torchvision numpy pandas matplotlib jupyter
jupyter notebook your_index_a02.ipynb
```

Running everything takes a few hours on a T4. Task 3(b) and Task 4 are the longest. A GPU is strongly recommended; on CPU only Tasks 1 and 2(a)/(b) are practical.

## Repository layout

```
.
├── your_index_a02.ipynb   # the notebook (code, plots, interpretations)
├── your_index_a02.pdf     # exported report for grading
└── README.md
```

## Results

Results, plots, and interpretations are in the notebook and the PDF report.

| Item | Result |
|------|--------|
| Task 1(c): MLP validation accuracy | `<fill in>` |
| Task 2(c): best optimizer / learning rate | `<fill in>` |
| Task 3(b): residual-56 validation accuracy | `<fill in>` |
| Task 4: final model parameters/test accuracy | `<fill in>` |

## Use of AI tools

Claude (Anthropic) was used to help draft the code, as permitted by the assignment. I have read, run and understood all of it, and the results and interpretations in the report are from my own runs.
