# MLinC — improvements

Ideas to make the project better, attention not included.
Items marked 🅰 are also things attention needs first.
Status: `[ ]` not started, `[~]` in progress, `[x]` done.

Last update: 2026-10-29

---

## 1. Measure correctly first

| Status | Add | Purpose |
|---|---|---|
| [ ] | Train/test split (see section 7) | Now the cost is measured on the training data. You do not know if the network works on new data. |
| [ ] | Loss curve: print the loss every N epochs, or write it to a CSV | You see convergence, a plateau or divergence, not only the final number. |
| [ ] | Fixed benchmark: one dataset, one seed, one number | You can say "change X improved the test cost from A to B". |

## 2. Training basics

| Status | Add | Purpose |
|---|---|---|
| [ ] | 🅰 Mini-batches: samples as columns of one matrix | One `matrix_mult` per batch, not one per sample. Attention works on whole sequences as matrices. |
| [ ] | 🅰 Adam optimizer (momentum first) | It adapts the step size for each parameter. It is the standard optimizer for transformers. |
| [ ] | Xavier/He init | Replaces `rand() % 100`. It keeps the signal at the same size through deeper networks. |
| [ ] | 🅰 Softmax + cross-entropy | Needed for classification. Attention uses softmax inside, so you must write its backward pass anyway. |
| [ ] | Gradient clipping, weight decay | Clipping stops sudden divergence. Weight decay reduces overfitting. |

## 3. Code structure (most important before attention)

| Status | Add | Purpose |
|---|---|---|
| [ ] | 🅰 Generic layer interface: each layer type has `forward()` and `backward()` | `layer_backprop` is written only for dense layers. With this interface, a new layer type is a new file, not a rewrite of backprop. |
| [ ] | 🅰 LayerNorm, residual connections, embedding layer | The other parts of a transformer block. Each is small and easy to check with the gradient check. |
| [ ] | Move the network to `Layers.c` / `Layers.h` | `tests.c` includes a header, not `MAIN.c`. `Layers.c` is empty now. |
| [ ] | Save and load weights + normalizer | Train once, use the model later. |

Choice: hand-written backward + gradient check (recommended, fits the current code)
or a small autograd (bigger project, no hand-written backward for attention).

## 4. A real dataset

1. [ ] XOR: the smallest non-linear test. It should reach a cost of about 0 in seconds.
2. [ ] MNIST (60,000 training images): needs a file loader, mini-batches, softmax and cross-entropy.
   A simple MLP usually reaches about 97–98 % test accuracy.

## 5. Speed (measure before and after each change)

1. [ ] `matrix_mult` loop order `i-k-j`, and remove the `transpose` branch from the inner loop.
2. [ ] Use `arena_reset` with one scratch arena (the function exists, not used yet).
3. [ ] OpenMP across rows, then SIMD.
4. [ ] CUDA for `matrix_mult` (CUDA 13.3 is installed).

## 6. Tooling and cleanup

- [ ] Makefile or CMake with `-Wall -Wextra -Werror`.
- [ ] `MLinC.Tests` project in Visual Studio (`tests.c` + `Arena.c` + `Matrix.c`, no `MAIN.c`).
- [ ] CI that runs `tests.exe` on each push.
- [ ] Remove dead files: `MLinC.cpp`, `simple.c`. Write `README.md`.
- [ ] Fix the unused `x` warning in `linear_prime`: add `(void)x;`.
- [ ] `arena_init_st` duplicates `arena_init`.

## 7. Train/test split — plan

1. Load all samples into `inputs[]` and `targets[]` (from code now, from a CSV later).
2. Shuffle the sample order with Fisher–Yates and a fixed seed. Swap `inputs[i]` and `targets[i]` together.
3. The first 80 % are the train set, the last 20 % are the test set. Both stay contiguous arrays.
4. Fit the normalizers on the train set only: `normalizer_fit(arena, inputs, n_train)`.
5. Train on the train set. Every N epochs, print the train cost and the test cost. Do no backprop on the test set.

CSV format (one sample per line, inputs first, then targets):

```
# inputs=2 outputs=3
10,11,0,0,0
11,12,2,2,2
```

---

## Recommended order

```
split + loss curve -> mini-batches -> layer interface -> softmax/cross-entropy
   -> Adam -> XOR, then MNIST -> LayerNorm/residual/embedding -> attention
```

## Ready for attention when

1. MNIST trains with mini-batches and Adam to about 97 % test accuracy.
2. Each layer type passes the gradient check on its own.
3. A new layer type does not change `layer_backprop`.
