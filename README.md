# ADOM: ADMM-Based Optimization Model for Stripe Noise Removal in Remote Sensing Images

A MATLAB implementation of the **ADOM** algorithm for removing stripe noise from remote sensing images (RSI). This project extends the original paper to support **vertical**, **horizontal**, **diagonal**, and **bidirectional** stripe noise removal.

> Based on: *"ADOM: ADMM-Based Optimization Model for Stripe Noise Removal in Remote Sensing Image"*  
> Namwon Kim, Seong-Soo Han, Chang-Sung Jeong — IEEE Access, Vol. 11, 2023  
> DOI: [10.1109/ACCESS.2023.3319268](https://doi.org/10.1109/ACCESS.2023.3319268)

---

## Overview

Remote sensing images often suffer from stripe noise caused by physical limitations in sensor systems (e.g., detector gaps, gain/offset differences). ADOM addresses this by formulating stripe noise removal as an optimization problem solved via the **Alternating Direction Method of Multipliers (ADMM)**.

The model decomposes the observed image `O = D + S`, where `D` is the clean image and `S` is the stripe noise component. It then recovers `D = O − S` through iterative optimization.

### Key Features

- **Weight-Based Detection Strategy** — Dynamically adjusts norm weights (`wn`) and group norm weights (`wg`) using momentum coefficient and residual parameter to accurately detect stripe noise while preserving image details (edges, textures).
- **Evidence-Based Starting Point Control** — Uses Nesterov's method with a threshold parameter `p` to find a better starting point and accelerate convergence.
- **Momentum-Based Step-Size Control** — Accelerates optimization using a momentum coefficient `α` and a damping coefficient `δ` for stability.
- **FFT-Based Subproblem Solver** — Efficient closed-form solution using Fast Fourier Transform for the S-subproblem.
- **Multi-Direction Support** — Handles vertical, horizontal, diagonal, and combined (bidirectional/all-direction) stripe noise.

---

## Repository Structure

```
├── ADOM_vert.mlx          # Vertical stripe noise removal
├── ADOM_hori.mlx          # Horizontal stripe noise removal
├── ADOM_diag.mlx          # Diagonal stripe noise removal
├── ADOM_2D.mlx            # Bidirectional (vertical + horizontal) removal
├── ADOM_stripe.mlx        # Add synthetic stripe noise to an image
├── ADOM_striperemoval.mlx # Load a pre-striped image and run ADOM
├── All-destripe.mlx       # All-direction (vertical + horizontal + diagonal) removal
└── README.md
```

---

## Algorithm Overview

Each iteration of ADOM performs 4 steps:

```
1. Weight Control       → Update wn (norm weight) and wg (group norm weight)
2. Starting Point Control → Update momentum coefficient α and damping coefficient d
3. Step-Size Control    → Update S and Lagrange multipliers using α and d
4. ADMM Subproblem Solving → Solve subproblems A, B, C, S via soft-thresholding + FFT
```

The optimization function minimized is:

```
argmin_S { ||∇_along S||_1  +  λ1 ||∇_perp (O−S)||_{wn,1}  +  λ2 ||S||_{wg,2,1} }
```

---

## Requirements

- MATLAB R2019b or later (tested with MATLAB Live Script `.mlx`)
- Image Processing Toolbox (for `imread`, `imshow`, `rgb2gray`, `im2double`)
- No additional third-party toolboxes required

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Prepare your image

Update the image path in the script you want to run:

```matlab
img = imread("path/to/your/Image.jpg");
```

### 3. Run a script

Open MATLAB and run any of the `.mlx` files directly. Each script includes:
- Synthetic stripe noise generation
- ADOM destriping
- Side-by-side display of Original / Striped / Destriped images

---

## Usage Examples

### Vertical Stripe Removal (`ADOM_vert.mlx`)

```matlab
lambda1 = 0.05;  % Regularization for horizontal gradient
lambda2 = 0.1;   % Regularization for group sparsity
rho1 = 1; rho2 = 1; rho3 = 1;  % ADMM penalty parameters
p = 10;          % Threshold for acceleration
tol = 1e-4;      % Convergence tolerance
max_iter = 200;  % Maximum iterations

destriped = ADOM(O_striped, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter);
```

### Horizontal Stripe Removal (`ADOM_hori.mlx`)

```matlab
destriped = ADOM_Horizontal(O_striped, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter);
```

### Bidirectional Stripe Removal (`ADOM_2D.mlx`)

```matlab
% Run vertical first, then horizontal on the result
destriped_vert = ADOM_Vertical(O_striped, A1, B1, rho1, rho2, rho3, p, tol, max_iter);
destriped      = ADOM_Horizontal(destriped_vert, A2, B2, rho1, rho2, rho3, p, tol, max_iter);
```

### All-Direction Removal (`All-destripe.mlx`)

```matlab
% Unified ADOM function with direction parameter
destriped_v = ADOM(O_striped, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter, 'vertical');
destriped_h = ADOM(destriped_v, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter, 'horizontal');
destriped   = ADOM(destriped_h, lambda1, lambda2, rho1, rho2, rho3, p, tol, max_iter, 'diagonal');
```

---

## Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `lambda1` | Weight for horizontal/perpendicular gradient term | `0.01` – `0.1` |
| `lambda2` | Weight for group sparsity term | `0.01` – `0.1` |
| `rho1` | ADMM penalty for subproblem A | `1` |
| `rho2` | ADMM penalty for subproblem B | `1` |
| `rho3` | ADMM penalty for subproblem C | `1` |
| `p` | Threshold for switching acceleration mode | `10` |
| `tol` | Convergence tolerance | `1e-4` |
| `max_iter` | Maximum number of iterations | `200` |

**Tuning tips:**
- For **non-periodical/broken stripes**: try `lambda1 = lambda2 = 0.1`
- For **periodical/multiplicative stripes**: try `lambda1 = lambda2 = 0.01`
- Increasing `lambda2` strengthens group sparsity enforcement (useful for dense stripes)

---

## Noise Cases Supported

| Case | Stripe Type | Description |
|------|-------------|-------------|
| 1 | Periodical | Same intensity, 40% of columns |
| 2 | Non-Periodical | Varying intensity, 40% of columns |
| 3 | Broken | Random length stripes, 20% of columns |
| 4 | Multiplicative | Gain (0.8–1.2) + offset noise, 60% of columns |
| 5 | Mixed | Non-periodical + broken + wide stripes |

---

## Computational Complexity

Given an image of size `m × n`:

| Subproblem | Complexity per iteration |
|------------|--------------------------|
| A, B | O(mn) — pixel-wise soft-thresholding |
| C | O(mn) — group-wise soft-thresholding |
| S | O(mn log mn) — FFT-based solver |
| **Total** | **O(k · mn log mn)** for k iterations |

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{kim2023adom,
  title   = {ADOM: ADMM-Based Optimization Model for Stripe Noise Removal in Remote Sensing Image},
  author  = {Kim, Namwon and Han, Seong-Soo and Jeong, Chang-Sung},
  journal = {IEEE Access},
  volume  = {11},
  pages   = {106587--106606},
  year    = {2023},
  doi     = {10.1109/ACCESS.2023.3319268}
}
```

---



## Acknowledgements

- Original algorithm by Namwon Kim, Seong-Soo Han, and Chang-Sung Jeong (Korea University / Kangwon National University)
- This MATLAB implementation extends the original vertical-stripe model to support horizontal, diagonal, and multi-direction stripe removal
