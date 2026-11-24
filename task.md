### Copy-Paste Implementation Ticket for Coding Agent

> Please sync the codebase to **`SWEEP-Net Specification Revised Certified.pdf`** with the following 6 targeted updates (keep all existing unit tests passing and add quick assertions for each):
> **1. Half-Squared Euclidean Ground Cost on $\mathbb{B}^d$ (`sinkhorn.py`, `estep.py` — Eq. 9 & Prop 8.1):**
> * Replace cosine cost `1 - z_i @ r_j` with **`C_ij = 0.5 * ||z_i - r_j||^2`** (`0.5 * torch.cdist(z, r, p=2).pow(2)` or `0.5 * (z.pow(2).sum(1, keepdim=True) + r.pow(2).sum(1) - 2 * z @ r.T).clamp(0.0, 2.0)`).
> * Guarantees $C(v, v) = 0$ even in the interior $\Vert{}v\Vert{} < 1$, stays in $[0, 2]$, and preserves `-inf` log-kernel same-trial masking.
> 
> 
> **2. Decoupled Geometric Drop Price & Log-Domain Sinkhorn (`sinkhorn.py` — Eq. 11 & Numerical Safety):**
> * Replace `dustbin_price = -eps * log1p(-rho)` with **`tau_y = tau_0 - kappa_cost * log1p(-rho_y)`** (config defaults: `tau_0: 0.35`, `kappa_cost: 0.50`).
> * Keep `sinkhorn.py` strictly in **log-domain** (log-sum-exp updates on $\log \tilde{K} = -\tilde{C}/\varepsilon$ with `-inf` support mask), since at $\rho_y \approx 0.9$, $\exp(-\tau_y/\varepsilon) \approx 10^{-7}$ can underflow in linear-domain scaling.
> 
> 
> **3. Calibrated Nullspace E-Step, Queue Buffer $\mathcal{B}_y$, and `mil_readout.py` Hookup (`queue.py`, `estep.py`, `sm_loss.py`, `mil_readout.py` — Remark 4.2, Eq. 7–8, Prop 10.1):**
> * Compute EMA empirical nullspace baseline variance per dimension:
> 
> $$\hat{\sigma}^2 = \max\!\left(\sigma_{\min}^2,\; \frac{\sum_t (1 - \alpha_t)\Vert{}h'_t - \hat{\mu}'_{s_t}\Vert{}^2}{(d - m)\sum_t (1 - \alpha_t)}\right)$$
> 
> 
> 
> and evaluate $\ell_s(h'_t) = -\Vert{}h'_t - \hat{\mu}'_s\Vert{}^2 / (2\hat{\sigma}^2)$ and $\ell_y(\tilde{z}_t) = \log\sum_{j\in Q_y} b_j \exp(-\Vert{}\tilde{z}_t - r_j\Vert{}^2 / (2\varepsilon))$.
> * Store per-window raw scores **`s_{j, y_j} = l_y(z~_j) - l_s(h'_j)`** inside `queue.py` alongside keys, labels, $\beta_j$, and `window_uid`s to form the rolling class buffer $\mathcal{B}_y$.
> * Solve $\frac{1}{\vert{}\mathcal{B}_y\vert{}}\sum_{t\in\mathcal{B}_y}\sigma((s_{t,y} + u)/T) = \rho_y$ via **25 steps of 1D bisection** to get $u_y^\star$, EMA-smooth into $\bar{u}_y$, compute $\alpha_t^{(1\text{A})} = \sigma((s_{t,y} + \bar{u}_y)/T)$, and recover **`c_y = u_bar_y - logit(rho_y)`**.
> * Export **`{c_y}`** and **`sigma_hat^2`** inside `lock_fullpass()`, and wire `mil_readout.py` so Phase 1B computes $\alpha_t^{(1\text{B})}$ using **`l_y(z~_t) + c_y`** and **`-||h'_t - mu_hat'_b||^2 / (2 * sigma_hat^2)`** (Eq. 18) with the 2-step contraction $\hat{\mu}'_b = G^2(\bar{h}'_b)$.
> 
> 
> **4. Capped Upper Bound for $\lambda_{\text{OT}}$ (`gates.py` — Eq. 15):**
> * Change the unbounded equality `lambda_OT = c * g * ||grad L_anc|| / ||grad L_em||` to the **capped upper bound**:
> 
> $$\lambda_{\text{OT}} = \min\!\left(\lambda_{\max},\; c\, g\, \frac{\Vert{}\nabla_{\tilde{z}} \mathcal{L}_{\text{anc}}\Vert{}}{\Vert{}\nabla_{\tilde{z}} \mathcal{L}_{\text{em}}\Vert{} + 10^{-8}}\right)$$
> 
> 
> 
> with default `lambda_max: 1.0` (preserves Anti-Ghost when $\Vert{}\nabla_{\tilde{z}} \mathcal{L}_{\text{em}}\Vert{} \to 0$).
> 
> 
> **5. Clamped Option-B Seeding Contraction (`sparsity.py` — Proposition 9.3):**
> * In the Option-B $\rho_y$ fixed-point seeding loop, clamp $F(u)$ at every iteration to **`[logit(rho_min), logit(1 - rho_min)]`** so $\tilde{F}: I_{\text{clamp}} \to I_{\text{clamp}}$ is a guaranteed self-map contraction even when $\Delta_t$ scores are skewed.
> 
> 
> **6. Asymmetric `FourierMixup3D` Dual Views (`dataset.py:371–372`, if not yet applied):**
> * Split into `self.query_augmentations` (includes `FourierMixup3D` with `p=0.5`) and `self.key_augmentations` (omits `FourierMixup3D`, `p=0`) so momentum keys, queue clouds $\nu_y$, and subject means $\mu_s^\star$ remain uncontaminated by donor mixing.
> 
> 
> After implementing, please run a quick synthetic forward/backward + `lock_fullpass()` $\to$ `mil_readout` smoke test confirming finite gradients, `C(v,v) == 0` for $\Vert{}v\Vert{} < 1$, calibrated `mean(alpha_y) ≈ rho_y`, and `lambda_OT <= lambda_max`.


