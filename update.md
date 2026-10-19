Three corrections locked in, and I'll stop violating them:

1. **SNN is downstream, not a design constraint.** You distill the *whole* AtrousMobileNet later for the energy argument; the adapter is free to be whatever is most accurate. I'm dropping every SNN‑compatibility restriction I previously imposed (the "one nonlinearity," the "expose g\_t for KD," etc.). Gone.
2. **No contrastive anything in the adapter.** No prototype loss, no center loss, no MoCo, no SupCon. The adapter is **pure supervised MIL** on bag labels. I keep re‑adding prototype terms; I won't again.
3. **The split is exactly:** adapter trains on the 15 source subjects = **675 bags** (trials), tests on subject 1 = **45 bags**. Inner model‑selection is carved *inside* the 15 source subjects (leave‑one‑source‑subject‑out), never on subject 1.

The reason your formulas kept "cutting off" is almost certainly that I buried them inside markdown tables and code fences, which many viewers truncate. So from here: **every equation lives in its own display block, nothing mathematical goes in a table or a code fence.** Tables below hold only plain text/numbers. That's why this is long — it's the complete, airtight sheet you asked for.

---

## A. Notation and index sets

- Emotion classes: $C = 5$. Index $c \in \{1,2,3,4,5\}$ with the mapping $1=\text{Disgust},\ 2=\text{Fear},\ 3=\text{Sad},\ 4=\text{Neutral},\ 5=\text{Happy}$ (matching your log order).
- A bag (one trial / one video) is indexed $b$. The training set is $\mathcal{B}_{\text{train}}$ with $|\mathcal{B}_{\text{train}}| = 675$.
- Bag $b$ contains $T_b$ one‑second windows, indexed $t \in \{1,\dots,T_b\}$.
- Frozen encoder output per window: $h_t \in \mathbb{R}^{d}$.
- Adapter bottleneck: $r_g$ (e.g. $32$ or $64$).
- Subject‑subspace rank (optional cleaning): $r \in \{0,8,16,32\}$. $r=0$ means no cleaning.
- One‑hot bag label: $y \in \{0,1\}^{C}$ with $y_c = 1$ iff the bag's emotion is $c$, else $0$. (Same $y$ for every window in the bag, since one trial = one emotion.)
- Stability constant: $\varepsilon = 10^{-8}$.
- Identity matrix of size $d$: $I_d$.
- All sums over $t$ run $1\dots T_b$; all sums over $c,k$ run $1\dots C$ unless stated.

---

## B. Parameter inventory (plain text only)

| Name | Shape (rows x cols) | Role |
|---|---|---|
| U\_s | d x r | fixed subject basis, orthonormal columns; only if r>0 |
| W1 | r\_g x d | SwiGLU gate projection |
| W2 | r\_g x d | SwiGLU value projection |
| q | r\_g | attention query vector |
| w\_c | r\_g | per-class window score vector (C of them) |
| b\_lse\_c | scalar | per-class window score bias (C) |
| v\_c | r\_g | per-class average-branch vector (C) |
| b\_avg\_c | scalar | per-class average-branch bias (C) |
| u\_c | r\_g | per-class attention-branch vector (C) |
| b\_attn\_c | scalar | per-class attention-branch bias (C) |
| w\_avg\_c | scalar | raw mix weight for average branch (C) |
| w\_lse\_c | scalar | raw mix weight for LSE branch (C) |
| w\_attn\_c | scalar | raw mix weight for attention branch (C) |
| tau\_a\_raw | scalar | raw attention temperature |
| tau\_c\_raw | scalar | raw per-class LSE temperature (C) |

Regularized set (weight decay applies only to these): $\Theta_{\text{reg}} = \{W1, W2, q, \{w_c\}, \{v_c\}, \{u_c\}\}$. Biases and temperatures are **not** weight‑decayed.

---

## C. Forward pass (one fixed bag $b$)

**Step 1 — optional subject‑subspace removal.** Define the projector onto the subject subspace $P = U_s U_s^{\top} \in \mathbb{R}^{d\times d}$. The cleaned window embedding is

$$h'_t = (I_d - U_s U_s^{\top})\, h_t$$

Component form, for $i = 1,\dots,d$:

$$[h'_t]_i = [h_t]_i - \sum_{k=1}^{r} [U_s]_{i k} \left( \sum_{j=1}^{d} [U_s]_{j k}\, [h_t]_j \right)$$

If $r=0$, set $h'_t = h_t$. Note $U_s$ is **fixed** (computed once from source‑train), so this is a constant linear map, not a learned fight against the encoder.

**Step 2 — SwiGLU core.** Define the sigmoid and SiLU explicitly:

$$\mathrm{sig}(x) = \frac{1}{1 + e^{-x}}$$

$$\mathrm{SiLU}(x) = x \cdot \mathrm{sig}(x) = \frac{x}{1 + e^{-x}}$$

Then the gate stream, value stream, and shared instance feature, for $m = 1,\dots,r_g$:

$$[\mathrm{gate}_t]_m = \sum_{j=1}^{d} [W1]_{m j}\, [h'_t]_j$$

$$[\mathrm{value}_t]_m = \sum_{j=1}^{d} [W2]_{m j}\, [h'_t]_j$$

$$[g_t]_m = [\mathrm{gate}_t]_m \cdot \mathrm{SiLU}\!\left([\mathrm{value}_t]_m\right)$$

$g_t \in \mathbb{R}^{r_g}$ is the single shared nonlinear per‑window feature that every downstream head reads.

**Step 3 — attention score and weights.** Define $\mathrm{softplus}(x) = \log(1 + e^{x})$ and $\mathrm{clamp}(x, lo, hi) = \min(hi, \max(lo, x))$. The attention temperature is

$$\tau_a = \mathrm{clamp}(\tau_{a,\text{raw}},\ 0.5,\ 5.0)$$

The raw score is linear in $g_t$:

$$a_t = \frac{1}{\sqrt{r_g}} \sum_{m=1}^{r_g} [q]_m\, [g_t]_m$$

The attention weights (softmax over the time axis of this bag):

$$\alpha_t = \frac{\exp\!\left(a_t / \tau_a\right)}{\sum_{j=1}^{T_b} \exp\!\left(a_j / \tau_a\right)}$$

**Step 4 — three pooling branches.**

Average branch bag vector and logit:

$$z_{\text{avg}} = \frac{1}{T_b} \sum_{t=1}^{T_b} g_t$$

$$\ell^{\text{avg}}_c = \sum_{m=1}^{r_g} [v_c]_m\, [z_{\text{avg}}]_m + b^{\text{avg}}_c$$

Localized (LogSumExp) branch per‑window evidence and bag logit:

$$s_{t,c} = \sum_{m=1}^{r_g} [w_c]_m\, [g_t]_m + b^{\text{lse}}_c$$

$$\tau_c = \mathrm{clamp}(\tau_{c,\text{raw}},\ 1.0,\ 20.0)$$

$$\ell^{\text{lse}}_c = \frac{1}{\tau_c} \log\!\left( \sum_{t=1}^{T_b} \exp\!\left( \tau_c \, s_{t,c} \right) \right)$$

Attention branch bag vector and logit:

$$z_{\text{attn}} = \sum_{t=1}^{T_b} \alpha_t\, g_t$$

$$\ell^{\text{attn}}_c = \sum_{m=1}^{r_g} [u_c]_m\, [z_{\text{attn}}]_m + b^{\text{attn}}_c$$

**Step 5 — positive per‑class mixture.** Convert raw mix params to strictly positive weights:

$$W^{\text{avg}}_c = \mathrm{softplus}(w^{\text{avg}}_c) = \log\!\left(1 + e^{w^{\text{avg}}_c}\right)$$

$$W^{\text{lse}}_c = \mathrm{softplus}(w^{\text{lse}}_c) = \log\!\left(1 + e^{w^{\text{lse}}_c}\right)$$

$$W^{\text{attn}}_c = \mathrm{softplus}(w^{\text{attn}}_c) = \log\!\left(1 + e^{w^{\text{attn}}_c}\right)$$

Final class logit:

$$\ell_c = W^{\text{avg}}_c\, \ell^{\text{avg}}_c + W^{\text{lse}}_c\, \ell^{\text{lse}}_c + W^{\text{attn}}_c\, \ell^{\text{attn}}_c$$

**Step 6 — predicted probabilities (softmax over classes):**

$$p_c = \frac{\exp(\ell_c)}{\sum_{k=1}^{C} \exp(\ell_k)}$$

Predicted class at test time: $\hat{c} = \arg\max_{c} p_c$.

---

## D. Loss (one fixed bag $b$, then averaged)

Cross‑entropy (only the true class contributes because of $y_c$):

$$L_{\text{CE}} = -\sum_{c=1}^{C} y_c \, \log(p_c + \varepsilon)$$

Focal loss, with focusing exponent $\gamma = 2$ and **no** class‑frequency weight (your classes are count‑balanced, so frequency weighting would do nothing):

$$L_{\text{focal}} = -\sum_{c=1}^{C} y_c \, (1 - p_c)^{\gamma} \, \log(p_c + \varepsilon)$$

Attention entropy. First the entropy of the attention distribution over this bag's windows:

$$H(\alpha) = -\sum_{t=1}^{T_b} \alpha_t \, \log(\alpha_t + \varepsilon)$$

We want attention **broad** (high entropy), i.e. we want to *maximize* $H$. Because the optimizer *minimizes* the total loss, the entropy term must enter with a negative sign. Define

$$L_{\text{ent}} = -\,H(\alpha) = \sum_{t=1}^{T_b} \alpha_t \, \log(\alpha_t + \varepsilon)$$

so that adding $+\lambda_e L_{\text{ent}}$ (with $\lambda_e > 0$) to the minimized objective **maximizes** $H$. (This fixes a sign ambiguity in my earlier note: a $+H$ term would have sharpened attention, the opposite of the intent.)

Weight decay over the regularized set, with $\|M\|_F^2 = \sum_{i,j} M_{ij}^2$:

$$L_{\text{wd}} = \sum_{M \in \Theta_{\text{reg}}} \|M\|_F^2$$

Per‑bag loss:

$$L_b = L_{\text{CE}} + \lambda_f\, L_{\text{focal}} + \lambda_e\, L_{\text{ent}} + \lambda_w\, L_{\text{wd}}$$

Total training objective, averaged over the 675 training bags:

$$L_{\text{total}} = \frac{1}{|\mathcal{B}_{\text{train}}|} \sum_{b \in \mathcal{B}_{\text{train}}} L_b$$

Hyperparameters (fixed, not tuned on subject 1):

$$\lambda_f = 0.5, \quad \lambda_e = 0.01, \quad \lambda_w = 10^{-3}, \quad \gamma = 2$$

**Deliberately absent** (so there is no ambiguity): no contrastive loss, no prototype/center loss, no MoCo queue, no GRL, no VIB, no inverse‑attention subject head, no subject adversary, no SNN‑shaping term. The adapter is supervised MIL only; subject handling lives solely in the optional fixed projection of Step 1.

---

## E. Initialization (exact values)

Attention temperature raw: $\tau_{a,\text{raw}} = 1.0$ (effective $1.0$ after clamp).

Per‑class LSE temperature raw (higher = sharper = more spike‑like; fear/disgust start sharper):

$$\tau_{1,\text{raw}} = 5.0\ (\text{Disgust}),\ \tau_{2,\text{raw}} = 5.0\ (\text{Fear}),\ \tau_{3,\text{raw}} = 2.0,\ \tau_{4,\text{raw}} = 2.0,\ \tau_{5,\text{raw}} = 2.0$$

Mix weights are initialized by their **effective** targets $E$ via the inverse softplus $w = \log(e^{E} - 1)$. The effective targets per class:

| Class | W\_avg effective | W\_lse effective | W\_attn effective |
|---|---|---|---|
| 1 Disgust | 0.3 | 1.5 | 0.5 |
| 2 Fear | 0.3 | 1.5 | 0.5 |
| 3 Sad | 1.0 | 0.3 | 0.5 |
| 4 Neutral | 1.0 | 0.3 | 0.5 |
| 5 Happy | 1.0 | 0.3 | 0.5 |

For reference, the inverse‑softplus values you actually store in $w^{\cdot}_c$: effective $1.0 \to 0.541$; effective $0.5 \to -0.433$; effective $0.3 \to -1.050$; effective $1.5 \to 1.247$. All other parameters (W1, W2, q, w\_c, v\_c, u\_c) use small zero‑centered init (e.g. Xavier); all biases to $0$.

---

## F. Training and evaluation protocol

- **Outer test:** subject 1, 45 bags. Never used for fitting or selection.
- **Adapter train:** the other 15 subjects, 675 bags.
- **Inner selection split:** from those 15 source subjects, hold one source subject out (45 bags) as inner‑val, train the adapter on the remaining 14 (630 bags). Rotate to get a source‑subject‑LOSO inner‑val macro F1; **select the checkpoint / hyper‑setting by inner‑val macro F1**, with a tie‑break toward higher inner‑val fear+disgust F1.
- **Fixed preprocessing stats** (scaler, $U_s$, any mean/std) are fit on the inner‑train subjects only, never on inner‑val or subject 1.
- **Reporting on subject 1:** per‑class precision/recall/F1 and macro F1 over the 45 bags; plus the confusion matrix. No loss‑based early stopping on subject 1.
- **Cheating‑gap monitor (inner split only):** per epoch and per class, $G_c = \text{F1}^{\text{inner-train}}_c - \text{F1}^{\text{inner-val}}_c$. If $G_1$ (Disgust) or $G_2$ (Fear) exceeds $0.15$ while inner‑val F1 for that class is flat or falling, stop that run and increase regularization / reduce capacity — do not push further.

---

## G. Mathematical guarantees and test statistics

These are the "prove it works" statements, each fully derived or fully defined so they are checkable.

**Guarantee 1 — the projection nulls any linear subject readout that lies in the removed subspace.** Suppose a linear subject probe has, for some subject, a weight vector $a \in \mathbb{R}^d$ that lies in $\mathrm{span}(U_s)$, i.e. $a = U_s \beta$ for some $\beta \in \mathbb{R}^r$. Its readout on the *cleaned* embedding is exactly zero:

$$a^{\top} h'_t = \beta^{\top} U_s^{\top} (I_d - U_s U_s^{\top}) h_t = \beta^{\top}\left(U_s^{\top} - U_s^{\top} U_s U_s^{\top}\right) h_t$$

Because $U_s$ has orthonormal columns, $U_s^{\top} U_s = I_r$, so the bracket is $U_s^{\top} - I_r U_s^{\top} = 0$, hence

$$a^{\top} h'_t = 0 \quad \text{for all } t.$$

Interpretation: every linear subject direction you remove via $U_s$ becomes *exactly* unreadable after projection. This is why "layered" (additive) subject structure is the right thing to project, and it is a hard guarantee, not a hope. The residual subject leakage you still measure (your 64%) lives *outside* $\mathrm{span}(U_s)$ and is what the rank sweep $r \in \{0,8,16,32\}$ targets.

**Guarantee 2 — LogSumExp is a controllable soft‑maximum (localization knob).** Let $M_c = \max_{t} s_{t,c}$. Then

$$e^{\tau_c M_c} \le \sum_{t=1}^{T_b} e^{\tau_c s_{t,c}} \le T_b\, e^{\tau_c M_c}$$

Taking $\log$ and dividing by $\tau_c > 0$:

$$M_c \le \ell^{\text{lse}}_c \le M_c + \frac{\log T_b}{\tau_c}$$

So as $\tau_c \to \infty$, $\ell^{\text{lse}}_c \to \max_t s_{t,c}$ (pure spike detector); as $\tau_c \to 0^+$, $\ell^{\text{lse}}_c \to \frac{1}{T_b}\sum_t s_{t,c}$ (pure average). This is the exact mathematical reason fear/disgust use larger $\tau_c$ init and why the LSE branch, not the attention branch, is the localized‑evidence path.

**Test statistic 1 — nonlinear‑disgust gap (the "encoder is not stupid" proof).** On the *same* inner split, let $R^{\text{lin}}_1$ be Disgust recall of a linear probe on frozen bag features, and $R^{\text{nl}}_1$ be Disgust recall of this adapter (Step T1 below, attention‑only). Define

$$\Delta_{\text{NL}} = R^{\text{nl}}_1 - R^{\text{lin}}_1$$

Decision rule: if $\Delta_{\text{NL}} \ge 0.05$, the Disgust manifold is nonlinearly separable but linearly inseparable, which is exactly the regime your UMAP arc predicts and exactly the regime where the SwiGLU adapter is justified over a linear head. This is the single most important number for the paper's architecture claim.

**Test statistic 2 — localization gain.** Let $F^{\text{attn}}_2$ be Fear F1 with the LSE branch disabled ($W^{\text{lse}}_2 = 0$) and $F^{\text{lse}}_2$ with it enabled. Define

$$\Delta_{\text{loc}} = F^{\text{lse}}_2 - F^{\text{attn}}_2$$

Decision rule: if $\Delta_{\text{loc}} \ge 0.03$, Fear evidence is temporally sparse and the soft‑max‑over‑time branch is doing real work (Guarantee 2 in action).

**Test statistic 3 — projection efficacy.** Let $A_{\text{sub}}(r)$ be the linear subject‑probe accuracy (your proxy‑A linear SVC) measured on $z_{\text{attn}}$ computed from cleaned embeddings at rank $r$, and $M_{\text{emo}}(r)$ the inner‑val emotion macro F1 at rank $r$. A rank $r^\star$ is acceptable iff

$$A_{\text{sub}}(0) - A_{\text{sub}}(r^\star) \ge 0.15 \quad \text{and} \quad M_{\text{emo}}(0) - M_{\text{emo}}(r^\star) \le 0.02$$

i.e. subject leakage drops by at least 15 points while emotion performance drops by at most 2 points. Pick the largest such $r^\star$.

---

## H. Test ladder (run in order; each adds exactly one thing)

Every row uses the frozen EA encoder and the protocol of Section F. "ON" lists which equations are active; the rest are zeroed as stated.

| Step | Active branches / terms | Pass condition (plain text) |
|---|---|---|
| T0 | nearest-centroid on z\_avg, no training | happy/sad/neutral above chance on subject 1 (floor + sanity) |
| T1 | avg + attn branches; loss = CE + wd only (set lambda\_f=lambda\_e=0, force W\_lse=0) | Delta\_NL >= 0.05 on Disgust (Eq test-stat 1) |
| T2 | add LSE branch (W\_lse free); loss = CE + wd | Delta\_loc >= 0.03 on Fear (Eq test-stat 2) |
| T3 | add focal (lambda\_f=0.5) | macro F1 up AND cheating gap G\_1,G\_2 not growing |
| T4 | add entropy reg (lambda\_e=0.01) | max\_t alpha\_t on inner-val below 0.5; val F1 not down |
| T5 | toggle projection rank r in {0,8,16,32} | pick r\* by Eq test-stat 3 |

T1 is your headline. If T1 passes, the architecture is vindicated (the linear probe was the wrong ruler, exactly as your Disgust arc implied) and you proceed to T2–T5 and then to full 16‑fold LOSO. If T1 fails, widen $r_g$ or add one residual SwiGLU block on $g_t$ and re‑run T1 — but I expect it to pass.

---

That is the complete, self‑contained specification: every symbol defined, every operation written in closed form, signs explicit, no truncation‑prone formatting, no contrastive term, no SNN constraint, and the split exactly as you run it (675 train / 45 test, inner‑val from the 15). Implement C→D verbatim, run T0→T5, and the two test statistics $\Delta_{\text{NL}}$ and $\Delta_{\text{loc}}$ are the mathematical evidence that the adapter is reading what the linear probe couldn't. Build T1 first and send me the Disgust number — that one number decides the whole story. 🧠