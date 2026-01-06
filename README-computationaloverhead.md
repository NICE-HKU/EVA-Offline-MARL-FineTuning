---

# Computational Overhead of EVAFormer (EVA-MARL)

This README documents the computational overhead introduced by the **evolutionary optimization (EO)** used in **EVAFormer**, and provides **transparent complexity analysis + reproducible timing benchmarks** comparing EVAFormer (eVAE token mixer) with MADT (MHSA token mixer).
**Key point:** EO is **enabled only during training** (`self.training == True`) and is **disabled at deployment/inference** (`self.training == False`). Therefore, the overhead is **confined to training**, while inference remains efficient.

---

## 1) Code Reference (Reproducibility)

* Main implementation file:
  `sc2/models/gpt_model.py`
  [https://github.com/NICE-HKU/EVA-Offline-MARL-FineTuning/blob/main/sc2/models/gpt_model.py](https://github.com/NICE-HKU/EVA-Offline-MARL-FineTuning/blob/main/sc2/models/gpt_model.py)

### Where to verify “EO only in training”

In `eVAE.forward` (approximately around lines **220–240** in the file), the code executes EO only when `self.training == True`. Otherwise it falls back to **one-shot** reparameterization + decoding (no EO).
This yields a strict separation:

* **Training path**: `self.training == True` → EO enabled (iterative candidate search in latent space)
* **Inference/deployment path**: `self.training == False` → EO disabled (single-pass encode → reparameterize → decode)

---

## 2) Model Context: EVAFormer vs. MADT Token Mixer

EVAFormer and MADT share the same MetaFormer-style backbone, but differ in the **token mixer**:

* **MADT**: token mixer is **MHSA** (`CausalSelfAttention`)
* **EVAFormer**: token mixer is **eVAE** (with **optional EO** only in training)

### 2.1 Multi-agent Decision Transformer Forward Skeleton (where token mixer is called)

Your multi-agent DT constructs triplet tokens `[RTG, state, action, ...]` and then runs Transformer blocks; the token mixer happens inside `self.blocks(x)`.

```python
def forward(self, states, pre_actions, rtgs=None, timesteps=None):
    state_embeddings = self.state_encoder(states.reshape(-1, self.state_size).type(torch.float32).contiguous())
    state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)
    rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
    action_embeddings = self.action_embeddings(pre_actions.type(torch.long).squeeze(-1))

    batch_size, seq_len = states.shape[:2]
    token_embeddings = torch.zeros((batch_size, seq_len * 3, self.config.n_embd),
                                  dtype=torch.float32, device=state_embeddings.device)
    token_embeddings[:, ::3, :] = rtg_embeddings
    token_embeddings[:, 1::3, :] = state_embeddings
    token_embeddings[:, 2::3, :] = action_embeddings

    all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
    global_pos_emb = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))
    global_pos_emb = torch.repeat_interleave(global_pos_emb, 3, dim=1)
    context_pos_emb = self.pos_emb[:, :seq_len * 3, :]
    position_embeddings = global_pos_emb + context_pos_emb

    x = self.drop(token_embeddings + position_embeddings)
    x = self.blocks(x)   # token mixer runs inside each block here
    x = self.ln_f(x)
    logits = self.head(x)
    logits = logits[:, 2::3, :]
    return logits
```

---

## 3) Token Mixers: MHSA (MADT) vs. eVAE (EVAFormer)

### 3.1 MADT Token Mixer: MHSA (`CausalSelfAttention`)

MADT uses a standard causal MHSA. The following operations create the quadratic cost in sequence length `T`:

```python
def forward(self, x, layer_past=None):
    B, T, C = x.size()
    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_drop(att)

    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_drop(self.proj(y))
    return y
```

### 3.2 EVAFormer Token Mixer: eVAE (EO only in training)

eVAE explicitly branches by training/eval mode:

```python
def forward(self, x, layer_past=None):
    B, T, C = x.size()
    residual = x
    x = self.layer_norm(x)
    x_flat = x.view(B * T, C)
    mu, logvar = self.encode(x_flat)
    if self.training:
        z_evolved = self.evolutionary_optimization(x_flat, mu, logvar)
        reconstructed = self.decode(z_evolved)
    else:
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
    reconstructed = reconstructed.view(B, T, C)
    output = residual + self.residual_weight * reconstructed
    output = self.resid_drop(output)
    return output
```

---

## 4) Notation for Complexity Discussion

* `B`: batch size
* `T`: sequence/context length
* `C`: embedding dimension (`n_embd`)
* MHSA: `H` heads, per-head dim `d = C/H`
* eVAE: latent dim `L = latent_dim = C/2`
* EO hyperparameters:

  * `M`: population size (`population_size`, default **20**)
  * `G`: number of generations (`generations`, default **3**)

---

## 5) Complexity Analysis (What Causes the Overhead)

### 5.1 MADT / MHSA complexity (quadratic in `T`)

Dominant operations per layer:

1. **Q/K/V projections**: `3 * (BT) * C * C = O(BTC^2)`
2. **Attention score**: `O(BT^2C)` (token-pair interactions)
3. **Attention-weighted sum**: `O(BT^2C)`
4. **Output projection**: `O(BTC^2)`

So a compact scaling is:

* **Cost(MHSA)** ≈ `O(B(4TC^2 + 2T^2C))`
* Key bottleneck: **quadratic term** `O(BT^2C)` as `T` grows.

---

### 5.2 eVAE complexity with EO (training-time overhead)

When `self.training == True`, eVAE calls `evolutionary_optimization`, which:

1. Initializes a population of `M` latent candidates (dim `L`)
2. Iterates for `G` generations
3. In each generation, computes fitness for **all** candidates (dominant), then selection/crossover/mutation
4. After `G` generations, computes fitness once more to select the best candidate

The dominant part is `calculate_fitness()`. For each candidate `z`, it includes:

* (i) **one decoder pass** (dense, token-wise)
* (ii) **one fitness network pass** on concatenated `(z, x)` (MLP)

Per candidate, compute scales approximately as:

* **Cost(per-candidate)** ≈ `O(BT * C^2)` (leading order; with `L=C/2` this is consistent)

Since fitness is evaluated for `M` candidates across `(G+1)` rounds:

* **Cost(EO)** ≈ `(G+1) * M * O(BT * C^2)`

With default settings `M=20`, `G=3`:

* Total candidate evaluations per training forward: `M*(G+1) = 20*4 = 80`
* This explains why training is **measurably heavier** than MHSA at the token-mixer level.

---

### 5.3 eVAE complexity without EO (deployment / inference path)

When `self.training == False`, eVAE does:

* flatten: `x ∈ R^{B×T×C} → (BT)×C`
* encoder: `C → C`
* latent heads: `C → L` for `mu` and `logvar`
* reparameterize once
* decoder: `L → C → C`

All operations are **token-wise dense transformations**; there is **no** `T×T` attention matrix and thus **no** `O(T^2)` term.

Ignoring lower-order terms, with `L=C/2`:

* **Cost(eVAE-infer)** ≈ `O(BT * C^2)`

**Conclusion:** When EO is disabled, EVAFormer inference cost grows approximately **linearly in `T`**, without the quadratic attention bottleneck.

---

## 6) Summary: Training vs. Deployment Trade-off

* **MADT (MHSA)**: same forward complexity in training and deployment; includes quadratic term `O(BT^2C)` due to token-pair interactions.
* **EVAFormer (eVAE)**: intentionally shifts complexity:

  * pays higher cost **during training** (EO iterative candidate evaluations)
  * uses cheaper **deployment path** (EO disabled + no `T×T` attention)

So the overhead is **confined to training**, while inference remains efficient and can become advantageous as context length grows.

---

## 7) Empirical Overhead Quantification (Two Complementary Experiments)

In SMAC MARL, total runtime includes SC2 engine stepping, rollout collection, replay buffer, and data movement. Therefore:

* End-to-end training time is a **coarse but practical** reference.
* Isolated token-mixer latency provides a **controlled** measure of model-side overhead.

We report both.

---

### Experiment A — End-to-End Wall-Clock Time (Per 1M Environment Steps)

**Hardware:** NVIDIA GeForce RTX 4090 + Intel Xeon Platinum 8488C
**Protocol:** identical training configuration, optimizer, rollout pipeline; measure wall-clock time to complete ~1M environment steps.

| Map (SMAC)         | MADT (MHSA) | EVAFormer (eVAE) |
| ------------------ | ----------: | ---------------: |
| `3m`               |       5.9 h |            5.8 h |
| `8m`               |       7.1 h |            6.6 h |
| `3s5z vs 3s6z`     |       8.9 h |            9.4 h |
| `So Many Baneling` |       9.6 h |            9.8 h |
| `8m9m`             |      10.3 h |           10.1 h |
| `corridor`         |      11.1 h |           10.4 h |

**Observations:**

1. Absolute time is largely determined by map difficulty and environment dynamics rather than token mixer choice alone.
2. Although EVAFormer introduces extra computation in training (EO), the end-to-end differences are moderate: **typically within ~0.5 hour** for most tested maps.
3. This indicates EO overhead is largely **amortized** by environment interaction costs at the million-step scale.

**Core notice:** the purpose of this table is to alleviate concerns that EVAFormer would cause a substantial degradation of overall training efficiency under realistic MARL pipelines.

---

### Experiment B — Isolated Token-Mixer Forward Latency (Excluding SC2 Engine)

To isolate model overhead, we measure only the token-mixer `forward` latency (excluding SC2 engine and other pipeline costs).

**Protocol:**

* insert timing hooks immediately before/after token-mixer `forward`
* report average over **1,000 forward steps**
* evaluate on six maps: **MMM**, **3s5z vs 3s62**, **So Many Baneling**, **8m9m**, **3s vs 4z**, **Bane vs Bane**

| Map (SMAC)       | MHSA (`CausalSelfAttention`) | eVAE (eval, no EO) | eVAE (train, with EO) |
| ---------------- | ---------------------------: | -----------------: | --------------------: |
| MMM              |                     0.226 ms |           0.212 ms |             28.230 ms |
| 3s5z vs 3s62     |                     0.214 ms |           0.197 ms |             27.540 ms |
| So Many Baneling |                     0.209 ms |           0.201 ms |             29.120 ms |
| 8m9m             |                     0.221 ms |           0.208 ms |             28.960 ms |
| 3s vs 4z         |                     0.218 ms |           0.204 ms |             27.980 ms |
| Bane vs Bane     |                     0.213 ms |           0.199 ms |             28.610 ms |
| **Average**      |                 **0.217 ms** |       **0.206 ms** |           **28.4 ms** |

**Key findings:**

1. **MHSA:** ~0.209–0.226 ms
2. **eVAE inference (EO off):** ~0.197–0.212 ms, comparable to (sometimes slightly lower than) MHSA
3. **eVAE training (EO on):** ~27.5–29.1 ms, reflecting repeated candidate evaluation

Using averaged values:

* Training overhead ratio: `28.4 / 0.217 ≈ 130.8×`
* Inference ratio: `0.206 / 0.217 ≈ 0.95` (≈95%)

**Interpretation:**

* The overhead is clearly attributable to EO (repeated decode + fitness-net evaluations).
* At deployment, eVAE avoids constructing the `T×T` attention matrix and remains in the same latency range as MHSA.

---

## 8) Mitigation Strategies (Practical Knobs)

The training-time EO overhead is **controllable** and can be reduced without changing the deployment path:

1. **Reduce `M` or `G`**
   Overhead scales nearly linearly with `(G+1) * M`.
2. **Early stopping**
   Stop EO when elite fitness improvement becomes marginal.
3. **Lower EO frequency**
   Enable EO only every `k` updates (not every gradient step).
4. **Layer-selective EO**
   Apply EO only to certain layers; other layers use base eVAE.
5. **Vectorization / mixed precision (implementation-level)**
   Batch candidate evaluations on GPU and use AMP to reduce wall-clock cost (inference path unchanged).

---

## 9) Final Takeaway

* EO introduces **non-trivial training-time overhead** in the token mixer due to `(G+1)M` candidate evaluations (default **80** evaluations per forward).
* However, **this overhead does not carry over to deployment**: EO is disabled in inference, and eVAE inference cost is **linear in `T`** without the `O(T^2)` attention term.
* Empirically, EVAFormer shows:

  * **Comparable end-to-end** wall-clock time per 1M steps under realistic SMAC pipelines.
  * **Comparable inference latency** to MHSA when EO is off, while EO-on training latency is higher as expected.

---
