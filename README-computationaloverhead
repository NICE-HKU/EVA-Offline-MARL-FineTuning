```latex
% =========================
% README: Computational Overhead of EVAFormer (EVA-MARL)
% =========================

\section*{Computational Overhead of EVAFormer (EVA-MARL)}
\label{sec:readme_overhead}

This document provides a detailed and reproducible description of the computational overhead introduced by the evolutionary optimization (EO) in \textbf{EVAFormer}, as well as its relationship to the standard Transformer token mixer (MHSA) used by \textbf{MADT}. We emphasize a key design choice in EVAFormer: \textbf{EO is enabled only during training and is disabled at deployment/inference}. As a result, EVAFormer shifts computation to training while preserving an efficient inference path.

\subsection*{1. Code Reference and Reproducibility}
\textbf{Repository entry:}
\begin{itemize}
  \item Main implementation file: \url{https://github.com/NICE-HKU/EVA-Offline-MARL-FineTuning/blob/main/sc2/models/gpt_model.py}
\end{itemize}

\textbf{Key verification point (Training vs. Inference behavior).}
In the \texttt{forward} function of \texttt{eVAE} (approximately around lines 220--240), the code executes evolutionary optimization only when \texttt{self.training == True}. Otherwise, it falls back to a one-shot reparameterization and decoding pass. This provides a strict separation:
\begin{itemize}
  \item \textbf{Training:} EO enabled $\Rightarrow$ iterative candidate search in latent space.
  \item \textbf{Deployment/Inference:} EO disabled $\Rightarrow$ single-pass encode--reparameterize--decode.
\end{itemize}

\subsection*{2. Model Context: EVAFormer vs. MADT Token Mixer}
EVAFormer and MADT share a MetaFormer-style architecture, but differ in the \textbf{token mixer}:
\begin{itemize}
  \item \textbf{MADT:} token mixer is MHSA (\texttt{CausalSelfAttention}).
  \item \textbf{EVAFormer:} token mixer is eVAE, optionally using EO in training.
\end{itemize}

\subsubsection*{2.1 Multi-agent Decision Transformer Forward Skeleton}
In our multi-agent extension, token embeddings are constructed in triplets (\texttt{[RTG, state, action, ...]}) and passed through Transformer blocks. The token mixer is executed inside \texttt{self.blocks(x)}. A simplified excerpt (corresponding to around lines 428--492) is:
\begin{lstlisting}[language=Python]
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
    x = self.blocks(x)          # token mixer happens inside each block
    x = self.ln_f(x)
    logits = self.head(x)
    logits = logits[:, 2::3, :]
    return logits
\end{lstlisting}

\subsection*{3. Token Mixers: MHSA (MADT) vs. eVAE (EVAFormer)}

\subsubsection*{3.1 MADT Token Mixer: MHSA (\texttt{CausalSelfAttention})}
MADT uses a vanilla causal MHSA implementation. Core operations:
\begin{lstlisting}[language=Python]
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
\end{lstlisting}

\subsubsection*{3.2 EVAFormer Token Mixer: eVAE (EO only in training)}
The eVAE token mixer follows an \textbf{if/else} branch for training vs. inference:
\begin{lstlisting}[language=Python]
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
\end{lstlisting}

\subsection*{4. Complexity Variables and Notation}
We use the following notation:
\begin{mdframed}
Let $B$ denote batch size, $T$ sequence/context length, and $C$ embedding dimension (\texttt{n\_embd}).
For MHSA, we use $H$ attention heads with per-head dimension $d=C/H$.
For EVAFormer, latent size is $L=\texttt{latent\_dim}=C/2$.
Evolutionary optimization uses population size $M$ (default $20$) and generations $G$ (default $3$).
\end{mdframed}

\subsection*{5. Complexity Analysis: MHSA vs. eVAE}
\subsubsection*{5.1 MADT (MHSA) Complexity}
MHSA performs dense projections and token-pair interactions:
\begin{itemize}
  \item Q/K/V projections: $\mathcal{O}(BTC^2)$
  \item Attention score and weighted sum: $\mathcal{O}(BT^2C)$ (quadratic in $T$)
  \item Output projection: $\mathcal{O}(BTC^2)$
\end{itemize}
Thus, the overall scaling per layer is:
\begin{equation}
\label{eq:mhsa_cost_readme}
\text{Cost}_{\text{MHSA}} \approx \mathcal{O}\!\left(B\left(4TC^2 + 2T^2C\right)\right),
\end{equation}
where the bottleneck is the quadratic term $\mathcal{O}(BT^2C)$ as $T$ grows.

\subsubsection*{5.2 eVAE with EO (Training) Complexity}
When EO is enabled, the overhead comes from repeated candidate evaluation:
\begin{itemize}
  \item EO evaluates $M$ latent candidates over $G$ generations.
  \item The dominant step is \texttt{calculate\_fitness} per candidate, consisting of (i) one decoder pass and
  (ii) one fitness network pass.
\end{itemize}
Per candidate, dense computation scales approximately as:
\begin{equation}
\label{eq:per_candidate_readme}
\text{Cost}_{\text{per-candidate}} \approx \mathcal{O}(BT\cdot C^2).
\end{equation}
Since EO evaluates $M$ candidates for each of $(G{+}1)$ fitness rounds (including final selection):
\begin{equation}
\label{eq:eo_cost_readme}
\text{Cost}_{\text{EO}} \approx (G{+}1)M \cdot \mathcal{O}(BT\cdot C^2).
\end{equation}
With the default setting $M{=}20$, $G{=}3$, EO performs $M(G{+}1)=80$ candidate-evaluations per training forward, which is a non-negligible training overhead.

\subsubsection*{5.3 eVAE without EO (Deployment/Inference) Complexity}
When EO is disabled, eVAE becomes token-wise dense mixing without token-to-token interactions:
\begin{itemize}
  \item Encoder: $C\to C$ costs $(BT)C^2$
  \item Latent heads ($\mu$ and $\log\sigma^2$): $2(BT)CL$
  \item Decoder: $L\to C$ and $C\to C$ costs $(BT)(LC + C^2)$
\end{itemize}
Ignoring lower-order terms ($\mathcal{O}(BTC)$), we obtain:
\begin{equation}
\label{eq:evae_base_readme}
\text{Cost}_{\text{eVAE-base}} \approx \mathcal{O}\!\left(BT\left(C^2 + 2CL + (LC + C^2)\right)\right).
\end{equation}
With $L=C/2$, this simplifies to:
\begin{equation}
\label{eq:evae_infer_readme}
\text{Cost}_{\text{eVAE-infer}} \approx \mathcal{O}(BT\cdot C^2),
\end{equation}
highlighting that \textbf{eVAE inference avoids the $\mathcal{O}(T^2)$ attention term}.

\subsection*{6. Key Takeaway: Training vs. Deployment Trade-off}
\begin{mdframed}
\textbf{MADT (MHSA)} has essentially the same complexity in training and deployment, with a quadratic token-pair term $\mathcal{O}(BT^2C)$.
\textbf{EVAFormer (eVAE)} pays extra compute during training due to EO, but uses a cheap deployment path without EO and without $\mathcal{O}(T^2)$.
Thus, \textbf{overhead is confined to training}, while deployment remains efficient and scales linearly with $T$.
\end{mdframed}

\subsection*{7. Empirical Overhead Quantification: Two Complementary Experiments}
In MARL (SMAC), total runtime includes environment stepping, rollout collection, simulator execution, replay buffer operations, and data movement. Therefore, end-to-end time alone cannot precisely reflect token-mixer overhead. We report:
\begin{itemize}
  \item \textbf{Experiment A:} end-to-end wall-clock time per 1M environment steps (coarse but practical).
  \item \textbf{Experiment B:} isolated token-mixer forward latency (controlled and verifiable).
\end{itemize}

\subsubsection*{7.1 Experiment A: End-to-End Wall-Clock Time (Per 1M Env Steps)}
\textbf{Hardware:} NVIDIA GeForce RTX 4090 GPU + Intel Xeon Platinum 8488C CPU.  
\textbf{Protocol:} For each SMAC map, measure hours needed to complete $\sim$1M environment steps under identical training configuration, optimizer settings, and rollout pipeline for both MADT and EVAFormer.

\begin{table}[h!]
\centering
\caption{Measured wall-clock time (hours) per 1M environment steps on RTX~4090 + Xeon~8488C. Values are end-to-end measurements including environment stepping, rollout collection, and model computation.}
\label{tab:1m_steps_readme}
\begin{tabular}{l|cc}
\hline
Map (SMAC) & MADT (MHSA) & EVAFormer (eVAE) \\
\hline
\texttt{3m} & 5.9 h & 5.8 h \\
\texttt{8m} & 7.1 h & 6.6 h \\
\hline
\texttt{3s5z vs 3s6z} & 8.9 h & 9.4 h \\
\texttt{So Many Baneling} & 9.6 h & 9.8 h \\
\texttt{8m9m} & 10.3 h & 10.1 h \\
\texttt{corridor} & 11.1 h & 10.4 h \\
\hline
\end{tabular}
\end{table}

\textbf{Observations.}
\begin{enumerate}
  \item Absolute training time is largely determined by map difficulty and environment dynamics, not only the token mixer.
  \item Although EVAFormer adds compute during training (EO), the end-to-end difference remains moderate:
        for most maps, the increase is below $\sim$0.5 hour per 1M steps.
  \item This suggests EO overhead is amortized by dominant environment interaction costs in MARL.
\end{enumerate}

\subsubsection*{7.2 Experiment B: Isolated Token-Mixer Forward Latency}
To directly quantify token-mixer overhead, we measure the forward latency of the token-mixer module only, excluding SC2 engine and other pipeline components.

\textbf{Protocol:}
\begin{itemize}
  \item Insert timing hooks immediately before/after the token-mixer \texttt{forward}.
  \item Report mean latency over 1,000 consecutive forward steps.
  \item Evaluate on six SMAC maps: \textbf{MMM}, \textbf{3s5z vs 3s62}, \textbf{So Many Baneling}, \textbf{8m9m}, \textbf{3s vs 4z}, \textbf{Bane vs Bane}.
\end{itemize}

\begin{table}[h!]
\centering
\caption{Measured forward latency (ms) of token-mixer only (excluding SC2 engine). Each value is the average over 1,000 forward steps. ``eVAE (train)'' enables EO (\texttt{self.training==True}); ``eVAE (eval)'' disables EO (\texttt{self.training==False}).}
\label{tab:latency_readme}
\scalebox{0.92}{
\begin{tabular}{lccc}
\hline
Map (SMAC) & MHSA & eVAE (eval, no EO) & eVAE (train, with EO) \\
\hline
MMM & 0.226 & 0.212 & 28.230 \\
3s5z vs 3s62 & 0.214 & 0.197 & 27.540 \\
So Many Baneling & 0.209 & 0.201 & 29.120 \\
8m9m & 0.221 & 0.208 & 28.960 \\
3s vs 4z & 0.218 & 0.204 & 27.980 \\
Bane vs Bane & 0.213 & 0.199 & 28.610 \\
\hline
\textbf{Average} & 0.217 & 0.206 & 28.4 \\
\hline
\end{tabular}}
\end{table}

\textbf{Interpretation.}
\begin{enumerate}
  \item \textbf{MHSA latency:} $\sim$0.209--0.226 ms.
  \item \textbf{eVAE inference latency (EO off):} $\sim$0.197--0.212 ms, comparable to (sometimes slightly lower than) MHSA.
  \item \textbf{eVAE training latency (EO on):} $\sim$27.5--29.1 ms, reflecting repeated candidate evaluations.
\end{enumerate}

Using the averaged values:
\begin{equation}
\frac{\text{eVAE(train)}}{\text{MHSA}} \approx \frac{28.4}{0.217} \approx 130.8\times,\qquad
\frac{\text{eVAE(eval)}}{\text{MHSA}} \approx \frac{0.206}{0.217} \approx 0.95.
\end{equation}
Therefore, EO introduces a two-orders-of-magnitude overhead \emph{inside the token mixer} during training, but \textbf{does not affect deployment latency}.

\subsection*{8. Practical Mitigation Strategies}
The EO overhead is \textbf{explainable and controllable}. In practice:
\begin{itemize}
  \item \textbf{Reduce population/generations:} overhead scales nearly linearly with $(G{+}1)M$.
  \item \textbf{Early stopping:} stop EO when elite fitness improvement becomes marginal.
  \item \textbf{Lower EO frequency:} apply EO only every $k$ updates instead of every update.
  \item \textbf{Layer-selective EO:} enable EO only for selected layers, while others use base eVAE.
\end{itemize}
These strategies reduce training overhead without changing the inference path, because EO is never executed at deployment.

\subsection*{9. Summary}
\begin{mdframed}
\textbf{Summary.} EO introduces non-trivial training-time overhead by performing $(G{+}1)M$ repeated decode/fitness evaluations, but this overhead is confined to training and adjustable via explicit hyperparameters.
End-to-end MARL training time remains comparable to MADT under realistic SMAC pipelines, and deployment/inference remains efficient:
eVAE inference avoids $\mathcal{O}(T^2)$ and achieves latency comparable to MHSA.
\end{mdframed}
```
