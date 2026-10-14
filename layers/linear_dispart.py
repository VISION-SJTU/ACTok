import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Diagnostic utilities ─────────────────────────────────────────────────────

@torch.no_grad()
def isotropy_score(X: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Participation-ratio isotropy ∈ (0, 1] of the feature *covariance* spectrum.
    1 = all covariance eigenvalues equal (perfect isotropy); near 0 = collapsed
    onto few directions.  DINO features score high.

    Defined on covariance eigenvalues λ_i ∝ s_i² (s_i = singular values of the
    centred X), consistently with `effective_rank` and `select_topk`:
        ρ(X) = (Σ λ_i)² / (r · Σ λ_i²).
    """
    X = X - X.mean(0, keepdim=True)
    _, S, _ = torch.linalg.svd(X, full_matrices=False)
    ev = S.pow(2).clamp(min=eps)        # covariance eigenvalues (∝ s_i²)
    return ev.sum().pow(2) / (ev.pow(2).sum() * ev.numel())


@torch.no_grad()
def effective_rank(X: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Effective rank (participation-ratio form) of the feature covariance:
        r_eff = (Σ λ_i)² / Σ λ_i²,   λ_i = eigenvalues of Cov(X)  (∝ s_i²).
    Equals D for an isotropic spectrum and r for a rank-r covariance.

    NOTE: this is the participation-ratio "effective rank", NOT the operator-norm
    stable rank ‖A‖_F²/‖A‖₂².  It is the same quantity used by `select_topk`
    (computed there directly from covariance eigenvalues), so the diagnostic and
    the K-selection criterion now agree.
    """
    X = X - X.mean(0, keepdim=True)
    _, S, _ = torch.linalg.svd(X, full_matrices=False)
    ev = S.pow(2).clamp(min=eps)        # covariance eigenvalues (∝ s_i²)
    return ev.sum().pow(2) / ev.pow(2).sum()


# Backward-compatible alias (the previous misnomer); prefer `effective_rank`.
stable_rank = effective_rank


@torch.no_grad()
def off_diagonal_cov_norm(X: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of off-diagonal covariance.  0 = fully decorrelated."""
    N, D = X.shape
    X = X - X.mean(0, keepdim=True)
    C = (X.T @ X) / (N - 1)
    off = C - torch.diag(C.diag())
    return off.pow(2).sum().sqrt()


# ── Adaptive K selection ──────────────────────────────────────────────────────

def select_topk(
    S: torch.Tensor,
    energy_thresh: float = 0.90,
    min_k: int = 16,
) -> int:
    """
    Given covariance eigenvalues S (descending), return the adaptive K:

        K = clip( min(K_eff, K_energy), min_k, D )

    K_eff — effective rank: floor( (Σ λ_i)² / Σ λ_i² )
        Participation-ratio effective number of active dimensions.
        For an isotropic spectrum: K_eff = D.  For a rank-r covariance: K_eff = r.
        (Same quantity as `effective_rank`, computed here from eigenvalues.)

    K_energy — energy rank: smallest k such that top-k eigenvalues account
        for >= energy_thresh fraction of total variance.
        Ensures we capture at least energy_thresh of the signal.

    Taking the minimum is conservative: it excludes any dimension that either
    criterion considers to be noise.  This is especially important for MAE,
    where the effective rank is much smaller than D due to reconstruction bias.

    NOTE: S must be covariance eigenvalues (e.g. from svd(C)), not the singular
    values of the feature matrix — both `K_eff` and `K_energy` are defined on the
    variance (eigenvalue) spectrum.
    """
    D = S.shape[0]
    k_eff = max(1, int(S.sum().pow(2) / S.pow(2).sum().clamp(min=1e-12)))
    k_energy = int((S.cumsum(0) < energy_thresh * S.sum()).sum()) + 1
    return max(min_k, min(k_eff, k_energy, D))


# ── ZCA whitening — full rank (legacy, for reference) ────────────────────────

class BatchZCAWhitener(nn.Module):
    """
    Per-batch full-rank ZCA whitening: X_w = (X − μ) U Σ^{−½} U^T.
    Suitable when features are known to be near-isotropic (e.g. DINOv2).
    For anisotropic teachers (MAE), prefer TopKZCAWhitener.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: [N, D].  Returns whitened [N, D]."""
        X_c = X - X.mean(0, keepdim=True)
        C = X_c.T @ X_c / X.shape[0]
        U, S, _ = torch.linalg.svd(C)
        W = U @ torch.diag(1.0 / (S + self.eps).sqrt()) @ U.T
        return X_c @ W


# ── Adaptive top-K ZCA whitening ─────────────────────────────────────────────

class TopKZCAWhitener(nn.Module):
    """
    Adaptive top-K ZCA whitening — the core building block for LD distillation
    of non-isotropic teachers such as MAE.

    Algorithm per forward call
    --------------------------
    1. From Z_t (teacher, no_grad): compute C_t = Zt_c^T Zt_c / N,
       eigendecompose C_t = U S U^T.
    2. Select K = select_topk(S) adaptively (stable rank ∩ energy threshold).
    3. Project both teacher and student onto teacher's top-K eigenvectors U_k:
           Zt_proj = Zt_c @ U_k          [N, K]  (no_grad)
           Zs_proj = Zs_c @ U_k          [N, K]  (grad flows through Zs_c)
    4. Compute whitening matrix W independently for each projection using
       stop-gradient covariances:
           W_t = ZCA_matrix(Zt_proj)     [K, K]  (no_grad)
           W_s = ZCA_matrix(Zs_proj.sg)  [K, K]  (no_grad, sg = stop_gradient)
    5. Apply whitening:
           Zt_white = Zt_proj_c @ W_t    [N, K]  (no_grad)
           Zs_white = Zs_proj_c @ W_s    [N, K]  (grad through Zs_proj_c)

    The returned Zs_proj (raw, before whitening) is used for variance and
    covariance losses — both must be satisfied in teacher's K-dim subspace.
    The whitened pair (Zs_white, Zt_white) is used for the alignment loss.

    Why project into teacher's subspace?
    -------------------------------------
    It defines the *informative* alignment target.  The student is only asked to
    reproduce structure in directions where the teacher has genuine signal.
    Directions outside U_k carry noise (near-zero σ); including them would
    make the ZCA ill-conditioned and the loss meaningless.
    """

    def __init__(
        self,
        energy_thresh: float = 0.90,
        min_k: int = 16,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.energy_thresh = energy_thresh
        self.min_k = min_k
        self.eps = eps
        self.last_k: int = 0  # updated each forward; readable for logging

    @torch.no_grad()
    def _teacher_basis(self, Z_t: torch.Tensor):
        """Returns (U_k [D,K], Zt_proj [N,K], K)."""
        Zt_c = Z_t - Z_t.mean(0, keepdim=True)
        C_t = Zt_c.T @ Zt_c / Z_t.shape[0]
        U, S, _ = torch.linalg.svd(C_t)                     # S descending
        K = select_topk(S, self.energy_thresh, self.min_k)
        U_k = U[:, :K]                                       # [D, K]
        Zt_proj = Zt_c @ U_k                                 # [N, K]
        return U_k, Zt_proj, K

    @staticmethod
    @torch.no_grad()
    def _zca_W(X_proj: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute ZCA whitening matrix for X_proj [N, K].  Returns [K, K]."""
        Xc = X_proj - X_proj.mean(0, keepdim=True)
        C = Xc.T @ Xc / Xc.shape[0]
        U, S, _ = torch.linalg.svd(C)
        return U @ torch.diag(1.0 / (S + eps).sqrt()) @ U.T  # [K, K]

    def forward(self, Z_s: torch.Tensor, Z_t: torch.Tensor):
        """
        Z_s : [N, D]  student features (requires_grad=True)
        Z_t : [N, D]  teacher features (detached)

        Returns
        -------
        Zs_proj  : [N, K]  student raw projection — use for var/cov losses
        Zs_white : [N, K]  student whitened — use for alignment
        Zt_white : [N, K]  teacher whitened — use for alignment
        K        : int     selected subspace dimension (stored in self.last_k)
        """
        U_k, Zt_proj, K = self._teacher_basis(Z_t)
        self.last_k = K

        # Project student — U_k is a no_grad constant, grad flows through Zs_c
        Zs_c = Z_s - Z_s.detach().mean(0, keepdim=True)
        Zs_proj = Zs_c @ U_k                                 # [N, K]

        # Whitening matrices (stop-gradient on covariance estimation)
        W_t = self._zca_W(Zt_proj, self.eps)                 # [K, K], no_grad
        W_s = self._zca_W(Zs_proj.detach(), self.eps)        # [K, K], no_grad

        Zt_c_k = Zt_proj - Zt_proj.mean(0, keepdim=True)
        Zs_c_k = Zs_proj - Zs_proj.detach().mean(0, keepdim=True)

        Zt_white = Zt_c_k @ W_t                              # [N, K], no_grad
        Zs_white = Zs_c_k @ W_s                              # [N, K], grad ✓

        return Zs_proj, Zs_white, Zt_white, K


# ── Linear disentanglement loss ───────────────────────────────────────────────

class LinearDispartLoss(nn.Module):
    """
    Enforce and transfer the linear disentanglement property, operating in the
    teacher's adaptively-selected top-K eigenbasis.

    Three loss components (all computed in the K-dimensional subspace), matching
    the boxed objective L_LD = λ_v L_v + λ_c L_c + λ_a L_a of the theory report:

    variance   (λ_var) — hinge that keeps every projected student dimension above
                         a target std γ, preventing dimensional collapse:
                            L_v = (1/K) Σ_j max(0, γ − sqrt(Var(Zs_proj_j)+δ)).
    covariance (λ_cov) — sum of squared off-diagonal entries of the student
                         projection covariance, enforcing decorrelation:
                            L_c = (1/K) Σ_{i≠j} Ĉ_ij².
    alignment  (λ_aln) — cosine distance between independently ZCA-whitened
                         student and teacher projections.  Whitening equalises
                         gradient magnitude across all K directions so the student
                         cannot ignore low-variance (but informative) teacher
                         dimensions.

    The variance and covariance terms operate on the *raw* projection Zs_proj
    (before whitening); together they drive Cov(Zs_proj) → σ²I as a structural
    property of the backbone, while the alignment term supplies the semantic
    target in the already-disentangled basis.  Earlier versions kept only the
    alignment term; the variance/covariance terms are restored here because plain
    whitened alignment does not by itself rule out degenerate (collapsed or
    correlated) student solutions — the student self-whitening masks them.

    K selection
    -----------
    K is chosen adaptively per batch by TopKZCAWhitener via select_topk().
    After each forward(), self.last_k carries the selected K for logging
    (it is NOT included in the returned loss dict so the caller can safely
    sum all dict values for the total loss).

    Default weights follow VICReg: λ_var = λ_aln = 25, λ_cov = 1.

    Typical K values
    ----------------
    DINOv2  (near-isotropic, D=768) : K ≈ 650–730
    MAE-B   (anisotropic,   D=768)  : K ≈ 150–350
    MAE-L   (anisotropic,   D=1024) : K ≈ 200–450
    """

    def __init__(
        self,
        lambda_align: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        gamma: float = 1.0,
        var_eps: float = 1e-4,
        energy_thresh: float = 0.90,
        min_k: int = 16,
        whitening_eps: float = 1e-3,
    ):
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.var_eps = var_eps
        self.whitener = TopKZCAWhitener(
            energy_thresh=energy_thresh,
            min_k=min_k,
            eps=whitening_eps,
        )

    @property
    def last_k(self) -> int:
        """Selected K from the most recent forward pass.  Use for logging."""
        return self.whitener.last_k

    def forward(self, Z_s: torch.Tensor, Z_t: torch.Tensor) -> dict:
        """
        Z_s : [N, D]  student features projected to teacher dim (with grad)
        Z_t : [N, D]  teacher features (detach() applied internally)

        Returns a dict with three scalar losses (ld_align, ld_var, ld_cov);
        self.last_k is updated as a side-effect for logging.
        """
        Zs_proj, Zs_white, Zt_white, K = self.whitener(Z_s, Z_t.detach())

        # ── Alignment in the whitened top-K space ────────────────────────────
        Zs_n = F.normalize(Zs_white, dim=-1)
        Zt_n = F.normalize(Zt_white, dim=-1)
        align_loss = 2.0 - 2.0 * (Zs_n * Zt_n).sum(dim=-1).mean()

        # ── Variance hinge on the raw student projection (anti-collapse) ─────
        std = torch.sqrt(Zs_proj.var(dim=0, unbiased=True) + self.var_eps)  # [K]
        var_loss = F.relu(self.gamma - std).mean()

        # ── Covariance (off-diagonal) penalty — decorrelation ────────────────
        N = Zs_proj.shape[0]
        Zs_c = Zs_proj - Zs_proj.mean(0, keepdim=True)
        cov = (Zs_c.T @ Zs_c) / (N - 1)                                     # [K,K]
        off = cov - torch.diag(cov.diag())
        cov_loss = off.pow(2).sum() / K

        return {
            "ld_align": self.lambda_align * align_loss,
            "ld_var": self.lambda_var * var_loss,
            "ld_cov": self.lambda_cov * cov_loss,
        }
