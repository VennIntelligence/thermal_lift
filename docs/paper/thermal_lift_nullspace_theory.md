# Observation-side anchoring and null-space drift: appendix-ready theory

This note is written for direct insertion into `docs/paper/supp/A_theory.md`.  It proves statements about the fixed **linear** observation operator only.  It does **not** prove that a neural network or SGD trajectory must drift; the theorem is conditional: once a reconstruction change has a component in the exact or approximate null space of the observation operator, observation-domain losses have little or no information with which to oppose that component.

---

## Main-text theorem box

> **Theorem box: observation-domain anchoring is blind to near-null drift.**  For fixed known shifts $t_k$, let $A=[DBHS_{t_1};\ldots;DBHS_{t_K}]$ be the stacked burst operator and let $\mathcal N_\varepsilon=\operatorname{span}\{v_i:\sigma_i\le \varepsilon\}$ be its right-singular $\varepsilon$-null space.  If $\delta\in\ker(A)$, every observation-domain loss $\ell(Ax,y)$ is exactly invariant under $x\mapsto x+\delta$.  If $\|\delta\|=1$ and $\delta\in\mathcal N_\varepsilon$, every $L_\ell$-Lipschitz observation loss changes by at most $L_\ell\varepsilon$, and the squared data term has curvature at most $\varepsilon^2$ along $\delta$.  A bounded high-pass/full-band observation weight $W$ can only replace $\varepsilon$ by $\|W\|\varepsilon$ and may zero additional bands; it cannot invert information suppressed by $A$.  With noise standard deviation $\eta$, any unbiased linear estimate of the coordinate $\langle v_i,x^\star\rangle$ has variance at least $\eta^2/\sigma_i^2$, so near-null components must be controlled by prior/input/selection rather than by another finite observation-side anchor.

---

## A.2.0 Setup and notation

We work in finite-dimensional real Euclidean spaces.  The high-resolution unknown is $x\in\mathbb R^N$ on the delivered $2\times$ grid.  The $k$-th low-resolution observation is $y_k\in\mathbb R^M$, and the single-frame operator is

$$
A_k = D B H S_{t_k}.
$$

The shift $t_k$ is treated as fixed and known.  In the experiment these shifts are obtained from the stage motion after applying the measured stage-to-pixel rotation $\theta=47.6^\circ$.  Once $t_k$ is fixed, the resampling operator $S_{t_k}$, the Gaussian PSF convolution $H$, the detector box integration $B$, and the downsampler $D$ are linear maps.  The stacked burst operator is

$$
A = \begin{bmatrix} A_1 \\ \vdots \\ A_K \end{bmatrix}\in\mathbb R^{KM\times N},\qquad K=248.
$$

Unless otherwise stated, $\|\cdot\|$ denotes the Euclidean norm.  Let a full right-singular basis of $A$ be denoted by $\{v_i\}_{i=1}^N$, with singular values

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_N\ge 0.
$$

Equivalently, after completing the left singular vectors arbitrarily on the zero part,

$$
A v_i = \sigma_i u_i,\qquad \|u_i\|=1\quad\text{when }\sigma_i>0.
$$

The exact null space and the $\varepsilon$-null space are

$$
\ker(A)=\operatorname{span}\{v_i:\sigma_i=0\},\qquad
\mathcal N_\varepsilon=\operatorname{span}\{v_i:\sigma_i\le \varepsilon\}.
$$

An observation-domain loss has the form

$$
\mathcal L(x)=\ell(Ax,y),
$$

where $y=(y_1,\ldots,y_K)\in\mathbb R^{KM}$.  For the sensitivity result below we assume that $\ell$ is $L_\ell$-Lipschitz in its first argument on the region being considered, i.e.

$$
|\ell(z_1,y)-\ell(z_2,y)|\le L_\ell\|z_1-z_2\|.
$$

This assumption is global for genuinely Lipschitz losses.  For the squared loss $\ell(z,y)=\tfrac12\|z-y\|^2$, it is local rather than global: on any bounded residual set $\|z-y\|\le R$, one may take $L_\ell=R$.  Proposition 3 treats the squared loss exactly and does not require this Lipschitz reduction.

A band-limited anchor is represented by a bounded linear map $W$ acting in the observation domain, for example a high-pass or full-band residual filter.  If $W$ is normalized so that $\|W\|\le 1$, it is non-expansive.  If it is not normalized, all bounds below acquire the finite factor $\|W\|$.

---

## A.2.1 Stacked null spaces and the role of phase diversity

\begin{lemma}[Stacked operator and joint null spaces]
Let $A=[A_1;\ldots;A_K]$.  Then

$$
\ker(A)=\bigcap_{k=1}^K \ker(A_k),
$$

and for every perturbation $\delta\in\mathbb R^N$,

$$
\|A\delta\|^2=\sum_{k=1}^K \|A_k\delta\|^2.
$$

Consequently, a unit vector is jointly $\varepsilon$-unobservable for the stacked burst exactly when its root-sum-square response over all frames is at most $\varepsilon$.
\end{lemma}

\begin{proof}
By definition,

$$
A\delta = (A_1\delta,\ldots,A_K\delta).
$$

Thus $A\delta=0$ if and only if $A_k\delta=0$ for every $k$, proving the intersection identity.  The norm identity follows from the Euclidean norm on the stacked observation vector:

$$
\|A\delta\|^2 = \sum_{k=1}^K \|A_k\delta\|^2.
$$
\end{proof}

This lemma is the exact multi-frame statement.  Additional sub-pixel phases can only add constraints: they replace a single null condition $A_1\delta=0$ by the intersection of $K$ conditions.  Therefore phase diversity can shrink the exact null space and can increase singular values in well-covered alias blocks.  It cannot, however, create information in a Fourier component that the common pre-sampling transfer function $BH$ has already annihilated, and it cannot make a component with very small joint transfer statistically reliable.

Under the standard periodic, shift-invariant idealization, this statement has a concrete Fourier form.  Let $m(\xi)$ be the scalar transfer multiplier of $BH$ at high-resolution frequency $\xi$.  For factor-$2$ decimation in two dimensions, each low-resolution base frequency $\omega$ receives contributions from the four high-resolution aliases

$$
\mathcal A(\omega)=\{\omega+\pi q:q\in\{0,1\}^2\},
$$

up to the chosen Fourier normalization.  A Fourier perturbation with coefficients $c_\xi$ on this alias set produces, in frame $k$, the low-resolution coefficient

$$
\widehat{(A_k\delta)}(\omega)
=\sum_{\xi\in\mathcal A(\omega)} m(\xi)\,e^{-i\xi\cdot t_k}\,c_\xi.
$$

Thus the stacked alias block is the $K\times |\mathcal A(\omega)|$ matrix

$$
\Phi_\omega(k,\xi)=m(\xi)e^{-i\xi\cdot t_k}.
$$

Exact alias-cancellation directions in that frequency cell are precisely

$$
\ker(\Phi_\omega).
$$

More phases can increase $\operatorname{rank}(\Phi_\omega)$ and hence reduce $\dim\ker(\Phi_\omega)$.  If the sampled phases are generic and all four multipliers $m(\xi)$ are nonzero, downsampling-only alias nulls in that cell may disappear.  If some $m(\xi)=0$, the corresponding frequency component is a common transfer zero and remains exactly invisible for every $K$.  If $m(\xi)$ is merely small, the corresponding singular directions lie in an $\varepsilon$-null space rather than in the exact null space.

This is the precise sense in which the $248$ sub-pixel phases shrink, but need not eliminate, the blind subspace: $D$ creates alias-cancellation directions; phase diversity removes only those not cancelled for all observed phases; $H$ and $B$ create common attenuation or transfer-zero directions that phase diversity cannot undo.

---

## A.2.3 Exact null-space blind spot

\begin{proposition}[P1: exact null-space invariance]
Let $\mathcal L(x)=\ell(Ax,y)$ for any function $\ell$ for which the displayed quantities are defined.  If $\delta\in\ker(A)$, then for every $x\in\mathbb R^N$,

$$
\mathcal L(x+\delta)=\mathcal L(x).
$$

If, in addition, $\ell$ is differentiable in its first argument, then

$$
\langle \nabla \mathcal L(x),\delta\rangle=0.
$$
\end{proposition}

\begin{proof}
Since $\delta\in\ker(A)$, $A\delta=0$.  Hence

$$
\mathcal L(x+\delta)=\ell(Ax+A\delta,y)=\ell(Ax,y)=\mathcal L(x).
$$

If $\ell$ is differentiable in its first argument, the chain rule gives

$$
\nabla\mathcal L(x)=A^\top \nabla_1\ell(Ax,y),
$$

where $\nabla_1$ denotes the gradient with respect to the first argument.  Therefore

$$
\langle \nabla\mathcal L(x),\delta\rangle
=\langle A^\top\nabla_1\ell(Ax,y),\delta\rangle
=\langle \nabla_1\ell(Ax,y),A\delta\rangle
=0.
$$
\end{proof}

This proposition is exact but weak.  It covers only directions with $\sigma_i=0$.  The observed drift in a finite, noisy thermal system need not be exactly in $\ker(A)$; it is enough for it to concentrate in directions whose singular values are small compared with the noise and the scale of the competing prior or network gradient.  That is the role of the next two propositions.

---

## A.2.4 Approximate null spaces: sensitivity and restoring force

\begin{proposition}[P2: $\varepsilon$-null sensitivity bound]
Let $A=U\Sigma V^\top$ be a full singular-value decomposition with right singular vectors $v_i$.  For every $\delta\in\mathbb R^N$,

$$
\|A\delta\|^2=\sum_{i=1}^N \sigma_i^2\langle v_i,\delta\rangle^2.
$$

If $\ell$ is $L_\ell$-Lipschitz in its first argument on the segment between $Ax$ and $A(x+\delta)$, then

$$
|\mathcal L(x+\delta)-\mathcal L(x)|\le L_\ell\|A\delta\|.
$$

In particular, if $\delta\in\mathcal N_\varepsilon$ and $\|\delta\|=1$, then

$$
\|A\delta\|\le \varepsilon,
\qquad
|\mathcal L(x+\delta)-\mathcal L(x)|\le L_\ell\varepsilon.
$$

For a band-weighted observation loss

$$
\mathcal L_W(x)=\ell(WAx,Wy),
$$

with bounded linear $W$, the corresponding bound is

$$
|\mathcal L_W(x+\delta)-\mathcal L_W(x)|\le L_\ell\|W\|\,\|A\delta\|.
$$

Thus, if $\|W\|\le 1$, observation-side band weighting is no more sensitive to any $A$-near-null direction than the unweighted observation loss.  If $A\delta$ lies in a band suppressed by $W$, it is strictly less sensitive and may be exactly zero.
\end{proposition}

\begin{proof}
Expand $\delta$ in the orthonormal right-singular basis:

$$
\delta=\sum_{i=1}^N \langle v_i,\delta\rangle v_i.
$$

Since $Av_i=\sigma_i u_i$ and the nonzero $u_i$ are orthonormal,

$$
A\delta=\sum_{i=1}^N \sigma_i\langle v_i,\delta\rangle u_i,
$$

with zero terms included for $\sigma_i=0$.  Therefore

$$
\|A\delta\|^2=\sum_{i=1}^N \sigma_i^2\langle v_i,\delta\rangle^2.
$$

The Lipschitz bound gives

$$
|\mathcal L(x+\delta)-\mathcal L(x)|
=|\ell(Ax+A\delta,y)-\ell(Ax,y)|
\le L_\ell\|A\delta\|.
$$

If $\delta\in\mathcal N_\varepsilon$ and $\|\delta\|=1$, then all singular vectors present in $\delta$ have $\sigma_i\le\varepsilon$, so

$$
\|A\delta\|^2
=\sum_{\sigma_i\le\varepsilon}\sigma_i^2\langle v_i,\delta\rangle^2
\le \varepsilon^2\sum_{\sigma_i\le\varepsilon}\langle v_i,\delta\rangle^2
=\varepsilon^2.
$$

For the weighted loss,

$$
\|WA\delta\|\le \|W\|\,\|A\delta\|,
$$

and the same Lipschitz argument applies.
\end{proof}

A related ordered-singular-value statement is also true, but it must be stated carefully.  If $s_j(T)$ denotes the $j$-th singular value of $T$ in nonincreasing order, then by the min-max principle,

$$
s_j(WA)\le \|W\|s_j(A).
$$

When $\|W\|\le 1$, the ordered singular values of $WA$ are bounded above by those of $A$.  However, the right singular vectors of $WA$ need not equal the right singular vectors of $A$.  The directionwise statement used above, $\|WAv_i\|\le\|W\|\sigma_i$, is the robust fact needed for the drift argument.

### Physical meaning of $\varepsilon$: MTF, Nyquist period, and noise

In the periodic shift-invariant approximation, the convolutional part $BH$ acts diagonally on Fourier modes.  For a spatial frequency $f$, the Gaussian PSF multiplier is

$$
\operatorname{MTF}_{\rm PSF}(f)=\exp(-2\pi^2\sigma^2 f^2),
$$

and the ideal detector aperture contributes

$$
\operatorname{MTF}_{\rm det}(f)=|\operatorname{sinc}(Wf)|,
\qquad W=10\,\mu{\rm m}.
$$

Thus the per-frame transfer magnitude is the product of these factors, up to the normalization convention for the Fourier transform and for whether the burst loss is summed over frames or averaged over frames.  If the burst loss is averaged over frames, one may equivalently analyze $K^{-1/2}A$; this rescales singular values but leaves $\ker(A)$ and $\mathcal N_\varepsilon$ unchanged after rescaling $\varepsilon$.

On the delivered $2\times$ grid, the sample spacing is $5\,\mu{\rm m}$ and the Nyquist period is $10\,\mu{\rm m}$, i.e. $f=1.0$ cycle per detector pixel.  The given Gaussian-only MTF values at this frequency are:

| PSF $\sigma$ in LR px | Gaussian MTF at $2\times$ Nyquist | Gaussian MTF squared | Gaussian-only $\operatorname{SNR}_{\rm eff}$ for $\Delta T=0.70^\circ{\rm C}$ and $\eta=0.0724^\circ{\rm C}$ |
|---:|---:|---:|---:|
| $0.20$ | $0.454$ | $0.206$ | $4.39$ |
| $0.35$ | $0.089$ | $7.92\times10^{-3}$ | $0.86$ |
| $0.50$ | $0.007$ | $4.90\times10^{-5}$ | $0.068$ |

The third column is the loss-energy scale because squared residual losses see approximately $\operatorname{MTF}^2$ in a diagonal Fourier model.  The last column uses only the stated nominal edge contrast and noise floor:

$$
\operatorname{SNR}_{\rm eff}(f)=\frac{\Delta T\operatorname{MTF}(f)}{\eta}.
$$

These numbers are upper bounds on the full optical-detector response at the exact $2\times$ Nyquist frequency, because the detector box also multiplies by $|\operatorname{sinc}(Wf)|$.  With the ideal $W=10\,\mu{\rm m}$ box and $f=1$ cycle per detector pixel, this sinc factor is zero.  In an implemented finite-grid model, boundary handling, finite support, and interpolation may turn an exact zero into a very small transfer, but either case places the corresponding $10\,\mu{\rm m}$-period structure in the exact null space or in a very small $\varepsilon$-null space.  Phase diversity can improve alias coverage for frequencies that survive $BH$; it cannot restore a frequency whose common optical-detector MTF is zero or below the noise floor.

\begin{proposition}[P3: vanishing restoring force for squared observation losses]
Let

$$
\mathcal L(x)=\frac12\|Ax-y\|^2.
$$

Then

$$
\nabla\mathcal L(x)=A^\top(Ax-y).
$$

For every right singular vector $v_i$ with $\sigma_i>0$,

$$
\langle \nabla\mathcal L(x),v_i\rangle
=\sigma_i\langle u_i,Ax-y\rangle
=\sigma_i^2\langle v_i,x\rangle-\sigma_i\langle u_i,y\rangle.
$$

Moreover, along the line $x+\alpha v_i$,

$$
\frac{d^2}{d\alpha^2}\mathcal L(x+\alpha v_i)=\sigma_i^2.
$$

Hence a weighted data term $\lambda\mathcal L$ has curvature $\lambda\sigma_i^2$ along $v_i$, and this restoring curvature vanishes as $\sigma_i\to0$ for every fixed finite $\lambda$.

For a bounded band-weighted squared loss

$$
\mathcal L_W(x)=\frac12\|W(Ax-y)\|^2,
$$

the curvature along $v_i$ is

$$
\frac{d^2}{d\alpha^2}\mathcal L_W(x+\alpha v_i)=\|WAv_i\|^2\le \|W\|^2\sigma_i^2.
$$

Thus a bounded observation-band weight cannot create nonvanishing curvature in an $A$-near-null direction.
\end{proposition}

\begin{proof}
Differentiating the squared residual gives

$$
\nabla\mathcal L(x)=A^\top(Ax-y).
$$

Taking the inner product with $v_i$ gives

$$
\langle \nabla\mathcal L(x),v_i\rangle
=\langle A^\top(Ax-y),v_i\rangle
=\langle Ax-y,Av_i\rangle
=\sigma_i\langle Ax-y,u_i\rangle.
$$

Since

$$
\langle Ax,u_i\rangle=\langle x,A^\top u_i\rangle=\sigma_i\langle x,v_i\rangle,
$$

we obtain

$$
\langle \nabla\mathcal L(x),v_i\rangle
=\sigma_i^2\langle v_i,x\rangle-\sigma_i\langle u_i,y\rangle.
$$

Now write the residual along the line $x+\alpha v_i$ as

$$
A(x+\alpha v_i)-y=(Ax-y)+\alpha\sigma_i u_i.
$$

Therefore

$$
\mathcal L(x+\alpha v_i)
=\frac12\|(Ax-y)+\alpha\sigma_i u_i\|^2,
$$

so the second derivative with respect to $\alpha$ is $\sigma_i^2$.  Multiplication by $\lambda$ multiplies this curvature by $\lambda$.

For the weighted loss,

$$
W(A(x+\alpha v_i)-y)=W(Ax-y)+\alpha WAv_i,
$$

so the second derivative is $\|WAv_i\|^2$.  Since

$$
\|WAv_i\|\le \|W\|\|Av_i\|=\|W\|\sigma_i,
$$

we get the stated bound.
\end{proof}

The equilibrium implication used in the main text is the following precise corollary.

\begin{proposition}[P3 corollary: finite observation weight cannot balance an $O(1)$ near-null force without large displacement]
Fix a singular direction $v_i$ with $\sigma_i>0$.  Suppose that, in addition to the weighted observation loss $\lambda\mathcal L$, there is a differentiable prior or network-induced term $G$ whose local derivative along $v_i$ is a nonzero scalar $g_i$.  In the one-dimensional model where this competing derivative is constant over the displacement considered, stationarity along $v_i$ requires a displacement of magnitude

$$
|\alpha|=\frac{|g_i|}{\lambda\sigma_i^2}
$$

from the data-only stationary point.  With a band-weighted loss the denominator is $\lambda\|WAv_i\|^2$, which is at most $\lambda\|W\|^2\sigma_i^2$.  Thus, for every fixed finite $\lambda$ and bounded $W$, the displacement needed to balance a fixed nonzero competing force diverges as $\sigma_i\to0$.
\end{proposition}

\begin{proof}
Along $x+\alpha v_i$, the data term is a scalar quadratic whose second derivative is $\sigma_i^2$.  If $\alpha=0$ is the stationary point of the data term alone, then the derivative of $\lambda\mathcal L$ at displacement $\alpha$ is $\lambda\sigma_i^2\alpha$.  Adding the constant competing derivative $g_i$ gives the scalar stationarity equation

$$
\lambda\sigma_i^2\alpha+g_i=0.
$$

Therefore

$$
\alpha=-\frac{g_i}{\lambda\sigma_i^2}.
$$

The weighted case is identical with $\sigma_i^2$ replaced by $\|WAv_i\|^2$, and Proposition P3 gives $\|WAv_i\|^2\le\|W\|^2\sigma_i^2$.
\end{proof}

This corollary is intentionally asymptotic and finite-weight.  For a fixed nonzero $\sigma_i$, an artificially chosen weight of order $1/\sigma_i^2$ can create order-one curvature in that particular direction, but such a choice is an inverse-MTF amplification rather than a new observation.  Proposition P4 shows why this does not recover reliable information in the presence of noise: the same division by $\sigma_i$ amplifies noise.

---

## A.2.5 Statistical unidentifiability

\begin{proposition}[P4: variance lower bound for a singular coordinate]
Assume the observation model

$$
y=Ax^\star+n,
$$

where $\mathbb E[n]=0$ and $\operatorname{Cov}(n)=\eta^2 I$.  Let $v_i$ be a right singular vector with $\sigma_i>0$, and let

$$
c_i=\langle v_i,x^\star\rangle.
$$

For any linear estimator $\widehat c_i=a^\top y$ that is unbiased for $c_i$ for all $x^\star$, its variance satisfies

$$
\operatorname{Var}(\widehat c_i)\ge \frac{\eta^2}{\sigma_i^2}.
$$

Equality is attained by $a=u_i/\sigma_i$.  If $\sigma_i=0$, no linear estimator can be unbiased for $c_i$ for all $x^\star$.  If the noise is Gaussian, the same bound is the Cram\'er--Rao bound for any regular unbiased estimator of this orthogonal coordinate.
\end{proposition}

\begin{proof}
Unbiasedness for every $x^\star$ means

$$
\mathbb E[a^\top y]=a^\top A x^\star=\langle v_i,x^\star\rangle
\quad\text{for all }x^\star.
$$

Equivalently,

$$
A^\top a=v_i.
$$

Taking the inner product with $v_i$ and using $Av_i=\sigma_i u_i$ gives

$$
1=\langle v_i,v_i\rangle
=\langle A^\top a,v_i\rangle
=\langle a,Av_i\rangle
=\sigma_i\langle a,u_i\rangle.
$$

By Cauchy--Schwarz,

$$
1\le \sigma_i\|a\|,
$$

so

$$
\|a\|^2\ge\frac{1}{\sigma_i^2}.
$$

Since $\operatorname{Cov}(n)=\eta^2 I$,

$$
\operatorname{Var}(\widehat c_i)
=\operatorname{Var}(a^\top n)
=\eta^2\|a\|^2
\ge\frac{\eta^2}{\sigma_i^2}.
$$

The choice $a=u_i/\sigma_i$ satisfies $A^\top a=v_i$ and attains equality.  If $\sigma_i=0$, then $Av_i=0$, so for any $a$,

$$
\langle A^\top a,v_i\rangle=\langle a,Av_i\rangle=0,
$$

which cannot equal $\langle v_i,v_i\rangle=1$; hence no globally unbiased linear estimator exists.

For Gaussian noise, the Fisher information in the singular-vector coordinate system is diagonal with entry $\sigma_i^2/\eta^2$ for coordinate $c_i$, giving the same Cram\'er--Rao lower bound.
\end{proof}

This separates three levels of failure.  Proposition P1 is exact invariance at $\sigma_i=0$.  Proposition P2/P3 says that losses and restoring forces scale like $\sigma_i$ or $\sigma_i^2$ in near-null directions.  Proposition P4 says the data themselves carry only Fisher information $\sigma_i^2/\eta^2$ about coordinate $c_i$.  In particular, if the largest plausible signal along a unit direction is $|c_i|\le C$, then its observation-domain signal-to-noise ratio is at most $\sigma_i C/\eta$.  When $\sigma_i C\lesssim\eta$, even the optimal linear unbiased estimate has standard deviation at least comparable to the signal size.

---

## A.3.2 Formal proxy anti-correlation lemma

The proxy statement is not unconditional.  A high-frequency perturbation can increase the artifact score and decrease raw-control correlation to first order only when it is both more high-frequency than the current residual and not aligned with the fixed raw-control high-pass map.  The lemma below states the exact first-order conditions.

Let $P$ be the centering operator $Pu=u-\operatorname{mean}(u)\mathbf 1$.  All norms and inner products in this section may be read after centering; to simplify notation, assume $u_0$, $e$, and $c$ are already centered and that $c\ne0$.  Let

$$
L_1=I-G_{\sigma=1},
\qquad
L_2=\nabla^2,
\qquad
\beta=0.25.
$$

Define

$$
N(u)=\|L_1u\|+\beta\|L_2u\|,
\qquad
S(u)=\|u\|,
$$

so that the artifact proxy is

$$
\operatorname{artifact}(u)=\frac{N(u)}{S(u)},
$$

up to the common constant factors converting norms to standard deviations, which cancel in the ratio.  The raw-control correlation is

$$
\operatorname{corr}(u)=\frac{\langle u,c\rangle}{\|u\|\|c\|}.
$$

\begin{lemma}[L5: first-order anti-correlation along an unobserved style direction]
Let $u(\alpha)=u_0+\alpha e$.  Assume

$$
\|u_0\|>0,
\qquad
\|L_1u_0\|>0,
\qquad
\|L_2u_0\|>0,
\qquad
\rho_0:=\operatorname{corr}(u_0)>0.
$$

Define the normalized broadband alignment

$$
s_0(e)=\frac{\langle u_0,e\rangle}{\|u_0\|^2}
$$

and the normalized high-frequency directional growth

$$
h_0(e)=
\frac{
\frac{\langle L_1u_0,L_1e\rangle}{\|L_1u_0\|}
+\beta\frac{\langle L_2u_0,L_2e\rangle}{\|L_2u_0\|}
}{
\|L_1u_0\|+\beta\|L_2u_0\|
}.
$$

If

$$
h_0(e)>s_0(e),
$$

then

$$
\left.\frac{d}{d\alpha}\operatorname{artifact}(u_0+\alpha e)\right|_{\alpha=0}>0.
$$

If, additionally,

$$
\frac{\langle e,c\rangle}{\|u_0\|\|c\|}
<
\rho_0\,s_0(e),
$$

then

$$
\left.\frac{d}{d\alpha}\operatorname{corr}(u_0+\alpha e)\right|_{\alpha=0}<0.
$$

In particular, for an edge-reinforcing style perturbation with $\langle u_0,e\rangle>0$ and negligible raw-control alignment $\langle e,c\rangle\approx0$, the correlation decreases to first order while the artifact score increases, provided the high-frequency directional growth condition $h_0(e)>s_0(e)$ holds.
\end{lemma}

\begin{proof}
For a nonzero vector-valued differentiable path $q(\alpha)$,

$$
\left.\frac{d}{d\alpha}\|q(\alpha)\|\right|_{\alpha=0}
=\frac{\langle q(0),q'(0)\rangle}{\|q(0)\|}.
$$

Applying this to $L_1u(\alpha)$, $L_2u(\alpha)$, and $u(\alpha)$ gives

$$
N'(0)=
\frac{\langle L_1u_0,L_1e\rangle}{\|L_1u_0\|}
+\beta\frac{\langle L_2u_0,L_2e\rangle}{\|L_2u_0\|},
$$

and

$$
S'(0)=\frac{\langle u_0,e\rangle}{\|u_0\|}.
$$

Since $\operatorname{artifact}(u)=N(u)/S(u)$,

$$
\left.\frac{d}{d\alpha}\operatorname{artifact}(u(\alpha))\right|_{\alpha=0}
=\frac{N'(0)S(u_0)-N(u_0)S'(0)}{S(u_0)^2}.
$$

This derivative is positive exactly when

$$
\frac{N'(0)}{N(u_0)}>
\frac{S'(0)}{S(u_0)}
=\frac{\langle u_0,e\rangle}{\|u_0\|^2},
$$

which is $h_0(e)>s_0(e)$.

For the correlation, write

$$
C(\alpha)=\operatorname{corr}(u_0+\alpha e)
=\frac{\langle u_0+\alpha e,c\rangle}{\|u_0+\alpha e\|\|c\|}.
$$

Differentiating at $\alpha=0$ yields

$$
C'(0)
=
\frac{\langle e,c\rangle}{\|u_0\|\|c\|}
-
\rho_0\frac{\langle u_0,e\rangle}{\|u_0\|^2}.
$$

Therefore $C'(0)<0$ exactly under the displayed inequality.
\end{proof}

If $\langle e,c\rangle=0$ but also $\langle u_0,e\rangle=0$, the first derivative of the correlation is zero rather than negative.  In that orthogonal special case, if $\rho_0>0$ then the correlation decreases at second order because the numerator is fixed while the denominator grows:

$$
\operatorname{corr}(u_0+\alpha e)
=\rho_0\left(1+\alpha^2\frac{\|e\|^2}{\|u_0\|^2}\right)^{-1/2}
$$

when $e\perp u_0$ and $e\perp c$.  Thus the strict first-order anti-correlation claimed in the main text requires the edge-reinforcing condition $\langle u_0,e\rangle>0$ or a sufficiently negative $\langle e,c\rangle$.

---

## Assumptions and caveats

1. **Fixed linear operator.**  The proofs assume that $t_k$ is known and fixed, so $S_{t_k}$ is a fixed linear resampling matrix and $A_k=DBHS_{t_k}$ is linear.  If shifts are optimized jointly with $x$, the map from parameters to observations is no longer the fixed linear operator analyzed here.

2. **Finite-dimensional discretization.**  The SVD statements are exact for the implemented finite matrices.  The Fourier/MTF interpretation additionally assumes periodic or interior-domain shift-invariant boundary handling.  Nonperiodic boundaries perturb the Fourier diagonalization but do not affect the algebraic SVD propositions.

3. **Multi-frame phase diversity.**  The exact stacked null space is $\bigcap_k\ker(A_k)$.  More sub-pixel phases can remove single-frame downsampling alias nulls if the alias block matrix gains rank.  They do not remove common transfer zeros of $BH$, and they do not make very small MTF directions reliable.  Thus phase diversity shrinks the blind space but does not eliminate the physically attenuated $\varepsilon$-null space.

4. **Exact versus approximate mechanisms.**  Exact null directions arise from common transfer zeros, such as the ideal detector-box sinc zero at the $10\,\mu{\rm m}$ period on the $2\times$ Nyquist boundary, and from alias-cancellation vectors that survive all sampled phases.  Approximate null directions arise from small but nonzero Gaussian/box MTF and from poorly conditioned joint alias coverage.

5. **Bounded observation-band weights.**  The statement “band weighting cannot help” means bounded $W$ and fixed finite loss weight $\lambda$.  An unbounded inverse filter or a weight chosen to scale like $1/\sigma_i^2$ can create algebraic curvature in a known nonzero singular direction, but it also amplifies noise by the same inverse singular value and is outside the finite-anchor claim.

6. **Loss regularity.**  Proposition P2 uses a Lipschitz bound for $\ell$ on the relevant observation residual range.  Squared loss is not globally Lipschitz on all of $\mathbb R^{KM}$, so P2 applies locally to it; Proposition P3 gives the exact squared-loss calculation without a Lipschitz assumption.

7. **Noise model.**  Proposition P4 assumes zero-mean noise with covariance $\eta^2I$ for the linear unbiased estimator bound.  If noise is correlated, the correct operator is the whitened operator $\eta C_n^{-1/2}A$ or, equivalently, the Fisher information $A^\top C_n^{-1}A$.

8. **No claim about neural-network dynamics.**  These propositions do not prove that a UNet will drift, nor do they analyze nonconvex SGD.  They prove only that if a drift component lies in $\ker(A)$ or $\mathcal N_\varepsilon$, then observation-domain losses provide zero or vanishing information and restoring force in that component.

9. **Proxy anti-correlation is conditional.**  Lemma L5 requires the perturbation to be high-frequency relative to $u_0$, edge-reinforcing or otherwise denominator-increasing, and nearly orthogonal to the fixed raw-control high-pass map $c$.  Without these conditions, the first-order signs can be zero or can change.

---

## Compact takeaway

The rigorous conclusion is modest but strong enough for the paper’s claim.  Exact null-space components are completely invisible to any observation-domain loss.  Near-null components change such losses only by $O(\sigma_i)$ and receive squared-loss restoring curvature only $O(\sigma_i^2)$.  Frequency-band anchoring after the forward operator can suppress or reweight already-observed residuals, but a bounded $W$ cannot recreate information that $A$ has erased or attenuated below the noise floor.  The physical MTF values at the $2\times$ Nyquist period place the finest delivered-grid structures precisely in this exact or approximate blind region, especially for $\sigma\in[0.35,0.5]$ LR pixels and after the detector box response is included.
