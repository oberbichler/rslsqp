//! Core numerical building blocks for the SLSQP solver.
//!
//! Port of the first half of `slsqp_core.f90`.  Contains:
//!
//! - [`enforce_bounds`] — project variables onto their feasible box.
//! - [`check_convergence`] — multi-criterion convergence test.
//! - [`g1`] — Givens rotation construction.
//! - [`ldl`] — rank-one update of a packed LDL^T factorisation (BFGS).
//! - [`h12_construct`] / [`h12_apply`] / [`h12`] — Householder transform
//!   construction and application.
//! - [`hfti`] — rank-deficient least-squares via column-pivoted Householder QR.

use crate::core_types::ColMat;
use crate::support::*;

// ===========================================================================
// Bound enforcement
// ===========================================================================

/// Project each element of `x` onto the box `[xl, xu]`.
///
/// Elements where the bound is `NaN` or beyond `±infbnd` are treated as
/// unbounded in that direction.
///
/// # Arguments
/// * `x`      — Variables to clip (modified in-place).
/// * `xl`     — Lower bounds (length `n`).
/// * `xu`     — Upper bounds (length `n`).
/// * `infbnd` — Threshold for "infinite" bounds.
pub fn enforce_bounds(x: &mut [f64], xl: &[f64], xu: &[f64], infbnd: f64) {
    let n = x.len();
    for i in 0..n {
        // Enforce lower bound (skip if NaN or effectively -∞).
        if !xl[i].is_nan() && xl[i] > -infbnd {
            if x[i] < xl[i] {
                x[i] = xl[i];
            }
        }
        // Enforce upper bound (skip if NaN or effectively +∞).
        if !xu[i].is_nan() && xu[i] < infbnd {
            if x[i] > xu[i] {
                x[i] = xu[i];
            }
        }
    }
}

// ===========================================================================
// Convergence check
// ===========================================================================

/// Multi-criterion convergence test for the SQP iteration.
///
/// Checks several stopping conditions in order and returns `converged` if any
/// is met, otherwise `not_converged`.
///
/// # Convergence criteria (checked in order)
///
/// 1. `|f - f0| < acc` — objective change below accuracy.
/// 2. `||s||_2 < acc` — search direction norm below accuracy.
/// 3. `|f| < tolf` — absolute objective below tolerance (if `tolf >= 0`).
/// 4. `|f - f0| < toldf` — objective change below tolerance (if `toldf >= 0`).
/// 5. `||x - x0||_2 < toldx` — iterate change below tolerance (if `toldx >= 0`).
///
/// The function returns `not_converged` immediately if:
/// - `h3 >= acc` (constraint violation too large).
/// - `inconsistent_linearization` is `true`.
/// - `f` is NaN.
///
/// # Arguments
/// * `n`     — Number of variables.
/// * `f`, `f0` — Current and previous objective values.
/// * `x`, `x0` — Current and previous iterates.
/// * `s`     — Search direction.
/// * `h3`    — Constraint violation measure.
/// * `acc`   — Primary convergence accuracy.
/// * `tolf`, `toldf`, `toldx` — Optional tolerances (negative = disabled).
/// * `converged`, `not_converged` — Mode values to return.
/// * `inconsistent_linearization` — Whether the linearisation was inconsistent.
pub fn check_convergence(
    n: usize,
    f: f64,
    f0: f64,
    x: &[f64],
    x0: &[f64],
    s: &[f64],
    h3: f64,
    acc: f64,
    tolf: f64,
    toldf: f64,
    toldx: f64,
    converged: i32,
    not_converged: i32,
    inconsistent_linearization: bool,
) -> i32 {
    // Quick rejection: constraint violation, inconsistency, or NaN objective.
    if h3 >= acc || inconsistent_linearization || f.is_nan() {
        return not_converged;
    }

    let mut ok = false;

    // Criterion 1: objective change.
    if !ok {
        ok = (f - f0).abs() < acc;
    }
    // Criterion 2: search direction norm.
    if !ok {
        ok = dnrm2(n, s, 1) < acc;
    }
    // Criterion 3: absolute objective (optional).
    if !ok && tolf >= ZERO {
        ok = f.abs() < tolf;
    }
    // Criterion 4: objective change tolerance (optional).
    if !ok && toldf >= ZERO {
        ok = (f - f0).abs() < toldf;
    }
    // Criterion 5: iterate change norm (optional).
    if !ok && toldx >= ZERO {
        let mut xmx0 = vec![0.0; n];
        for i in 0..n {
            xmx0[i] = x[i] - x0[i];
        }
        ok = dnrm2(n, &xmx0, 1) < toldx;
    }

    if ok { converged } else { not_converged }
}

// ===========================================================================
// Givens rotation
// ===========================================================================

/// Construct a Givens rotation that zeroes the second element.
///
/// Computes `(c, s, sig)` such that:
/// ```text
///     [ c  s ] [ a ]   [ sig ]
///     [-s  c ] [ b ] = [  0  ]
/// ```
/// where `sig = ±sqrt(a² + b²)` with the sign chosen to match the larger
/// input in magnitude.
///
/// # Returns
/// `(c, s, sig)` — cosine, sine, and resultant magnitude.
pub fn g1(a: f64, b: f64) -> (f64, f64, f64) {
    if a.abs() > b.abs() {
        let xr = b / a;
        let yr = (ONE + xr * xr).sqrt();
        let c = a.signum() * (ONE / yr);
        let s = c * xr;
        let sig = a.abs() * yr;
        (c, s, sig)
    } else if b.abs() > ZERO {
        let xr = a / b;
        let yr = (ONE + xr * xr).sqrt();
        let s = b.signum() * (ONE / yr);
        let c = s * xr;
        let sig = b.abs() * yr;
        (c, s, sig)
    } else {
        (ZERO, ONE, ZERO)
    }
}

// ===========================================================================
// LDL^T rank-one update (packed storage)
// ===========================================================================

/// Rank-one update of a packed LDL^T factorisation.
///
/// Updates the factorisation `A = L D L^T` (stored as a packed vector `a`) by:
/// ```text
///     A := A + sigma * z z^T
/// ```
///
/// The packed storage format stores the lower triangle row-by-row:
/// `a[0] = d_0, a[1] = l_{1,0}, a[2] = d_1, a[3] = l_{2,0}, ...`
/// where diagonal elements are the D values and off-diagonals are L values.
///
/// For negative `sigma`, a special preparation pass is performed to maintain
/// numerical stability (Gill–Murray–Wright procedure).
///
/// # Arguments
/// * `n`     — Dimension of the matrix.
/// * `a`     — Packed LDL^T factors (modified in-place).
/// * `z`     — Update vector (modified in-place as workspace).
/// * `sigma` — Scalar multiplier (positive = rank-one addition, negative = rank-one subtraction).
/// * `w`     — Workspace vector (length `n`, modified in-place).
pub fn ldl(n: usize, a: &mut [f64], z: &mut [f64], sigma: f64, w: &mut [f64]) {
    if sigma.abs() <= ZERO {
        return;
    }
    let mut ij: usize = 0;
    let mut t = ONE / sigma;

    if sigma <= ZERO {
        // --- Prepare negative update (Gill–Murray–Wright) ---
        // Forward pass: compute w = L^{-1} z and accumulate t.
        w[..n].copy_from_slice(&z[..n]);
        for i in 0..n {
            let v = w[i];
            t += v * v / a[ij];
            for j in (i + 1)..n {
                ij += 1;
                w[j] -= v * a[ij];
            }
            ij += 1;
        }
        // Clamp t to avoid division by zero.
        if t >= ZERO {
            t = EPMACH / sigma;
        }
        // Backward pass: compute new t values per row.
        for i in 0..n {
            let j = n - 1 - i;
            ij -= i + 1;
            let u_val = w[j];
            w[j] = t;
            t -= u_val * u_val / a[ij];
        }
    }

    // --- Main update loop ---
    ij = 0;
    for i in 0..n {
        let v = z[i];
        let delta = v / a[ij];
        let tp = if sigma < ZERO { w[i] } else { t + delta * v };
        let alpha_val = tp / t;
        a[ij] = alpha_val * a[ij];
        if i == n - 1 {
            return;
        }
        let beta_val = delta / tp;

        if alpha_val > FOUR {
            // Use the gamma formulation for better numerical stability
            // when alpha is large.
            let gamma_val = t / tp;
            for j in (i + 1)..n {
                ij += 1;
                let u_val = a[ij];
                a[ij] = gamma_val * u_val + beta_val * z[j];
                z[j] -= v * u_val;
            }
        } else {
            for j in (i + 1)..n {
                ij += 1;
                z[j] -= v * a[ij];
                a[ij] += beta_val * z[j];
            }
        }
        ij += 1;
        t = tp;
    }
}

// ===========================================================================
// Householder transform — construct / apply / convenience wrapper
// ===========================================================================

/// Construct a Householder reflector (mode 1).
///
/// Builds the Householder vector from `u` and returns the `up` scalar.
/// The pivot element `u[lpivot * iue]` is overwritten with the signed norm.
///
/// All indices (`lpivot`, `l1`, `m`) are **0-based**.
/// `u[j * iue]` addresses element `j` of the Householder vector (strided access).
///
/// # Arguments
/// * `lpivot` — Pivot index.
/// * `l1`     — First index of the "tail" portion of the vector.
/// * `m`      — One past the last index (exclusive upper bound).
/// * `u`      — Householder vector storage (modified in-place).
/// * `iue`    — Stride between consecutive elements of `u`.
///
/// # Returns
/// The `up` scalar (original pivot minus signed norm).
/// Returns `ZERO` if the pivot range is invalid or the column is zero.
pub fn h12_construct(lpivot: usize, l1: usize, m: usize, u: &mut [f64], iue: usize) -> f64 {
    if lpivot >= l1 || l1 >= m {
        return ZERO;
    }

    // Find the maximum absolute value for scaling.
    let mut cl = u[lpivot * iue].abs();
    for j in l1..m {
        cl = cl.max(u[j * iue].abs());
    }
    if cl <= ZERO {
        return ZERO; // Zero column — no reflector needed.
    }

    // Compute the scaled norm.
    let clinv = ONE / cl;
    let mut sm = (u[lpivot * iue] * clinv).powi(2);
    for j in l1..m {
        sm += (u[j * iue] * clinv).powi(2);
    }
    cl *= sm.sqrt();

    // Choose sign opposite to pivot for numerical stability.
    if u[lpivot * iue] > ZERO {
        cl = -cl;
    }
    let up = u[lpivot * iue] - cl;
    u[lpivot * iue] = cl;
    up
}

/// Apply a previously-constructed Householder reflector (mode 2).
///
/// Applies the Householder transform `H = I - (1/b) * u * u^T` to one or more
/// column vectors in `c`, where `b = up * u[lpivot * iue]`.
///
/// `u` is **read-only** — it supplies the Householder vector elements at
/// `u[j * iue]` for `j` in `l1..m`, plus the pivot at `u[lpivot * iue]`.
/// `up` is the scalar returned by [`h12_construct`].
///
/// # Arguments
/// * `lpivot` — Pivot index (0-based).
/// * `l1`     — First tail index (0-based).
/// * `m`      — Exclusive upper bound.
/// * `u`      — Householder vector (read-only).
/// * `iue`    — Stride for `u`.
/// * `up`     — Scalar from [`h12_construct`].
/// * `c`      — Target vectors (modified in-place).
/// * `ice`    — Element stride within each target vector.
/// * `icv`    — Vector stride (offset between consecutive target vectors).
/// * `ncv`    — Number of target vectors to transform.
pub fn h12_apply(
    lpivot: usize,
    l1: usize,
    m: usize,
    u: &[f64],
    iue: usize,
    up: f64,
    c: &mut [f64],
    ice: usize,
    icv: usize,
    ncv: usize,
) {
    if lpivot >= l1 || l1 >= m {
        return;
    }

    // b = up * u[lpivot]; must be negative for a valid reflector.
    if u[lpivot * iue].abs() <= ZERO || ncv == 0 {
        return;
    }
    let b = up * u[lpivot * iue];
    if b >= ZERO {
        return;
    }
    let inv_b = ONE / b;

    // ── Fast path: contiguous u (iue==1) and contiguous target elements (ice==1) ──
    //
    // This covers the dominant call patterns:
    //   - column-major ColMat operations (iue=1, ice=1, icv=rows)
    //   - hfti with col_buf (iue=1, ice=stride, icv=1) — NOT this path
    //   - lsi column-major QR (iue=1, ice=1, icv=e_rows)
    //
    // The contiguous case enables LLVM auto-vectorisation of both the dot
    // product (sm computation) and the rank-1 update.  For large m this is
    // 2-4× faster than the generic strided loop.
    if iue == 1 && ice == 1 {
        let tail_len = m - l1;
        let u_tail = &u[l1..l1 + tail_len];

        let mut c_off = if lpivot == 0 { 0 } else { lpivot };
        // First vector starts at ice * lpivot, but ice==1 so it's just lpivot.
        // With icv, successive vectors are offset by icv.
        // The generic formula: i2 = -icv + lpivot, then i2 += icv per vector.
        // So first i2 = lpivot, second i2 = lpivot + icv, etc.
        // i3_start = i2 + (l1 - lpivot)

        for _ in 0..ncv {
            let pivot_idx = c_off;
            let tail_start = c_off + (l1 - lpivot);

            // Dot product: sm = up * c[pivot] + u_tail · c_tail
            let mut sm = c[pivot_idx] * up;
            let c_tail = &c[tail_start..tail_start + tail_len];
            for k in 0..tail_len {
                sm += c_tail[k] * u_tail[k];
            }

            // Update: c -= (sm * inv_b) * u
            if sm.abs() > ZERO {
                let sm = sm * inv_b;
                c[pivot_idx] += sm * up;
                let c_tail = &mut c[tail_start..tail_start + tail_len];
                for k in 0..tail_len {
                    c_tail[k] += sm * u_tail[k];
                }
            }

            c_off += icv;
        }
        return;
    }

    // ── General strided path (handles arbitrary ice, icv, iue) ──
    let ice_s = ice as isize;
    let icv_s = icv as isize;
    let mut i2 = -icv_s + ice_s * (lpivot as isize);
    let incr = ice_s * (l1 as isize - lpivot as isize);

    for _ in 0..ncv {
        i2 += icv_s;
        let i3_start = i2 + incr;
        let mut i3 = i3_start;

        // Compute sm = u^T * c for the current target vector.
        let mut sm = c[i2 as usize] * up;
        for ii in l1..m {
            sm += c[i3 as usize] * u[ii * iue];
            i3 += ice_s;
        }

        // Apply: c -= (sm / b) * u.
        if sm.abs() > ZERO {
            let sm = sm * inv_b;
            c[i2 as usize] += sm * up;
            let mut i4 = i3_start;
            for ii in l1..m {
                c[i4 as usize] += sm * u[ii * iue];
                i4 += ice_s;
            }
        }
    }
}

/// Convenience wrapper combining construction and application in a single call.
///
/// If `mode != 2`, constructs the Householder reflector and stores `up` in
/// `up_box[0]`.  Then applies the reflector to the target vectors in `c`.
///
/// This preserves backward compatibility with the PyO3 bindings and call
/// sites where `u` and `c` do **not** alias the same buffer.
///
/// # Arguments
/// * `mode`   — 1 = construct + apply; 2 = apply only (using existing `up_box[0]`).
/// * `up_box` — Single-element slice carrying the `up` scalar between calls.
///
/// See [`h12_construct`] and [`h12_apply`] for the remaining parameters.
#[allow(dead_code)]
pub fn h12(
    mode: i32,
    lpivot: usize,
    l1: usize,
    m: usize,
    u: &mut [f64],
    iue: usize,
    up_box: &mut [f64],
    c: &mut [f64],
    ice: usize,
    icv: usize,
    ncv: usize,
) {
    if mode != 2 {
        up_box[0] = h12_construct(lpivot, l1, m, u, iue);
    }
    h12_apply(lpivot, l1, m, u, iue, up_box[0], c, ice, icv, ncv);
}

// ===========================================================================
// HFTI — Rank-deficient least-squares via column-pivoted Householder QR
// ===========================================================================

/// Rank-deficient least-squares via column-pivoted Householder QR.
///
/// Solves `min ‖A x − b‖₂` for one or more right-hand sides when `A` may be
/// rank-deficient.  This is a direct port of the original Fortran HFTI
/// algorithm that applies Householder reflectors in-place to both `A` and `B`
/// during factorisation — **no explicit Q matrix is ever formed**.
///
/// Complexity: `O(m n min(m,n))` for the QR, plus `O(m n_b min(m,n))` for the
/// RHS transforms — much cheaper than the `O(m³)` cost of forming Q.
///
/// # Algorithm
///
/// 1. Column-pivoted QR via Householder reflectors applied directly to `A`
///    and `B` (simultaneously).
/// 2. Pseudo-rank `k` determined by comparing `|R[i,i]|` to `tau`.
/// 3. Back-substitution on the `k × k` upper-triangular factor.
/// 4. If `k < n`, backward Householder transforms yield the minimum-norm
///    solution.
/// 5. Inverse column permutation recovers the original variable ordering.
///
/// # Arguments
/// * `a`     — Coefficient matrix (m × n column-major `ColMat`), overwritten.
/// * `m_val` — Number of rows of `A`.
/// * `n`     — Number of columns of `A`.
/// * `b`     — Right-hand side matrix (max(m,n) × nb column-major `ColMat`),
///             overwritten with the solution.
/// * `nb`    — Number of right-hand side columns.
/// * `tau`   — Pivot tolerance for determining pseudo-rank.
///
/// # Returns
/// `(krank, rnorm)` — pseudo-rank and per-column residual norms.
pub fn hfti(
    a: &mut ColMat,
    m_val: usize,
    n: usize,
    b: &mut ColMat,
    nb: usize,
    tau: f64,
) -> (usize, Vec<f64>) {
    profile_section!("hfti");
    let m = m_val;
    let nu = n;
    let nbu = nb;
    let ldiag = m.min(nu);
    let mut rnorm = vec![0.0; nbu.max(1)];

    if ldiag == 0 {
        return (0, rnorm);
    }

    // Dispatch to LAPACK blocked QR when BLAS is available for large problems.
    // dgeqp3 + dormqr + dtrtrs uses BLAS L3 (dgemm) internally,
    // which is 5-12× faster than our per-column L2 Householder applies.
    #[cfg(feature = "blas")]
    if ldiag >= 16 {
        return crate::lapack::hfti_lapack(
            &mut a.data,
            a.stride,
            m,
            nu,
            &mut b.data,
            b.stride,
            nbu,
            tau,
        );
    }

    let a_stride = a.stride;
    let b_stride = b.stride;

    // Column permutation tracking
    let mut ip = vec![0_usize; nu];
    for j in 0..nu {
        ip[j] = j;
    }

    // Householder scalars (one per column)
    let mut h_arr = vec![0.0_f64; nu];

    // Workspace for column extraction (to break aliasing)
    let max_dim = m.max(nu);
    let mut col_buf = vec![0.0_f64; max_dim];

    // Compute initial column norms squared
    // Column-major: column j is contiguous at a.data[j*a_stride..j*a_stride+m]
    let mut col_norms = vec![0.0_f64; nu];
    for j in 0..nu {
        let col = &a.data[j * a_stride..j * a_stride + m];
        col_norms[j] = col.iter().map(|v| v * v).sum();
    }

    let mut krank = ldiag;

    // ── Forward pass: column-pivoted QR with Householder ──────────────
    for j in 0..ldiag {
        // Find pivot column (max remaining column norm)
        let mut max_norm = col_norms[j];
        let mut jmax = j;
        for jj in (j + 1)..nu {
            if col_norms[jj] > max_norm {
                max_norm = col_norms[jj];
                jmax = jj;
            }
        }

        // Swap columns j and jmax in A, and their metadata.
        // Column-major: columns are contiguous slices — use slice swap.
        if jmax != j {
            ip.swap(j, jmax);
            col_norms.swap(j, jmax);
            // Since j < jmax, split_at_mut gives non-overlapping access.
            let (left, right) = a.data.split_at_mut(jmax * a_stride);
            let col_j = &mut left[j * a_stride..j * a_stride + m];
            let col_jmax = &mut right[..m];
            col_j.swap_with_slice(col_jmax);
        }

        // Construct Householder reflector for column j, rows j..m.
        // Column-major: column j is contiguous starting at a.data[j*a_stride].
        // iue = 1 because elements within a column are contiguous.
        let up = h12_construct(
            j,
            j + 1,
            m,
            &mut a.data[j * a_stride..],
            1,
        );
        h_arr[j] = up;

        // Apply Householder to remaining columns of A (j+1..n).
        // Column-major: columns are contiguous with stride a_stride between them.
        // Use h12_apply_batch (iue=1, ice=1, icv=a_stride).
        if j + 1 < nu {
            // Copy column j to col_buf to break u/c aliasing
            col_buf[..m].copy_from_slice(&a.data[j * a_stride..j * a_stride + m]);
            crate::lapack::h12_apply_batch(
                j,
                j + 1,
                m,
                &col_buf,
                up,
                &mut a.data[(j + 1) * a_stride..],
                a_stride,
                nu - j - 1,
            );
        }

        // Apply Householder to all columns of B.
        if nbu > 0 {
            col_buf[..m].copy_from_slice(&a.data[j * a_stride..j * a_stride + m]);
            crate::lapack::h12_apply_batch(j, j + 1, m, &col_buf, up, &mut b.data, b_stride, nbu);
        }

        // Update remaining column norms (deflate)
        for jj in (j + 1)..nu {
            let col = &a.data[jj * a_stride..jj * a_stride + m];
            col_norms[jj] = col[j + 1..m].iter().map(|v| v * v).sum();
        }

        // Check rank: if diagonal element is too small, truncate
        if a[(j, j)].abs() <= tau {
            krank = j;
            break;
        }
    }

    // ── Residual norms from B[krank..m, :] ────────────────────────────
    for jb in 0..nbu {
        let col = &b.data[jb * b_stride..jb * b_stride + m];
        let res_sq: f64 = col[krank..m].iter().map(|v| v * v).sum();
        rnorm[jb] = res_sq.sqrt();
    }

    // ── Back-substitution: R[0:k,0:k] y = B[0:k,:] ──────────────────
    // R is stored in the upper triangle of A after the Householder transforms.
    if krank > 0 {
        for jb in 0..nbu {
            for i in (0..krank).rev() {
                let mut s = b[(i, jb)];
                for jj in (i + 1)..krank {
                    s -= a[(i, jj)] * b[(jj, jb)];
                }
                b[(i, jb)] = s / a[(i, i)];
            }
        }

        // ── Minimum-norm solution when k < n ─────────────────────────
        // Apply backward Householder transforms to map the k-dimensional
        // solution into the full n-dimensional space.
        if krank < nu {
            // Zero out entries krank..n in B
            for jb in 0..nbu {
                for i in krank..nu {
                    b[(i, jb)] = 0.0;
                }
            }

            // Apply the Householder transforms in reverse order
            for j in (0..krank).rev() {
                col_buf[..m].copy_from_slice(&a.data[j * a_stride..j * a_stride + m]);
                crate::lapack::h12_apply_batch(
                    j,
                    j + 1,
                    m,
                    &col_buf,
                    h_arr[j],
                    &mut b.data,
                    b_stride,
                    nbu,
                );
            }
        }

        // ── Inverse column permutation ───────────────────────────────
        // Use col_buf as temporary for one RHS column at a time.
        for jb in 0..nbu {
            for j in 0..nu {
                col_buf[ip[j]] = b[(j, jb)];
            }
            for j in 0..nu {
                b[(j, jb)] = col_buf[j];
            }
        }
    } else {
        // Zero rank — zero out the solution vector.
        for jb in 0..nbu {
            for i in 0..nu {
                b[(i, jb)] = 0.0;
            }
        }
    }

    (krank, rnorm)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: f64 = 1e-12;

    // -- g1 (Givens rotation) ------------------------------------------------

    #[test]
    fn g1_zeroes_second_element() {
        let (c, s, sig) = g1(3.0, 4.0);
        // Verify: c*a + s*b = sig,  -s*a + c*b = 0
        assert!((c * 3.0 + s * 4.0 - sig).abs() < TOL);
        assert!((-s * 3.0 + c * 4.0).abs() < TOL);
    }

    #[test]
    fn g1_sig_is_hypotenuse() {
        let (_, _, sig) = g1(3.0, 4.0);
        assert!((sig - 5.0).abs() < TOL);
    }

    #[test]
    fn g1_a_dominant() {
        let (c, s, sig) = g1(5.0, 1.0);
        assert!(sig > 0.0);
        assert!((c * 5.0 + s * 1.0 - sig).abs() < TOL);
        assert!((-s * 5.0 + c * 1.0).abs() < TOL);
    }

    #[test]
    fn g1_b_dominant() {
        let (c, s, sig) = g1(1.0, 5.0);
        assert!(sig > 0.0);
        assert!((c * 1.0 + s * 5.0 - sig).abs() < TOL);
        assert!((-s * 1.0 + c * 5.0).abs() < TOL);
    }

    #[test]
    fn g1_both_zero() {
        let (c, s, sig) = g1(0.0, 0.0);
        assert_eq!(c, 0.0);
        assert_eq!(s, 1.0);
        assert_eq!(sig, 0.0);
    }

    #[test]
    fn g1_negative_values() {
        let (c, s, sig) = g1(-3.0, -4.0);
        assert!((c * -3.0 + s * -4.0 - sig).abs() < TOL);
        assert!((-s * -3.0 + c * -4.0).abs() < TOL);
    }

    #[test]
    fn g1_rotation_is_orthogonal() {
        let (c, s, _) = g1(7.0, 11.0);
        // c² + s² = 1 for an orthogonal rotation
        assert!((c * c + s * s - 1.0).abs() < TOL);
    }

    // -- h12_construct / h12_apply (Householder) -----------------------------

    #[test]
    fn h12_construct_basic() {
        // Column: [3.0, 1.0, 1.0, 1.0]
        let mut u = vec![3.0, 1.0, 1.0, 1.0];
        let up = h12_construct(0, 1, 4, &mut u, 1);

        // After construction, u[0] should contain the signed norm
        // up should be the original pivot minus the signed norm
        // The norm of [3,1,1,1] = sqrt(12) ≈ 3.464
        let norm = (3.0_f64.powi(2) + 1.0 + 1.0 + 1.0).sqrt();
        assert!((u[0].abs() - norm).abs() < TOL);
        assert!(up != 0.0); // non-trivial reflector
    }

    #[test]
    fn h12_construct_invalid_range_returns_zero() {
        let mut u = vec![1.0, 2.0];
        // l1 >= m → invalid
        let up = h12_construct(0, 2, 2, &mut u, 1);
        assert_eq!(up, 0.0);
    }

    #[test]
    fn h12_apply_preserves_orthogonality() {
        // Construct Householder from column [2.0, 1.0, 1.0]
        let mut u = vec![2.0, 1.0, 1.0];
        let up = h12_construct(0, 1, 3, &mut u, 1);

        // Apply to a target vector [1.0, 0.0, 0.0]
        let mut c = vec![1.0, 0.0, 0.0];
        h12_apply(0, 1, 3, &u, 1, up, &mut c, 1, 1, 1);

        // The Householder reflector H = I - (2/||v||²) vv^T is orthogonal,
        // so it preserves norms: ||H*c|| = ||c||
        let norm_after = nrm2(&c);
        assert!((norm_after - 1.0).abs() < TOL);
    }

    #[test]
    fn h12_construct_then_apply_roundtrip() {
        // Construct a Householder reflector from the first column of a 3×3 matrix
        let mut col = vec![2.0, 1.0, 2.0];
        let up = h12_construct(0, 1, 3, &mut col, 1);

        // Apply H to two target vectors
        let mut c1 = vec![1.0, 0.0, 0.0];
        let mut c2 = vec![0.0, 1.0, 0.0];
        h12_apply(0, 1, 3, &col, 1, up, &mut c1, 1, 1, 1);
        h12_apply(0, 1, 3, &col, 1, up, &mut c2, 1, 1, 1);

        // H is an involution (H² = I), so applying twice should recover the original
        h12_apply(0, 1, 3, &col, 1, up, &mut c1, 1, 1, 1);
        h12_apply(0, 1, 3, &col, 1, up, &mut c2, 1, 1, 1);
        assert!((c1[0] - 1.0).abs() < TOL);
        assert!(c1[1].abs() < TOL);
        assert!(c1[2].abs() < TOL);
        assert!(c2[0].abs() < TOL);
        assert!((c2[1] - 1.0).abs() < TOL);
        assert!(c2[2].abs() < TOL);
    }

    #[test]
    fn h12_convenience_wrapper_mode1() {
        let mut u = vec![3.0, 1.0, 1.0];
        let mut up_box = vec![0.0];
        let mut c = vec![1.0, 2.0, 3.0];
        h12(1, 0, 1, 3, &mut u, 1, &mut up_box, &mut c, 1, 1, 1);
        // up_box should now contain the up value
        assert!(up_box[0] != 0.0);
    }

    // -- ldl (LDL^T rank-one update) -----------------------------------------

    #[test]
    fn ldl_positive_update_preserves_positive_definiteness() {
        // Start with identity: packed LDL^T of 2×2 identity is [1, 0, 1]
        // (d0=1, l10=0, d1=1)
        let mut a = vec![1.0, 0.0, 1.0];
        let mut z = vec![1.0, 1.0];
        let mut w = vec![0.0; 2];

        ldl(2, &mut a, &mut z, 1.0, &mut w);

        // After A := I + z*z^T = [[2, 1], [1, 2]]
        // LDL^T: d0=2, l10=0.5, d1 = 2 - 0.5²*2 = 1.5
        // Packed: [d0, l10, d1] = [2.0, 0.5, 1.5]
        assert!((a[0] - 2.0).abs() < TOL);
        assert!((a[1] - 0.5).abs() < TOL);
        assert!((a[2] - 1.5).abs() < TOL);
    }

    #[test]
    fn ldl_negative_update() {
        // Start with 2I packed: [2, 0, 2]
        let mut a = vec![2.0, 0.0, 2.0];
        let mut z = vec![1.0, 0.0];
        let mut w = vec![0.0; 2];

        // A := 2I - e1*e1^T = [[1, 0], [0, 2]]
        ldl(2, &mut a, &mut z, -1.0, &mut w);

        // Packed LDL^T of [[1,0],[0,2]]: d0=1, l10=0, d1=2
        assert!((a[0] - 1.0).abs() < TOL);
        assert!(a[1].abs() < TOL);
        assert!((a[2] - 2.0).abs() < TOL);
    }

    #[test]
    fn ldl_zero_sigma_noop() {
        let mut a = vec![1.0, 0.0, 1.0];
        let a_before = a.clone();
        let mut z = vec![1.0, 1.0];
        let mut w = vec![0.0; 2];

        ldl(2, &mut a, &mut z, 0.0, &mut w);
        assert_eq!(a, a_before);
    }

    #[test]
    fn ldl_3x3_positive_update() {
        // 3×3 identity packed: [1, 0, 0, 1, 0, 1]
        // Update: A := I + [1,1,1][1,1,1]^T = [[2,1,1],[1,2,1],[1,1,2]]
        let mut a = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let mut z = vec![1.0, 1.0, 1.0];
        let mut w = vec![0.0; 3];

        ldl(3, &mut a, &mut z, 1.0, &mut w);

        // Reconstruct A = L D L^T and verify
        let (d1, l21, l31, d2, l32, d3) = (a[0], a[1], a[2], a[3], a[4], a[5]);
        // Row 0: A[0,0] = d1
        assert!((d1 * 1.0 - 2.0).abs() < TOL, "A[0,0] = {} expected 2", d1);
        // Row 1: A[1,0] = l21 * d1,  A[1,1] = l21^2 * d1 + d2
        let a10 = l21 * d1;
        let a11 = l21 * l21 * d1 + d2;
        assert!((a10 - 1.0).abs() < TOL, "A[1,0] = {} expected 1", a10);
        assert!((a11 - 2.0).abs() < TOL, "A[1,1] = {} expected 2", a11);
        // Row 2: A[2,0] = l31 * d1,  A[2,1] = l31*l21*d1 + l32*d2,
        //         A[2,2] = l31^2*d1 + l32^2*d2 + d3
        let a20 = l31 * d1;
        let a21 = l31 * l21 * d1 + l32 * d2;
        let a22 = l31 * l31 * d1 + l32 * l32 * d2 + d3;
        assert!((a20 - 1.0).abs() < TOL, "A[2,0] = {} expected 1", a20);
        assert!((a21 - 1.0).abs() < TOL, "A[2,1] = {} expected 1", a21);
        assert!((a22 - 2.0).abs() < TOL, "A[2,2] = {} expected 2", a22);
    }

    // -- enforce_bounds ------------------------------------------------------

    #[test]
    fn enforce_bounds_clips_to_box() {
        let mut x = vec![-5.0, 0.5, 10.0];
        let xl = vec![-1.0, 0.0, 0.0];
        let xu = vec![1.0, 1.0, 5.0];
        enforce_bounds(&mut x, &xl, &xu, 1e20);
        assert_eq!(x, vec![-1.0, 0.5, 5.0]);
    }

    #[test]
    fn enforce_bounds_nan_bound_ignored() {
        let mut x = vec![-100.0, 100.0];
        let xl = vec![f64::NAN, 0.0];
        let xu = vec![0.0, f64::NAN];
        enforce_bounds(&mut x, &xl, &xu, 1e20);
        assert_eq!(x[0], -100.0); // lower bound NaN → not enforced
        assert_eq!(x[1], 100.0); // upper bound NaN → not enforced
    }

    #[test]
    fn enforce_bounds_infinite_bound_ignored() {
        let mut x = vec![-1e30, 1e30];
        let xl = vec![-1e50, 0.0];
        let xu = vec![0.0, 1e50];
        let infbnd = 1e40;
        enforce_bounds(&mut x, &xl, &xu, infbnd);
        assert_eq!(x[0], -1e30); // xl beyond -infbnd → ignored
        assert_eq!(x[1], 1e30); // xu beyond infbnd → ignored
    }

    // -- check_convergence ---------------------------------------------------

    #[test]
    fn check_convergence_detects_convergence() {
        let x = vec![1.0, 2.0];
        let x0 = vec![1.0, 2.0];
        let s = vec![0.0, 0.0];
        let mode = check_convergence(
            2, 1.0, 1.0, &x, &x0, &s, 0.0, 1e-6, -1.0, -1.0, -1.0, 0, 1, false,
        );
        assert_eq!(mode, 0); // converged
    }

    #[test]
    fn check_convergence_constraint_violation_rejects() {
        let x = vec![1.0];
        let x0 = vec![1.0];
        let s = vec![0.0];
        let mode = check_convergence(
            1, 1.0, 1.0, &x, &x0, &s, 1.0, 1e-6, -1.0, -1.0, -1.0, 0, 1, false,
        );
        assert_eq!(mode, 1); // not converged (h3 >= acc)
    }

    #[test]
    fn check_convergence_nan_objective_rejects() {
        let x = vec![1.0];
        let x0 = vec![1.0];
        let s = vec![0.0];
        let mode = check_convergence(
            1,
            f64::NAN,
            1.0,
            &x,
            &x0,
            &s,
            0.0,
            1e-6,
            -1.0,
            -1.0,
            -1.0,
            0,
            1,
            false,
        );
        assert_eq!(mode, 1); // not converged (NaN)
    }

    // -- hfti (rank-deficient least squares) ---------------------------------

    #[test]
    fn hfti_full_rank_identity() {
        use crate::core_types::ColMat;
        // Solve I*x = b → x = b
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = ColMat::from_vv(&[vec![3.0], vec![7.0]]);
        let (krank, rnorm) = hfti(&mut a, 2, 2, &mut b, 1, 1e-12);
        assert_eq!(krank, 2);
        assert!(rnorm[0] < TOL);
        assert!((b[(0, 0)] - 3.0).abs() < TOL);
        assert!((b[(1, 0)] - 7.0).abs() < TOL);
    }

    #[test]
    fn hfti_overdetermined_system() {
        use crate::core_types::ColMat;
        // 3×2 system: A = [[1,0],[0,1],[1,1]], b = [1, 2, 2]
        // Least-squares solution
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        let mut b = ColMat::zeros(3, 1);
        b[(0, 0)] = 1.0;
        b[(1, 0)] = 2.0;
        b[(2, 0)] = 2.0;
        let (krank, rnorm) = hfti(&mut a, 3, 2, &mut b, 1, 1e-12);
        assert_eq!(krank, 2);
        // The LS solution for [[1,0],[0,1],[1,1]]x=[1,2,2] is approximately [1/3, 5/3]
        // Verify by checking residual is small but nonzero
        assert!(rnorm[0] > 0.0);
        assert!(rnorm[0] < 2.0); // bounded residual
    }
}
