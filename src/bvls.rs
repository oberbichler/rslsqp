//! Bounded Variable Least Squares (BVLS) solver.
//!
//! Port of `bvls_module.f90`.  Solves the problem:
//!
//! ```text
//!     min  || A x - b ||_2
//!     s.t. bnd_lower[j] <= x[j] <= bnd_upper[j]   for all j
//! ```
//!
//! The main entry point is [`bvls`], with [`bvls_wrapper`] providing a
//! convenient NNLS interface (non-negative least squares: lower = 0, upper = ∞).
//!
//! # Algorithm
//!
//! The algorithm is an active-set method that partitions variables into:
//! - **Set P** (indices `0..nsetp`): "free" variables currently being optimised.
//! - **Set Z** (indices `iz1..=iz2`): "bound" variables fixed at a bound value.
//!
//! Each iteration selects a variable from Z to move into P (based on the dual
//! vector / gradient), then adjusts the solution and moves infeasible variables
//! back to Z until all variables in P satisfy their bounds.

use crate::core_types::ColMat;
use crate::support::nrm2;

/// Machine epsilon for `f64` — used as the column-independence threshold.
const EPS: f64 = f64::EPSILON;

// ---------------------------------------------------------------------------
// Helper routines
// ---------------------------------------------------------------------------

/// Householder transformation construction for BVLS.
///
/// Computes a Householder vector in-place within `u[p..]` and returns the
/// scalar `up` (the original pivot element minus the signed norm).  After the
/// call, `u[p]` contains the signed norm with opposite sign to the original
/// `u[p]` (to maximise numerical stability).
///
/// # Arguments
/// * `p` — Pivot index (0-based).
/// * `u` — Vector to transform (modified in-place from index `p` onward).
///
/// # Returns
/// The `up` scalar needed to apply the Householder reflector.
pub fn htc(p: usize, u: &mut [f64]) -> f64 {
    let vnorm = {
        let v = nrm2(&u[p..]);
        if u[p] > 0.0 { -v } else { v }
    };
    let up = u[p] - vnorm;
    u[p] = vnorm;
    up
}

/// Givens rotation construction.
///
/// Given scalars `(sa, sb)`, computes the rotation parameters `(r, c, s)` such
/// that:
/// ```text
///     [ c  s ] [ sa ]   [ r ]
///     [-s  c ] [ sb ] = [ 0 ]
/// ```
///
/// # Returns
/// `(r, c, s)` — the resultant magnitude and cosine/sine of the rotation.
pub fn rotg(sa: f64, sb: f64) -> (f64, f64, f64) {
    let roe = if sa.abs() > sb.abs() { sa } else { sb };
    let scale = sa.abs() + sb.abs();
    if scale <= 0.0 {
        return (sa, 1.0, 0.0);
    }
    let r = scale * ((sa / scale).powi(2) + (sb / scale).powi(2)).sqrt();
    let r = if roe < 0.0 { -r } else { r };
    let c = sa / r;
    let s = sb / r;
    (r, c, s)
}

// ---------------------------------------------------------------------------
// BvlsResult
// ---------------------------------------------------------------------------

/// Result of the [`bvls`] solver.
pub struct BvlsResult {
    /// Solution vector `x` (length `n`).
    pub x: Vec<f64>,
    /// Residual norm `|| A x - b ||_2`.
    pub rnorm: f64,
    /// Number of variables in the free set P.
    #[allow(dead_code)]
    pub nsetp: usize,
    /// Dual vector / gradient of the unconstrained objective (length `n`).
    #[allow(dead_code)]
    pub w: Vec<f64>,
    /// Permutation index array (length `n`).
    /// Indices `0..nsetp` are in set P, the rest in set Z.
    #[allow(dead_code)]
    pub index: Vec<usize>,
    /// Error flag: 0 = success, 1 = trivial (m or n is 0),
    /// 3 = invalid bounds, 4 = iteration limit exceeded.
    pub ierr: i32,
}

// ---------------------------------------------------------------------------
// Main BVLS solver
// ---------------------------------------------------------------------------

/// Bounded Variable Least Squares solver.
///
/// Solves `min || A x - b ||_2` subject to element-wise bounds on `x`.
///
/// **Note:** `a` and `b` are modified in-place (the algorithm performs QR
/// factorisation on the active columns of `A`).
///
/// # Arguments
/// * `a`         — Coefficient matrix (m × n), modified in-place.
/// * `b`         — Right-hand side vector (length m), modified in-place.
/// * `bnd_lower` — Lower bounds on `x` (length n); use `-f64::MAX` for unbounded.
/// * `bnd_upper` — Upper bounds on `x` (length n); use `f64::MAX` for unbounded.
/// * `max_iter`  — Maximum iterations (0 → default `3 * n`).
///
/// # Returns
/// A [`BvlsResult`] containing the solution, residual norm, and diagnostics.
pub fn bvls(
    a: &mut ColMat,
    b: &mut [f64],
    bnd_lower: &[f64],
    bnd_upper: &[f64],
    max_iter: usize,
) -> BvlsResult {
    let m = a.rows;
    let n = a.cols;

    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut index: Vec<usize> = (0..n).collect();
    let mut s = vec![0.0; n];
    let mut z = vec![0.0; m];

    // Trivial case: empty problem.
    if m == 0 || n == 0 {
        return BvlsResult {
            x,
            rnorm: 0.0,
            nsetp: 0,
            w,
            index,
            ierr: 1,
        };
    }

    let itmax = if max_iter == 0 {
        3 * n
    } else {
        max_iter
    };
    let huge = f64::MAX;

    // iz1..=iz2 tracks the Z-set boundaries within `index`.
    let mut iz2 = n as isize - 1;
    let mut iz1: isize = 0;
    // nsetp = number of free variables in P (nsetp+1 = first Z-set position).
    let mut nsetp: isize = -1;
    let mut npp1: usize = 0;
    let mut ierr = 0_i32;
    let mut iteration = 0_usize;
    let mut up = 0.0_f64;

    // -----------------------------------------------------------------------
    // Initialize X: project each variable onto its feasible interval
    // and subtract A[:,j]*x[j] from b for each non-zero x[j].
    // -----------------------------------------------------------------------
    let mut iz = iz1;
    while iz <= iz2 {
        let j = index[iz as usize];
        if bnd_lower[j] <= -huge {
            if bnd_upper[j] >= huge {
                x[j] = 0.0;
            } else {
                x[j] = bnd_upper[j].min(0.0);
            }
        } else if bnd_upper[j] >= huge {
            x[j] = bnd_lower[j].max(0.0);
        } else {
            let rng = bnd_upper[j] - bnd_lower[j];
            if rng <= 0.0 {
                // Fixed variable: move to end of Z-set.
                index[iz as usize] = index[iz2 as usize];
                index[iz2 as usize] = j;
                iz -= 1;
                iz2 -= 1;
                x[j] = bnd_lower[j];
                w[j] = 0.0;
            } else if rng > 0.0 {
                x[j] = bnd_lower[j].max(bnd_upper[j].min(0.0));
            } else {
                // Invalid bounds (NaN range).
                return BvlsResult {
                    x,
                    rnorm: 0.0,
                    nsetp: 0,
                    w,
                    index,
                    ierr: 3,
                };
            }
        }
        // Adjust b for the initial x[j] contribution.
        if x[j].abs() > 0.0 {
            for i in 0..m {
                b[i] -= a[(i, j)] * x[j];
            }
        }
        iz += 1;
    }

    // -----------------------------------------------------------------------
    // Main iteration loop — select variables to enter/leave the free set P
    // -----------------------------------------------------------------------
    loop {
        if ierr != 0 || iz1 > iz2 || nsetp >= m as isize {
            break;
        }

        // --- Select a variable from Z to move into P ---
        let mut find = false;
        let mut sel_j = 0_usize;
        let mut sel_iz = 0_isize;

        for iz_idx in iz1..=iz2 {
            let j = index[iz_idx as usize];
            let free1 = x[j] > bnd_lower[j]; // not at lower bound
            let free2 = x[j] < bnd_upper[j]; // not at upper bound
            let free = free1 && free2; // strictly interior

            if free {
                // Variable is interior — test if column is sufficiently
                // independent to enter the factorisation.
                let asave = a[(npp1, j)];
                let col: Vec<f64> = (0..m).map(|i| a[(i, j)]).collect();
                let mut col_mut = col.clone();
                let up_val = htc(npp1, &mut col_mut);
                for i in 0..m {
                    a[(i, j)] = col_mut[i];
                }
                let unorm = nrm2(&col_mut[..npp1]);

                if a[(npp1, j)].abs() > EPS * unorm {
                    // Column is independent — trial solve.
                    z[..m].copy_from_slice(&b[..m]);
                    let norm_val = a[(npp1, j)];
                    a[(npp1, j)] = up_val;
                    // Apply Householder to z.
                    if norm_val.abs() > 0.0 {
                        let mut sm = 0.0;
                        for i in npp1..m {
                            sm += (a[(i, j)] / norm_val) * z[i];
                        }
                        sm /= up_val;
                        for i in npp1..m {
                            z[i] += sm * a[(i, j)];
                        }
                    }
                    a[(npp1, j)] = norm_val;

                    // Add back the x[j] contribution to upper rows of z.
                    if x[j].abs() > 0.0 {
                        for i in 0..=npp1 {
                            z[i] += a[(i, j)] * x[j];
                        }
                    }
                    find = true;
                }

                if !find {
                    // Restore column — insufficient independence.
                    for i in 0..m {
                        a[(i, j)] = col[i];
                    }
                    a[(npp1, j)] = asave;
                    w[j] = 0.0;
                } else {
                    sel_j = j;
                    sel_iz = iz_idx;
                    up = up_val;
                    break;
                }
            } else {
                // Variable is at a bound — compute dual coefficient w[j].
                let mut wj = 0.0;
                for l in npp1..m {
                    wj += a[(l, j)] * b[l];
                }
                w[j] = wj;

                // Only test if the dual suggests moving away from the bound.
                let do_test = (w[j] < 0.0 && free1) || (w[j] > 0.0 && free2);
                if do_test {
                    // Test column independence (same procedure as above).
                    let asave = a[(npp1, j)];
                    let col: Vec<f64> = (0..m).map(|i| a[(i, j)]).collect();
                    let mut col_mut = col.clone();
                    let up_val = htc(npp1, &mut col_mut);
                    for i in 0..m {
                        a[(i, j)] = col_mut[i];
                    }
                    let unorm = nrm2(&col_mut[..npp1]);

                    if a[(npp1, j)].abs() > EPS * unorm {
                        z[..m].copy_from_slice(&b[..m]);
                        let norm_val = a[(npp1, j)];
                        a[(npp1, j)] = up_val;
                        if norm_val.abs() > 0.0 {
                            let mut sm = 0.0;
                            for i in npp1..m {
                                sm += (a[(i, j)] / norm_val) * z[i];
                            }
                            sm /= up_val;
                            for i in npp1..m {
                                z[i] += sm * a[(i, j)];
                            }
                        }
                        a[(npp1, j)] = norm_val;

                        if x[j].abs() > 0.0 {
                            for i in 0..=npp1 {
                                z[i] += a[(i, j)] * x[j];
                            }
                        }

                        // Check that the trial solution moves in the right direction.
                        let ztest = z[npp1] / a[(npp1, j)];
                        find = (w[j] < 0.0 && ztest < x[j]) || (w[j] > 0.0 && ztest > x[j]);
                    }

                    if !find {
                        for i in 0..m {
                            a[(i, j)] = col[i];
                        }
                        a[(npp1, j)] = asave;
                        w[j] = 0.0;
                    } else {
                        sel_j = j;
                        sel_iz = iz_idx;
                        up = up_val;
                        break;
                    }
                }
            }
        }

        if !find {
            break; // No variable can improve — optimal.
        }

        // --- Move selected variable j from Z-set to P-set ---
        {
            let j = sel_j;
            b[..m].copy_from_slice(&z[..m]);
            index[sel_iz as usize] = index[iz1 as usize];
            index[iz1 as usize] = j;
            iz1 += 1;
            nsetp = npp1 as isize;
            npp1 += 1;

            // Apply Householder to remaining Z-columns.
            let norm_val = a[(nsetp as usize, j)];
            a[(nsetp as usize, j)] = up;
            if norm_val.abs() > 0.0 {
                for jz in iz1..=iz2 {
                    let jj = index[jz as usize];
                    let mut sm = 0.0;
                    for i in (nsetp as usize)..m {
                        sm += (a[(i, j)] / norm_val) * a[(i, jj)];
                    }
                    sm /= up;
                    for i in (nsetp as usize)..m {
                        a[(i, jj)] += sm * a[(i, j)];
                    }
                }
            }
            a[(nsetp as usize, j)] = norm_val;

            // Zero out sub-diagonal entries in column j.
            for i in npp1..m {
                a[(i, j)] = 0.0;
            }
            w[j] = 0.0;

            // Solve the upper-triangular system for z.
            let mut ii = 0_usize;
            for i in (0..=(nsetp as usize)).rev() {
                if i != nsetp as usize {
                    for k in 0..=i {
                        z[k] -= a[(k, ii)] * z[i + 1];
                    }
                }
                ii = index[i];
                z[i] /= a[(i, ii)];
            }
        }

        // --- Test the P-set solution against bound constraints ---
        loop {
            iteration += 1;
            if iteration > itmax {
                ierr = 4;
                break;
            }

            // Find the most constraining bound violation (if any).
            let mut alpha = 2.0;
            let mut jj: usize = 0;
            let mut ibound: isize = -1;
            for ip in 0..=(nsetp as usize) {
                let l_idx = index[ip];
                let lbound = if z[ip] <= bnd_lower[l_idx] {
                    0_isize // violates lower bound
                } else if z[ip] >= bnd_upper[l_idx] {
                    1_isize // violates upper bound
                } else {
                    -1_isize // feasible
                };
                if lbound >= 0 {
                    let bnd_val = if lbound == 0 {
                        bnd_lower[l_idx]
                    } else {
                        bnd_upper[l_idx]
                    };
                    let t = (bnd_val - x[l_idx]) / (z[ip] - x[l_idx]);
                    if alpha > t {
                        alpha = t;
                        jj = ip;
                        ibound = lbound;
                    }
                }
            }
            let hitbnd = (alpha - 2.0).abs() > 0.0;

            if !hitbnd {
                break; // All P-set variables are feasible.
            }

            // Interpolate x towards z (step alpha < 1).
            for ip in 0..=(nsetp as usize) {
                let l_idx = index[ip];
                x[l_idx] += alpha * (z[ip] - x[l_idx]);
            }

            let mut i_var = index[jj];

            // Move infeasible variables from P back to Z.
            loop {
                {
                    let bnd_val = if ibound == 0 {
                        bnd_lower[i_var]
                    } else {
                        bnd_upper[i_var]
                    };
                    x[i_var] = bnd_val;

                    // Subtract contribution from b for the fixed variable.
                    if x[i_var].abs() > 0.0 {
                        for k in 0..=jj {
                            b[k] -= a[(k, i_var)] * x[i_var];
                        }
                    }

                    // Apply Givens rotations to restore triangular form.
                    for j in jj..(nsetp as usize) {
                        let ii = index[j + 1];
                        index[j] = ii;
                        let (sm, cc, ss) = rotg(a[(j, ii)], a[(j + 1, ii)]);
                        a[(j, ii)] = sm;
                        // Apply plane rotation to rows j and j+1.
                        let row_j = a.row(j);
                        s[..n].copy_from_slice(&row_j);
                        for k in 0..n {
                            a[(j, k)] = cc * s[k] + ss * a[(j + 1, k)];
                            a[(j + 1, k)] = cc * a[(j + 1, k)] - ss * s[k];
                        }
                        a[(j, ii)] = sm;
                        a[(j + 1, ii)] = 0.0;
                        // Rotate the right-hand side b as well.
                        let sm_b = b[j];
                        b[j] = cc * sm_b + ss * b[j + 1];
                        b[j + 1] = cc * b[j + 1] - ss * sm_b;
                    }

                    // Move variable from P to Z.
                    npp1 = nsetp as usize;
                    nsetp -= 1;
                    iz1 -= 1;
                    index[iz1 as usize] = i_var;
                }

                if nsetp < 0 {
                    break;
                }

                // Check if any remaining P-variable violates its bounds.
                ibound = -1;
                for jj2 in 0..=(nsetp as usize) {
                    let i2 = index[jj2];
                    if x[i2] <= bnd_lower[i2] {
                        ibound = 0;
                        i_var = i2;
                        jj = jj2;
                        break;
                    } else if x[i2] >= bnd_upper[i2] {
                        ibound = 1;
                        i_var = i2;
                        jj = jj2;
                        break;
                    }
                }
                if ibound < 0 {
                    break; // All remaining P-variables are feasible.
                }
            }

            if nsetp < 0 {
                break;
            }

            // Re-solve the triangular system with the reduced P-set.
            z[..m].copy_from_slice(&b[..m]);
            for i in (0..=(nsetp as usize)).rev() {
                if i != nsetp as usize {
                    let ii = index[i + 1];
                    for k in 0..=i {
                        z[k] -= a[(k, ii)] * z[i + 1];
                    }
                }
                let ii = index[i];
                z[i] /= a[(i, ii)];
            }
        }

        // Copy the feasible solution from z back into x.
        for ip in 0..npp1 {
            let i_idx = index[ip];
            x[i_idx] = z[ip];
        }
    }

    // -----------------------------------------------------------------------
    // Termination — compute residual norm
    // -----------------------------------------------------------------------
    let mut rnorm = 0.0;
    if ierr <= 0 {
        if npp1 < m {
            rnorm = nrm2(&b[npp1..m]);
        } else {
            w.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    BvlsResult {
        x,
        rnorm,
        nsetp: (nsetp + 1).max(0) as usize,
        w,
        index,
        ierr,
    }
}

// ---------------------------------------------------------------------------
// NNLS convenience wrapper
// ---------------------------------------------------------------------------

/// Non-Negative Least Squares (NNLS) wrapper around [`bvls`].
///
/// Solves `min || A x - b ||_2` subject to `x >= 0` by calling [`bvls`] with
/// `bnd_lower = 0` and `bnd_upper = f64::MAX` for all variables.
///
/// # Arguments
/// * `a`        — Coefficient matrix (m × n), modified in-place.
/// * `_m`       — Number of rows (unused — taken from `a.rows`).
/// * `n`        — Number of columns / variables.
/// * `b`        — Right-hand side (length m), modified in-place.
/// * `max_iter` — Maximum iterations (0 → default).
///
/// # Returns
/// `(x, rnorm, mode)` where `mode` follows the NNLS convention:
/// 1 = success, 2 = trivial, 3 = iteration limit exceeded.
pub fn bvls_wrapper(
    a: &mut ColMat,
    _m: usize,
    n: usize,
    b: &mut [f64],
    max_iter: usize,
) -> (Vec<f64>, f64, i32) {
    let bnd_lower = vec![0.0; n];
    let bnd_upper = vec![f64::MAX; n];

    let result = bvls(a, b, &bnd_lower, &bnd_upper, max_iter);

    // Map BVLS error codes to NNLS-style mode values.
    let mode = match result.ierr {
        0 => 1,     // success
        1 | 2 => 2, // trivial (empty problem or degenerate)
        3 => -999,  // invalid bounds (should not occur for NNLS)
        4 => 3,     // iteration limit exceeded
        _ => -9999, // unexpected
    };

    (result.x, result.rnorm, mode)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // -- htc (Householder transform construction) ----------------------------

    #[test]
    fn htc_basic_reflector() {
        let mut u = vec![3.0, 1.0, 1.0, 1.0];
        let up = htc(0, &mut u);

        // u[0] should be the signed norm of the original vector
        let orig_norm = (9.0 + 1.0 + 1.0 + 1.0_f64).sqrt();
        assert!((u[0].abs() - orig_norm).abs() < TOL);
        // up is original_pivot - signed_norm
        assert!((up - (3.0 - u[0])).abs() < TOL);
    }

    #[test]
    fn htc_single_element() {
        let mut u = vec![5.0];
        let up = htc(0, &mut u);
        // For a single element, norm = |u[0]|, signed to oppose sign
        assert!((u[0].abs() - 5.0).abs() < TOL);
        assert!((up - (5.0 - u[0])).abs() < TOL);
    }

    #[test]
    fn htc_pivot_in_middle() {
        let mut u = vec![0.0, 0.0, 3.0, 4.0];
        let up = htc(2, &mut u);
        let norm = (9.0 + 16.0_f64).sqrt();
        assert!((u[2].abs() - norm).abs() < TOL);
        assert!((up - (3.0 - u[2])).abs() < TOL);
    }

    #[test]
    fn htc_negative_pivot() {
        let mut u = vec![-3.0, 4.0];
        let up = htc(0, &mut u);
        let norm = (9.0 + 16.0_f64).sqrt();
        assert!((u[0].abs() - norm).abs() < TOL);
        // Sign chosen opposite to original pivot (-3), so u[0] should be positive
        assert!(u[0] > 0.0);
        assert!((up - (-3.0 - u[0])).abs() < TOL);
    }

    // -- rotg (Givens rotation construction) ---------------------------------

    #[test]
    fn rotg_zeroes_second_element() {
        let (r, c, s) = rotg(3.0, 4.0);
        // c*sa + s*sb = r,  -s*sa + c*sb = 0
        assert!((c * 3.0 + s * 4.0 - r).abs() < TOL);
        assert!((-s * 3.0 + c * 4.0).abs() < TOL);
    }

    #[test]
    fn rotg_hypotenuse() {
        let (r, _, _) = rotg(3.0, 4.0);
        assert!((r.abs() - 5.0).abs() < TOL);
    }

    #[test]
    fn rotg_both_zero() {
        let (r, c, s) = rotg(0.0, 0.0);
        assert_eq!(r, 0.0);
        assert_eq!(c, 1.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn rotg_sa_dominant() {
        let (r, c, s) = rotg(10.0, 1.0);
        assert!(r > 0.0); // sign follows sa (dominant)
        assert!((c * 10.0 + s * 1.0 - r).abs() < TOL);
        assert!((-s * 10.0 + c * 1.0).abs() < TOL);
    }

    #[test]
    fn rotg_sb_dominant() {
        let (r, c, s) = rotg(1.0, 10.0);
        assert!(r > 0.0); // sign follows sb (dominant)
        assert!((c * 1.0 + s * 10.0 - r).abs() < TOL);
        assert!((-s * 1.0 + c * 10.0).abs() < TOL);
    }

    #[test]
    fn rotg_negative_values() {
        let (r, c, s) = rotg(-3.0, -4.0);
        assert!(r < 0.0); // both negative → roe is negative
        assert!((c * -3.0 + s * -4.0 - r).abs() < TOL);
        assert!((-s * -3.0 + c * -4.0).abs() < TOL);
    }

    #[test]
    fn rotg_is_orthogonal() {
        let (_, c, s) = rotg(7.0, 11.0);
        assert!((c * c + s * s - 1.0).abs() < TOL);
    }

    // -- bvls (Bounded Variable Least Squares) -------------------------------

    #[test]
    fn bvls_identity_unconstrained() {
        // min || I*x - b || with wide bounds → x = b
        let mut a = ColMat::from_vv(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let mut b = vec![1.0, 2.0, 3.0];
        let bnd_lower = vec![-100.0; 3];
        let bnd_upper = vec![100.0; 3];

        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!(result.rnorm < TOL);
        assert!((result.x[0] - 1.0).abs() < TOL);
        assert!((result.x[1] - 2.0).abs() < TOL);
        assert!((result.x[2] - 3.0).abs() < TOL);
    }

    #[test]
    fn bvls_active_lower_bound() {
        // min || I*x - [-5, 3] || s.t. x >= 0
        // Solution: x = [0, 3]
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![-5.0, 3.0];
        let bnd_lower = vec![0.0, 0.0];
        let bnd_upper = vec![f64::MAX, f64::MAX];

        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!(result.x[0].abs() < TOL); // clamped at lower bound
        assert!((result.x[1] - 3.0).abs() < TOL);
    }

    #[test]
    fn bvls_active_upper_bound() {
        // min || I*x - [10, 3] || s.t. x <= 5
        // Solution: x = [5, 3]
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![10.0, 3.0];
        let bnd_lower = vec![-f64::MAX, -f64::MAX];
        let bnd_upper = vec![5.0, 5.0];

        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - 5.0).abs() < TOL);
        assert!((result.x[1] - 3.0).abs() < TOL);
    }

    #[test]
    fn bvls_overdetermined() {
        // min || [[1],[1]] * x - [1, 3] || s.t. 0 <= x <= 10
        // LS solution: x = 2 (average), which is in bounds
        let mut a = ColMat::from_vv(&[vec![1.0], vec![1.0]]);
        let mut b = vec![1.0, 3.0];
        let bnd_lower = vec![0.0];
        let bnd_upper = vec![10.0];

        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - 2.0).abs() < TOL);
    }

    #[test]
    fn bvls_empty_problem() {
        let mut a = ColMat::zeros(0, 0);
        let mut b = vec![];
        let result = bvls(&mut a, &mut b, &[], &[], 0);
        assert_eq!(result.ierr, 1); // trivial
    }

    #[test]
    fn bvls_fixed_variable() {
        // bnd_lower = bnd_upper → variable is fixed
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![5.0, 3.0];
        let bnd_lower = vec![2.0, -f64::MAX]; // x[0] fixed at 2
        let bnd_upper = vec![2.0, f64::MAX];

        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - 2.0).abs() < TOL); // fixed
        assert!((result.x[1] - 3.0).abs() < TOL); // free
    }

    // -- bvls_wrapper (NNLS interface) ---------------------------------------

    #[test]
    fn bvls_wrapper_identity_positive_rhs() {
        // Same as NNLS: min ||I*x - b|| s.t. x >= 0
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![3.0, 7.0];
        let (x, rnorm, mode) = bvls_wrapper(&mut a, 2, 2, &mut b, 0);
        assert_eq!(mode, 1); // success
        assert!(rnorm < TOL);
        assert!((x[0] - 3.0).abs() < TOL);
        assert!((x[1] - 7.0).abs() < TOL);
    }

    #[test]
    fn bvls_wrapper_clamps_negative() {
        // min ||I*x - [-2, 5]|| s.t. x >= 0 → x = [0, 5]
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![-2.0, 5.0];
        let (x, _rnorm, mode) = bvls_wrapper(&mut a, 2, 2, &mut b, 0);
        assert_eq!(mode, 1);
        assert!(x[0].abs() < TOL);
        assert!((x[1] - 5.0).abs() < TOL);
    }

    #[test]
    fn bvls_wrapper_empty_returns_trivial() {
        let mut a = ColMat::zeros(0, 0);
        let mut b = vec![];
        let (_x, _rnorm, mode) = bvls_wrapper(&mut a, 0, 0, &mut b, 0);
        assert_eq!(mode, 2); // trivial
    }

    // -- ported from Python TestBvlsNNLS -------------------------------------

    #[test]
    fn bvls_nnls_identity_negative_rhs() {
        // A = I, b = [-1, -2], lower = 0 → x = [0, 0]
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![-1.0, -2.0];
        let bnd_lower = vec![0.0, 0.0];
        let bnd_upper = vec![f64::MAX, f64::MAX];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!(result.x[0].abs() < TOL);
        assert!(result.x[1].abs() < TOL);
    }

    #[test]
    fn bvls_nnls_identity_mixed_rhs() {
        // A = I, b = [3, -1], lower = 0 → x = [3, 0]
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![3.0, -1.0];
        let bnd_lower = vec![0.0, 0.0];
        let bnd_upper = vec![f64::MAX, f64::MAX];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - 3.0).abs() < TOL);
        assert!(result.x[1].abs() < TOL);
    }

    #[test]
    fn bvls_nnls_overdetermined_3x2() {
        // 3×2 overdetermined NNLS → both components positive
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        let mut b = vec![1.0, 1.0, 3.0];
        let bnd_lower = vec![0.0, 0.0];
        let bnd_upper = vec![f64::MAX, f64::MAX];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!(result.x[0] > 0.0);
        assert!(result.x[1] > 0.0);
    }

    #[test]
    fn bvls_nnls_single_variable() {
        // 1×1 system: 2x = 6 → x = 3
        let mut a = ColMat::from_vv(&[vec![2.0]]);
        let mut b = vec![6.0];
        let bnd_lower = vec![0.0];
        let bnd_upper = vec![f64::MAX];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - 3.0).abs() < TOL);
    }

    // -- ported from Python TestBvlsBounded ----------------------------------

    #[test]
    fn bvls_upper_bound_active_both() {
        // Both variables clamped by upper = 5
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![10.0, 10.0];
        let bnd_lower = vec![0.0, 0.0];
        let bnd_upper = vec![5.0, 5.0];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - 5.0).abs() < TOL);
        assert!((result.x[1] - 5.0).abs() < TOL);
    }

    #[test]
    fn bvls_lower_bound_active_negative() {
        // Both variables clamped by lower = -5
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![-10.0, -10.0];
        let bnd_lower = vec![-5.0, -5.0];
        let bnd_upper = vec![100.0, 100.0];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - (-5.0)).abs() < TOL);
        assert!((result.x[1] - (-5.0)).abs() < TOL);
    }

    #[test]
    fn bvls_fixed_and_free_mixed() {
        // x[0] fixed at 3, x[1] free → x = [3, 10]
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![10.0, 10.0];
        let bnd_lower = vec![3.0, 0.0];
        let bnd_upper = vec![3.0, 100.0];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - 3.0).abs() < TOL);
        assert!((result.x[1] - 10.0).abs() < TOL);
    }

    #[test]
    fn bvls_unconstrained_huge_bounds() {
        // With huge bounds, result matches ordinary LS
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![3.0, 7.0];
        let huge = f64::MAX;
        let bnd_lower = vec![-huge, -huge];
        let bnd_upper = vec![huge, huge];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - 3.0).abs() < TOL);
        assert!((result.x[1] - 7.0).abs() < TOL);
    }

    #[test]
    fn bvls_negative_bounds() {
        // Bounds entirely in negative range
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![-3.0, -7.0];
        let bnd_lower = vec![-10.0, -10.0];
        let bnd_upper = vec![-1.0, -1.0];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 0);
        assert_eq!(result.ierr, 0);
        assert!((result.x[0] - (-3.0)).abs() < TOL);
        assert!((result.x[1] - (-7.0)).abs() < TOL);
    }

    // -- ported from Python TestBvlsEdgeCases --------------------------------

    #[test]
    fn bvls_max_iter_exceeded() {
        // Very tight iteration limit should trigger ierr=4 or early success
        let mut a = ColMat::from_vv(&[vec![1.0, 1.0], vec![1.0, -1.0], vec![0.0, 1.0]]);
        let mut b = vec![2.0, 0.0, 1.0];
        let bnd_lower = vec![0.0, 0.0];
        let bnd_upper = vec![10.0, 10.0];
        let result = bvls(&mut a, &mut b, &bnd_lower, &bnd_upper, 1);
        assert!(result.ierr == 0 || result.ierr == 4);
    }

    // -- ported from Python TestBvlsWrapper ----------------------------------

    #[test]
    fn bvls_wrapper_overdetermined() {
        // 3×2 overdetermined, positive solution
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        let mut b = vec![1.0, 1.0, 3.0];
        let (x, _rnorm, mode) = bvls_wrapper(&mut a, 3, 2, &mut b, 0);
        assert_eq!(mode, 1);
        assert!(x[0] >= 0.0);
        assert!(x[1] >= 0.0);
    }

    // -- scipy cross-checks (hardcoded reference values) ---------------------

    #[test]
    fn bvls_vs_scipy_3x2() {
        let mut a = ColMat::from_vv(&[
            vec![-0.9891213503478509, -0.3677866514678832],
            vec![1.2879252612892487, 0.1939744191326132],
            vec![0.9202308996398569, 0.5771037912572513],
        ]);
        let mut b = vec![-0.6364636463709805, 0.5419522204102933, -0.3165954511658161];
        let (x, rnorm, mode) = bvls_wrapper(&mut a, 3, 2, &mut b, 0);
        assert_eq!(mode, 1);
        let x_ref = [0.2974200425504605, 0.0];
        let rnorm_ref = 0.7006042745870632;
        assert!((x[0] - x_ref[0]).abs() < 1e-8);
        assert!((x[1] - x_ref[1]).abs() < 1e-8);
        assert!((rnorm - rnorm_ref).abs() < 1e-8);
    }

    #[test]
    fn bvls_vs_scipy_5x3() {
        let mut a = ColMat::from_vv(&[
            vec![0.8960399523648943, -2.1239485391809105, 1.6402038096847686],
            vec![
                -0.03530739774353654,
                -2.6027652074064505,
                -0.190065252686386,
            ],
            vec![
                -0.6641100114991543,
                -0.4147027402236024,
                -0.29244522068809464,
            ],
            vec![1.9450048780036302, 1.0060206492356876, 0.5363991669960861],
            vec![-0.3804561025507811, 0.587488775812378, 1.8558757664419043],
        ]);
        let mut b = vec![
            1.009243203146077,
            0.8104351593717798,
            -0.8661760953263219,
            1.7410657168317318,
            0.3846901361137855,
        ];
        let (x, rnorm, mode) = bvls_wrapper(&mut a, 5, 3, &mut b, 0);
        assert_eq!(mode, 1);
        let x_ref = [0.8003229024292042, 0.0, 0.2743869833118082];
        let rnorm_ref = 0.9576266903169139;
        for i in 0..3 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1e-8,
                "x[{}]: {} vs {}",
                i,
                x[i],
                x_ref[i]
            );
        }
        assert!((rnorm - rnorm_ref).abs() < 1e-8);
    }

    #[test]
    fn bvls_vs_scipy_10x4() {
        let mut a = ColMat::from_vv(&[
            vec![
                0.6602111215733654,
                -1.7658717082572897,
                -0.08039937824510862,
                0.42379105297090075,
            ],
            vec![
                -0.03943571145431186,
                1.859978061053346,
                0.8084189852562238,
                1.294924772965804,
            ],
            vec![
                0.40064886296312435,
                1.2483681654575487,
                -1.8190229373004365,
                0.3343774385974276,
            ],
            vec![
                1.772505055189508,
                -0.2729861000297362,
                -1.1304767942481746,
                -1.6182725311847022,
            ],
            vec![
                -2.320006175714351,
                0.2992449430377799,
                -2.103252256991572,
                -0.6410700376162922,
            ],
            vec![
                0.5090855498052809,
                -0.4662713948813545,
                1.7244328113638232,
                -0.5860223101488526,
            ],
            vec![
                1.3742071148923605,
                -0.27557311884013275,
                0.39893508374644254,
                0.36449069713861354,
            ],
            vec![
                -0.0425004719249466,
                -0.22772657112153394,
                -0.829529496092918,
                -0.5888974548659531,
            ],
            vec![
                0.5575527761962881,
                0.5588656112239981,
                0.9123496909988327,
                0.3085386161787194,
            ],
            vec![
                -1.5367945514830914,
                -0.5754474916426862,
                0.3312131623584405,
                0.2965348786765032,
            ],
        ]);
        let mut b = vec![
            -0.011947213526250196,
            -0.16441058531830322,
            -0.11702559038010149,
            0.5724069571304163,
            0.7830102723689154,
            0.08090692075511713,
            -2.3404724718946457,
            -1.6897054927639186,
            0.10495330976468087,
            -1.2749864696481292,
        ];
        let (x, rnorm, mode) = bvls_wrapper(&mut a, 10, 4, &mut b, 0);
        assert_eq!(mode, 1);
        let x_ref = [0.0, 0.15412208892010765, 0.0, 0.0];
        let rnorm_ref = 3.2767428729048436;
        for i in 0..4 {
            assert!(
                (x[i] - x_ref[i]).abs() < 1e-8,
                "x[{}]: {} vs {}",
                i,
                x[i],
                x_ref[i]
            );
        }
        assert!((rnorm - rnorm_ref).abs() < 1e-8);
    }
}
