//! Main SLSQP routines: line-search, SQP iteration, and top-level entry point.
//!
//! Port of the `slsqpb` section of `slsqp_core.f90`.  Contains:
//!
//! - [`linmin`]  — Derivative-free line-search (Brent's method).
//! - [`slsqpb`] — Main SQP iteration with reverse communication.
//! - [`SlsqpWorkspace`] — Pre-allocated workspace for the SLSQP solver.
//! - [`slsqp_step`] — Top-level entry point using `SlsqpWorkspace`.
//!
//! The reverse-communication protocol works as follows:
//!
//! 1. Caller sets `mode = 0` for initialisation, then calls `slsqp_step`.
//! 2. `slsqp_step` returns `mode = 1` → caller must evaluate `f(x)`, `c(x)`,
//!    `g(x)`, `a(x)` (objective, constraints, gradient, Jacobian).
//! 3. Caller calls `slsqp_step` again with the new function values.
//! 4. Repeat until `mode = 0` (converged) or `mode > 0` (error).
//! 5. `mode < 0` is an internal signal requesting Jacobian re-evaluation.

use crate::core_basic::*;
use crate::core_ls::lsq_ws;
use crate::core_types::*;
use crate::support::*;

// ---------------------------------------------------------------------------
// Constants for Brent's line-search
// ---------------------------------------------------------------------------

/// Golden-section ratio `(3 - sqrt(5)) / 2 ≈ 0.381966`.
const LINMIN_C: f64 = (3.0 - 2.23606797749979) / 2.0;

/// `sqrt(f64::EPSILON) ≈ 1.49e-8` — tolerance scaling for line-search.
const LINMIN_SQRTEPS: f64 = 1.4901161193847656e-08;

// ===========================================================================
// Linmin — Brent's derivative-free line-search
// ===========================================================================

/// Derivative-free line-search using Brent's method (golden section with
/// parabolic interpolation).
///
/// This is a reverse-communication routine:
/// - **`mode = 0`** (initial call): initialise the bracket `[ax, bx]` and
///   return the first trial point.  Returns `(x, 1)`.
/// - **`mode = 1`**: the caller has evaluated `f` at the first trial point.
///   Update internal state and return the next trial point.  Returns `(u, 2)`.
/// - **`mode = 2`**: the caller has evaluated `f` at `u`.  Update the bracket
///   and return the next trial point or declare convergence.
///   Returns `(u, 2)` or `(x_best, 3)`.
/// - **`mode = 3`** (returned): converged — `x` is the minimiser.
///
/// # Arguments
/// * `mode` — Current mode (see above).
/// * `ax`   — Left endpoint of the initial bracket.
/// * `bx`   — Right endpoint of the initial bracket.
/// * `f`    — Function value at the current trial point.
/// * `tol`  — Absolute convergence tolerance.
/// * `ldat` — Persistent line-search state (modified in-place).
///
/// # Returns
/// `(trial_point, new_mode)` — next trial point and updated mode.
pub fn linmin(mode: i32, ax: f64, bx: f64, f: f64, tol: f64, ldat: &mut LinminData) -> (f64, i32) {
    let c = LINMIN_C;
    let sqrteps = LINMIN_SQRTEPS;

    if mode == 1 {
        // First function evaluation received — initialise function values.
        ldat.fx = f;
        ldat.fv = ldat.fx;
        ldat.fw = ldat.fv;
    } else if mode == 2 {
        // Subsequent evaluation at u — update bracket and best points.
        ldat.fu = f;
        if ldat.fu > ldat.fx {
            // u is worse than x — shrink bracket towards x.
            if ldat.u < ldat.x {
                ldat.a = ldat.u;
            }
            if ldat.u >= ldat.x {
                ldat.b = ldat.u;
            }
            if ldat.fu <= ldat.fw || (ldat.w - ldat.x).abs() <= ZERO {
                ldat.v = ldat.w;
                ldat.fv = ldat.fw;
                ldat.w = ldat.u;
                ldat.fw = ldat.fu;
            } else if ldat.fu <= ldat.fv
                || (ldat.v - ldat.x).abs() <= ZERO
                || (ldat.v - ldat.w).abs() <= ZERO
            {
                ldat.v = ldat.u;
                ldat.fv = ldat.fu;
            }
        } else {
            // u is better than (or equal to) x — update best point.
            if ldat.u >= ldat.x {
                ldat.a = ldat.x;
            }
            if ldat.u < ldat.x {
                ldat.b = ldat.x;
            }
            ldat.v = ldat.w;
            ldat.fv = ldat.fw;
            ldat.w = ldat.x;
            ldat.fw = ldat.fx;
            ldat.x = ldat.u;
            ldat.fx = ldat.fu;
        }
    } else {
        // Initialisation (mode == 0 or any other value).
        ldat.a = ax;
        ldat.b = bx;
        ldat.e = ZERO;
        ldat.v = ldat.a + c * (ldat.b - ldat.a);
        ldat.w = ldat.v;
        ldat.x = ldat.w;
        return (ldat.x, 1); // Request first function evaluation.
    }

    // --- Check convergence ---
    ldat.m = 0.5 * (ldat.a + ldat.b);
    ldat.tol1 = sqrteps * ldat.x.abs() + tol;
    ldat.tol2 = ldat.tol1 + ldat.tol1;

    if (ldat.x - ldat.m).abs() <= ldat.tol2 - 0.5 * (ldat.b - ldat.a) {
        return (ldat.x, 3); // Converged.
    }

    // --- Try parabolic interpolation ---
    ldat.r = ZERO;
    ldat.q = ldat.r;
    ldat.p = ldat.q;
    if ldat.e.abs() > ldat.tol1 {
        // Fit a parabola through (v, fv), (w, fw), (x, fx).
        ldat.r = (ldat.x - ldat.w) * (ldat.fx - ldat.fv);
        ldat.q = (ldat.x - ldat.v) * (ldat.fx - ldat.fw);
        ldat.p = (ldat.x - ldat.v) * ldat.q - (ldat.x - ldat.w) * ldat.r;
        ldat.q = ldat.q - ldat.r;
        ldat.q = ldat.q + ldat.q;
        if ldat.q > ZERO {
            ldat.p = -ldat.p;
        }
        if ldat.q < ZERO {
            ldat.q = -ldat.q;
        }
        ldat.r = ldat.e;
        ldat.e = ldat.d;
    }

    // --- Choose between golden section and parabolic step ---
    if ldat.p.abs() >= 0.5 * (ldat.q * ldat.r).abs()
        || ldat.p <= ldat.q * (ldat.a - ldat.x)
        || ldat.p >= ldat.q * (ldat.b - ldat.x)
    {
        // Golden-section step.
        if ldat.x >= ldat.m {
            ldat.e = ldat.a - ldat.x;
        }
        if ldat.x < ldat.m {
            ldat.e = ldat.b - ldat.x;
        }
        ldat.d = c * ldat.e;
    } else {
        // Parabolic interpolation step.
        ldat.d = ldat.p / ldat.q;
        // Safeguard: ensure u stays within [a + tol2, b - tol2].
        if ldat.u - ldat.a < ldat.tol2 {
            ldat.d = if ldat.m >= ldat.x {
                ldat.tol1.abs()
            } else {
                -ldat.tol1.abs()
            };
        }
        if ldat.b - ldat.u < ldat.tol2 {
            ldat.d = if ldat.m >= ldat.x {
                ldat.tol1.abs()
            } else {
                -ldat.tol1.abs()
            };
        }
    }

    // Ensure the step is at least tol1 in magnitude.
    if ldat.d.abs() < ldat.tol1 {
        ldat.d = if ldat.d >= 0.0 {
            ldat.tol1.abs()
        } else {
            -ldat.tol1.abs()
        };
    }
    ldat.u = ldat.x + ldat.d;
    (ldat.u, 2) // Request function evaluation at u.
}

// ===========================================================================
// Helper functions for slsqpb
// ===========================================================================

/// Reset the BFGS Hessian approximation to the identity matrix.
///
/// If more than 5 resets have occurred (`ireset > 5`), performs a convergence
/// check instead and returns `mode = 8` (or converged `mode = 0`).
/// Otherwise, zeroes the packed L and sets diagonal elements to 1.
///
/// # Returns
/// `(acc, iter_, mode)` — the possibly-updated accuracy, iteration count,
/// and mode flag.
fn reset_bfgs(
    n: usize,
    n1: usize,
    n2: usize,
    l: &mut [f64],
    s: &[f64],
    acc: f64,
    sdat: &mut SlsqpbData,
    tolf: f64,
    toldf: f64,
    toldx: f64,
    f: f64,
    x: &[f64],
    x0: &[f64],
    c: &[f64],
    m: usize,
    meq: usize,
    _mu: &[f64],
    inconsistent_linearization: bool,
    iter_: usize,
) -> (f64, usize, i32) {
    sdat.ireset += 1;
    if sdat.ireset > 5 {
        // Too many resets — check convergence instead.
        let mut h3 = ZERO;
        for j in 0..m {
            let h1 = if j < meq { c[j] } else { ZERO };
            h3 += (-c[j]).max(h1);
        }
        let mode = check_convergence(
            n,
            f,
            sdat.f0,
            x,
            x0,
            s,
            h3,
            sdat.tol,
            tolf,
            toldf,
            toldx,
            0,
            8,
            inconsistent_linearization,
        );
        (acc, iter_, mode)
    } else {
        // Reset to identity: zero out packed L, then set diagonals to 1.
        for i in 0..n2 {
            l[i] = ZERO;
        }
        let mut j = 0;
        for i in 0..n {
            l[j] = ONE;
            j += n1 - i - 1;
        }
        (acc, iter_, -1) // Signal: request Jacobian re-evaluation.
    }
}

/// Perform one step of the inexact (Armijo-type) line-search.
///
/// Scales the search direction `s` by `alpha`, then computes the new trial
/// point `x = x0 + s` (with bounds enforcement).
fn inexact_linesearch(
    n: usize,
    x: &mut [f64],
    x0: &[f64],
    s: &mut [f64],
    sdat: &mut SlsqpbData,
    xl: &[f64],
    xu: &[f64],
    infbnd: f64,
) {
    sdat.line += 1;
    sdat.h3 = sdat.alpha * sdat.h3;
    dscal(n, sdat.alpha, s, 1);
    dcopy(n, x0, 1, x, 1);
    daxpy(n, ONE, s, 1, x, 1);
    enforce_bounds(x, xl, xu, infbnd);
}

/// Perform one step of the exact (Brent) line-search.
///
/// Calls [`linmin`] to determine the step length `alpha`, then computes the
/// trial point `x = x0 + alpha * s`.  When `linmin` returns converged
/// (`line == 3`), scales `s` by `alpha` for the final search direction.
fn exact_linesearch(
    n: usize,
    x: &mut [f64],
    x0: &[f64],
    s: &mut [f64],
    sdat: &mut SlsqpbData,
    ldat: &mut LinminData,
    t: f64,
    alphamin: f64,
    alphamax: f64,
    _xl: &[f64],
    _xu: &[f64],
    _infbnd: f64,
) {
    if sdat.line != 3 {
        let (alpha, new_mode) = linmin(sdat.line, alphamin, alphamax, t, sdat.tol, ldat);
        sdat.alpha = alpha;
        sdat.line = new_mode;
        dcopy(n, x0, 1, x, 1);
        daxpy(n, sdat.alpha, s, 1, x, 1);
    } else {
        // Linmin converged — finalise the search direction.
        dscal(n, sdat.alpha, s, 1);
    }
}

/// Compute the constraint violation `h3` and check convergence.
///
/// # Returns
/// `(acc, iter_, mode)` — unchanged `acc`/`iter_` plus the convergence mode.
fn convergence_check(
    acc: f64,
    converged: i32,
    not_converged: i32,
    n: usize,
    f: f64,
    f0: f64,
    x: &[f64],
    x0: &[f64],
    s: &[f64],
    c: &[f64],
    m: usize,
    meq: usize,
    _mu: &[f64],
    tol: f64,
    tolf: f64,
    toldf: f64,
    toldx: f64,
    inconsistent_linearization: bool,
    iter_: usize,
) -> (f64, usize, i32) {
    // Compute constraint violation: sum of max(0, -c_j) for inequalities,
    // |c_j| for equalities.
    let mut h3 = ZERO;
    for j in 0..m {
        let h1 = if j < meq { c[j] } else { ZERO };
        h3 += (-c[j]).max(h1);
    }
    let mode = check_convergence(
        n,
        f,
        f0,
        x,
        x0,
        s,
        h3,
        tol,
        tolf,
        toldf,
        toldx,
        converged,
        not_converged,
        inconsistent_linearization,
    );
    (acc, iter_, mode)
}

// ===========================================================================
// slsqpb — Main SQP iteration (reverse communication)
// ===========================================================================

/// Main SQP iteration with reverse communication.
///
/// This function implements one "step" of the SLSQP algorithm. The caller
/// invokes it repeatedly, providing function/gradient evaluations between
/// calls.  The algorithm state is carried in the workspace arrays and the
/// `sdat`/`ldat` structs.
///
/// # Reverse-communication protocol
///
/// | `mode` in | Meaning |
/// |-----------|---------|
/// | `0`       | Initialise — set up workspace, reset BFGS. |
/// | `1`       | Function evaluation received — check line-search. |
/// | `< 0`     | Jacobian evaluation received — update BFGS, start new QP. |
///
/// | `mode` out | Meaning |
/// |------------|---------|
/// | `0`        | Converged — `x` is the solution. |
/// | `1`        | Request function/gradient evaluation at current `x`. |
/// | `< 0`      | Request Jacobian re-evaluation (internal BFGS reset). |
/// | `2..9`     | Error or special condition (see Fortran documentation). |
///
/// # Arguments
/// * `m`, `meq`, `la`, `n` — Total constraints, equality constraints,
///   leading dimension of `a`, number of variables.
/// * `x`  — Current iterate (modified in-place).
/// * `xl`, `xu` — Variable bounds.
/// * `f`  — Current objective value.
/// * `c`  — Current constraint values (modified in-place).
/// * `g`  — Current gradient (modified in-place for augmented problem).
/// * `a`  — Current constraint Jacobian (modified in-place for augmented problem).
/// * `acc` — Convergence accuracy.
/// * `iter_` — Iteration counter.
/// * `mode` — Current reverse-communication mode.
/// * `r`, `l`, `x0`, `mu`, `s`, `u`, `v`, `_w` — Workspace arrays.
/// * `sdat`, `ldat` — Persistent iteration / line-search state.
/// * `alphamin`, `alphamax` — Step-length bounds.
/// * `tolf`, `toldf`, `toldx` — Optional convergence tolerances.
/// * `max_iter_ls` — Maximum iterations for the NNLS sub-solver.
/// * `nnls_mode` — 1 = NNLS, 2 = BVLS.
/// * `infbnd` — Threshold for "infinite" bounds.
///
/// # Returns
/// `(acc, iter_, mode)` — updated accuracy, iteration count, and mode.
pub fn slsqpb(
    m: usize,
    meq: usize,
    la: usize,
    n: usize,
    x: &mut [f64],
    xl: &[f64],
    xu: &[f64],
    f: f64,
    c: &mut [f64],
    g: &mut [f64],
    a: &mut ColMat,
    mut acc: f64,
    mut iter_: usize,
    mode: i32,
    r: &mut [f64],
    l: &mut [f64],
    x0: &mut [f64],
    mu: &mut [f64],
    s: &mut [f64],
    u: &mut [f64],
    v: &mut [f64],
    _w: &mut [f64],
    sdat: &mut SlsqpbData,
    ldat: &mut LinminData,
    alphamin: f64,
    alphamax: f64,
    tolf: f64,
    toldf: f64,
    toldx: f64,
    max_iter_ls: usize,
    nnls_mode: NnlsMode,
    infbnd: f64,
    ls_ws: &mut LsWorkspace,
) -> (f64, usize, i32) {
    profile_section!("slsqpb_total");

    let mut n1 = if sdat.n1 > 0 { sdat.n1 } else { n + 1 };
    let mut n2 = sdat.n2;
    let mut inconsistent_linearization = false;
    let mut mode = mode;

    // ===================================================================
    // Branch on incoming mode
    // ===================================================================

    if mode < 0 {
        // -----------------------------------------------------------
        // Jacobian evaluation received → BFGS update
        // -----------------------------------------------------------
        profile_section!("bfgs_update");

        // Compute u = g - A^T r - v  (gradient of the Lagrangian minus B*s).
        {
            profile_section!("bfgs_grad_lagrangian");
            for i in 0..n {
                let mut dot = 0.0;
                for j in 0..m {
                    dot += a[(j, i)] * r[j];
                }
                u[i] = g[i] - dot - v[i];
            }
        }

        // Compute v = B*s = L * D * L^T * s (three-pass multiplication).
        {
            profile_section!("bfgs_ldlt_multiply");

            // Try BLAS cblas_dtpmv for the three-pass multiply.
            #[cfg(feature = "blas")]
            let blas_ok = crate::lapack::bfgs_ldlt_multiply_blas(n, l, s, v);
            #[cfg(not(feature = "blas"))]
            let blas_ok = false;

            if !blas_ok {
                // Pass 1: v = L^T * s  (upper-triangular multiply).
                let mut k: usize = 0;
                for i in 0..n {
                    let mut h1 = ZERO;
                    // k points to diagonal[i] — skip it for off-diagonal entries
                    for j in (i + 1)..n {
                        k += 1;
                        h1 += l[k] * s[j];
                    }
                    v[i] = s[i] + h1;
                    k += 1;
                }

                // Pass 2: v = D * v  (diagonal scaling).
                let mut k: usize = 0;
                for i in 0..n {
                    v[i] = l[k] * v[i];
                    k += n1 - i - 1;
                }

                // Pass 3: v = L * v  (lower-triangular multiply).
                for i in (0..n).rev() {
                    let mut h1 = ZERO;
                    let mut k = i;
                    for j in 0..i {
                        h1 += l[k] * v[j];
                        k += n - j - 1;
                    }
                    v[i] += h1;
                }
            }
        }

        // Powell's modification of the BFGS update to ensure positive
        // definiteness.  h1 = s^T u, h2 = s^T v, h3 = 0.2 * h2.
        sdat.h1 = ddot(n, s, 1, u, 1);
        sdat.h2 = ddot(n, s, 1, v, 1);
        sdat.h3 = 0.2 * sdat.h2;
        if sdat.h1 < sdat.h3 {
            // Damped update: u := h4*u + (1-h4)*v.
            sdat.h4 = (sdat.h2 - sdat.h3) / (sdat.h2 - sdat.h1);
            sdat.h1 = sdat.h3;
            dscal(n, sdat.h4, u, 1);
            daxpy(n, ONE - sdat.h4, v, 1, u, 1);
        }

        if sdat.h1 == ZERO || sdat.h2 == ZERO {
            // Degenerate curvature — reset BFGS to identity.
            let result = reset_bfgs(
                n,
                n1,
                n2,
                l,
                s,
                acc,
                sdat,
                tolf,
                toldf,
                toldx,
                f,
                x,
                x0,
                c,
                m,
                meq,
                mu,
                inconsistent_linearization,
                iter_,
            );
            acc = result.0;
            iter_ = result.1;
            mode = result.2;
            if sdat.ireset > 5 {
                return (acc, iter_, mode);
            }
        } else {
            // Rank-one LDL^T updates: B := B + u u^T / h1 - v v^T / h2.
            profile_section!("bfgs_ldl_updates");
            ldl(n, l, u, ONE / sdat.h1, v);
            ldl(n, l, v, -ONE / sdat.h2, u);
        }
    } else if mode == 0 {
        // -----------------------------------------------------------
        // Initialisation
        // -----------------------------------------------------------
        sdat.itermx = iter_;
        sdat.iexact = if acc >= ZERO { 0 } else { 1 };
        acc = acc.abs();
        sdat.tol = TEN * acc;
        iter_ = 0;
        sdat.ireset = 0;

        // Set up workspace dimensions.
        sdat.n1 = n + 1;
        n1 = n + 1;
        sdat.n2 = n1 * n / 2;
        n2 = n1 * n / 2;
        sdat.n3 = n2 + 1;

        // Zero search direction and multiplier estimates.
        dfill(n, ZERO, s, 1);
        dfill(m, ZERO, mu, 1);

        // Initialise BFGS to identity.
        let result = reset_bfgs(
            n,
            n1,
            n2,
            l,
            s,
            acc,
            sdat,
            tolf,
            toldf,
            toldx,
            f,
            x,
            x0,
            c,
            m,
            meq,
            mu,
            inconsistent_linearization,
            iter_,
        );
        acc = result.0;
        iter_ = result.1;
        mode = result.2;
        if sdat.ireset > 5 {
            return (acc, iter_, mode);
        }
    } else {
        // -----------------------------------------------------------
        // Function evaluation received → line-search check
        // -----------------------------------------------------------

        // Compute the augmented Lagrangian merit function t = f + mu^T max(-c, c_eq).
        sdat.t = f;
        for j in 0..m {
            let h1 = if j < meq { c[j] } else { ZERO };
            sdat.t += mu[j] * (-c[j]).max(h1);
        }
        let h1 = sdat.t - sdat.t0;

        if sdat.iexact == 0 {
            // Inexact line-search (Armijo).
            if h1 <= sdat.h3 / TEN || sdat.line > 10 {
                // Sufficient decrease or too many line-search steps.
                let result = convergence_check(
                    acc,
                    0,
                    -1,
                    n,
                    f,
                    sdat.f0,
                    x,
                    x0,
                    s,
                    c,
                    m,
                    meq,
                    mu,
                    sdat.tol,
                    tolf,
                    toldf,
                    toldx,
                    inconsistent_linearization,
                    iter_,
                );
                acc = result.0;
                iter_ = result.1;
                mode = result.2;
            } else {
                // Insufficient decrease — try a shorter step.
                sdat.alpha = (sdat.h3 / (TWO * (sdat.h3 - h1)))
                    .max(alphamin)
                    .min(alphamax);
                inexact_linesearch(n, x, x0, s, sdat, xl, xu, infbnd);
                mode = 1;
                return (acc, iter_, mode);
            }
        } else {
            // Exact line-search (Brent).
            exact_linesearch(
                n, x, x0, s, sdat, ldat, sdat.t, alphamin, alphamax, xl, xu, infbnd,
            );
            if sdat.line == 3 {
                // Linmin converged — check SQP convergence.
                let result = convergence_check(
                    acc,
                    0,
                    -1,
                    n,
                    f,
                    sdat.f0,
                    x,
                    x0,
                    s,
                    c,
                    m,
                    meq,
                    mu,
                    sdat.tol,
                    tolf,
                    toldf,
                    toldx,
                    inconsistent_linearization,
                    iter_,
                );
                acc = result.0;
                iter_ = result.1;
                mode = result.2;
            } else {
                // Linmin needs another function evaluation.
                mode = 1;
                return (acc, iter_, mode);
            }
        }
        return (acc, iter_, mode);
    }

    // ===================================================================
    // Main SQP iteration loop
    // ===================================================================
    loop {
        iter_ += 1;
        mode = 9; // iteration-limit error (overwritten below if not exceeded)
        if iter_ > sdat.itermx {
            return (acc, iter_, mode);
        }

        // ---------------------------------------------------------------
        // Compute search direction via QP sub-problem
        // ---------------------------------------------------------------

        // Compute relative bounds: u = xl - x, v = xu - x.
        dcopy(n, xl, 1, u, 1);
        dcopy(n, xu, 1, v, 1);
        daxpy(n, -ONE, x, 1, u, 1);
        daxpy(n, -ONE, x, 1, v, 1);
        sdat.h4 = ONE;

        let a_view = a.view(la, n);

        mode = {
            profile_section!("lsq_ws_call");
            lsq_ws(
                m,
                meq,
                n,
                sdat.n3,
                la,
                l,
                &g[..n],
                a_view,
                &c[..la],
                &u[..n],
                &v[..n],
                s,
                r,
                max_iter_ls,
                nnls_mode,
                infbnd,
                ls_ws,
            )
        };

        // ---------------------------------------------------------------
        // Handle inconsistent linearisation (augmented problem)
        // ---------------------------------------------------------------
        inconsistent_linearization = false;
        if mode == 6 {
            if n == meq {
                mode = 4;
            }
        }
        if mode == 4 {
            // The QP sub-problem is infeasible — solve an augmented problem
            // with an artificial variable to find a feasible descent direction.
            inconsistent_linearization = true;
            for j in 0..m {
                if j < meq {
                    a[(j, n)] = -c[j];
                } else {
                    a[(j, n)] = (-c[j]).max(ZERO);
                }
            }
            dfill(n, ZERO, s, 1);
            sdat.h3 = ZERO;
            g[n] = ZERO;
            l[n2] = HUN;
            s[n] = ONE;
            u[n] = ZERO;
            v[n] = ONE;
            sdat.incons = 0;

            // Retry the augmented QP (with n+1 variables).
            loop {
                let a_view2 = a.view(la, n + 1);
                mode = lsq_ws(
                    m,
                    meq,
                    n + 1,
                    sdat.n3,
                    la,
                    l,
                    &g[..n + 1],
                    a_view2,
                    &c[..la],
                    &u[..n + 1],
                    &v[..n + 1],
                    s,
                    r,
                    max_iter_ls,
                    nnls_mode,
                    infbnd,
                    ls_ws,
                );
                sdat.h4 = ONE - s[n];
                if mode == 4 {
                    // Still infeasible — increase regularisation.
                    l[n2] = TEN * l[n2];
                    sdat.incons += 1;
                    if sdat.incons > 5 {
                        return (acc, iter_, mode);
                    }
                } else if mode != 1 {
                    return (acc, iter_, mode);
                } else {
                    break;
                }
            }
        } else if mode != 1 {
            return (acc, iter_, mode);
        }

        // ---------------------------------------------------------------
        // Update Lagrange multiplier estimates for the L1 merit function
        // ---------------------------------------------------------------

        // Compute v = g - A^T r (gradient of the Lagrangian).
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..m {
                dot += a[(j, i)] * r[j];
            }
            v[i] = g[i] - dot;
        }

        // Save current iterate and compute directional derivative.
        sdat.f0 = f;
        dcopy(n, x, 1, x0, 1);
        sdat.gs = ddot(n, g, 1, s, 1);
        sdat.h1 = sdat.gs.abs();
        sdat.h2 = ZERO;
        for j in 0..m {
            let h3 = if j < meq { c[j] } else { ZERO };
            sdat.h2 += (-c[j]).max(h3);
            // Update multiplier estimates (exponential smoothing).
            let h3_abs = r[j].abs();
            mu[j] = h3_abs.max((mu[j] + h3_abs) / TWO);
            sdat.h1 += h3_abs * c[j].abs();
        }

        // ---------------------------------------------------------------
        // Check KKT convergence
        // ---------------------------------------------------------------
        mode = 0;
        if sdat.h1 < acc && sdat.h2 < acc && !inconsistent_linearization && !f.is_nan() {
            return (acc, iter_, mode); // Converged!
        }

        // ---------------------------------------------------------------
        // Compute merit function and check descent direction
        // ---------------------------------------------------------------
        sdat.h1 = ZERO;
        for j in 0..m {
            let h3 = if j < meq { c[j] } else { ZERO };
            sdat.h1 += mu[j] * (-c[j]).max(h3);
        }
        sdat.t0 = f + sdat.h1;
        sdat.h3 = sdat.gs - sdat.h1 * sdat.h4;
        if sdat.h3 >= ZERO {
            // Not a descent direction — reset BFGS and retry.
            let result = reset_bfgs(
                n,
                n1,
                n2,
                l,
                s,
                acc,
                sdat,
                tolf,
                toldf,
                toldx,
                f,
                x,
                x0,
                c,
                m,
                meq,
                mu,
                inconsistent_linearization,
                iter_,
            );
            acc = result.0;
            iter_ = result.1;
            mode = result.2;
            if sdat.ireset > 5 {
                return (acc, iter_, mode);
            }
        } else {
            break; // Descent direction found — proceed to line-search.
        }
    }

    // ===================================================================
    // Line-search
    // ===================================================================
    sdat.line = 0;
    sdat.alpha = alphamax;
    if sdat.iexact == 1 {
        // Exact line-search (Brent).
        exact_linesearch(
            n, x, x0, s, sdat, ldat, sdat.t, alphamin, alphamax, xl, xu, infbnd,
        );
        if sdat.line == 3 {
            let result = convergence_check(
                acc,
                0,
                -1,
                n,
                f,
                sdat.f0,
                x,
                x0,
                s,
                c,
                m,
                meq,
                mu,
                sdat.tol,
                tolf,
                toldf,
                toldx,
                inconsistent_linearization,
                iter_,
            );
            acc = result.0;
            iter_ = result.1;
            mode = result.2;
        } else {
            mode = 1; // Request function evaluation.
        }
    } else {
        // Inexact line-search (Armijo).
        inexact_linesearch(n, x, x0, s, sdat, xl, xu, infbnd);
        mode = 1; // Request function evaluation.
    }

    (acc, iter_, mode)
}

// ===========================================================================
// SlsqpWorkspace — owns all sub-arrays, eliminates copy-in/copy-out
// ===========================================================================

/// Pre-allocated workspace for the SLSQP solver.
///
/// Replaces the flat `w` array approach: each sub-array is owned separately,
/// so `slsqpb` can borrow them concurrently without copying.
pub struct SlsqpWorkspace {
    /// Lagrange multiplier estimates (length `la`).
    pub mu_arr: Vec<f64>,
    /// Packed LDL^T BFGS factors (length `n1*n/2 + 1`).
    pub l_arr: Vec<f64>,
    /// Previous iterate (length `n`).
    pub x0_arr: Vec<f64>,
    /// QP multipliers / workspace (length `n+n+la+2`).
    pub r_arr: Vec<f64>,
    /// Search direction (length `n+1`).
    pub s_arr: Vec<f64>,
    /// Workspace — relative lower bounds / BFGS (length `n+1`).
    pub u_arr: Vec<f64>,
    /// Workspace — relative upper bounds / BFGS (length `n+1`).
    pub v_arr: Vec<f64>,
    /// Additional workspace.
    pub w_sub: Vec<f64>,
    /// Persistent SQP iteration state.
    pub sdat: SlsqpbData,
    /// Persistent line-search state.
    pub ldat: LinminData,
    /// Pre-allocated workspace for the LS chain (lsq → lsei → lsi → ldp → nnls).
    /// Eliminates ~1MB of allocation traffic per SQP iteration for large n.
    pub ls_ws: LsWorkspace,
}

impl SlsqpWorkspace {
    /// Allocate a workspace for a problem with `n` variables, `m` total
    /// constraints, `meq` equality constraints.
    pub fn new(n: usize, m: usize, meq: usize) -> Self {
        let n1 = n + 1;
        let la = 1.max(m);
        let mineq = m - meq + 2 * n1;

        let l_len = n1 * n / 2 + 1;
        let r_len = n + n + la + 2;
        let w_sub_len = (3 * n1 + m) * (n1 + 1)
            + (n1 - meq + 1) * (mineq + 2)
            + 2 * mineq
            + (n1 + mineq) * (n1 - meq)
            + 2 * meq
            + n1 * n / 2
            + 2 * m
            + 3 * n
            + 4 * n1
            + 1;

        SlsqpWorkspace {
            mu_arr: vec![0.0; la],
            l_arr: vec![0.0; l_len],
            x0_arr: vec![0.0; n],
            r_arr: vec![0.0; r_len],
            s_arr: vec![0.0; n1],
            u_arr: vec![0.0; n1],
            v_arr: vec![0.0; n1],
            w_sub: vec![0.0; w_sub_len],
            sdat: SlsqpbData::default(),
            ldat: LinminData::default(),
            ls_ws: LsWorkspace::new(),
        }
    }

    /// Reset workspace to zeros (for re-use across multiple optimisations).
    pub fn reset(&mut self) {
        self.mu_arr.fill(0.0);
        self.l_arr.fill(0.0);
        self.x0_arr.fill(0.0);
        self.r_arr.fill(0.0);
        self.s_arr.fill(0.0);
        self.u_arr.fill(0.0);
        self.v_arr.fill(0.0);
        self.w_sub.fill(0.0);
        self.sdat = SlsqpbData::default();
        self.ldat = LinminData::default();
        self.ls_ws = LsWorkspace::new();
    }
}

/// One step of SLSQP using the workspace struct (no copy-in/copy-out).
///
/// This is functionally identical to [`slsqp`] but operates on owned
/// sub-arrays in [`SlsqpWorkspace`] rather than slicing a flat `w` array.
pub fn slsqp_step(
    m: usize,
    meq: usize,
    la: usize,
    n: usize,
    x: &mut [f64],
    xl: &[f64],
    xu: &[f64],
    f: f64,
    c: &mut [f64],
    g: &mut [f64],
    a: &mut ColMat,
    acc: f64,
    iter_: usize,
    mode: i32,
    ws: &mut SlsqpWorkspace,
    alphamin: f64,
    alphamax: f64,
    tolf: f64,
    toldf: f64,
    toldx: f64,
    max_iter_ls: usize,
    nnls_mode: NnlsMode,
    infinite_bound: f64,
) -> (f64, usize, i32) {
    let infbnd = if infinite_bound == ZERO {
        f64::MAX
    } else {
        infinite_bound.abs()
    };
    let n1 = n + 1;

    ws.sdat.n1 = n1;

    if meq > n {
        return (acc, 0, 2);
    }

    slsqpb(
        m,
        meq,
        la,
        n,
        x,
        xl,
        xu,
        f,
        c,
        g,
        a,
        acc,
        iter_,
        mode,
        &mut ws.r_arr,
        &mut ws.l_arr,
        &mut ws.x0_arr,
        &mut ws.mu_arr,
        &mut ws.s_arr,
        &mut ws.u_arr,
        &mut ws.v_arr,
        &mut ws.w_sub,
        &mut ws.sdat,
        &mut ws.ldat,
        alphamin,
        alphamax,
        tolf,
        toldf,
        toldx,
        max_iter_ls,
        nnls_mode,
        infbnd,
        &mut ws.ls_ws,
    )
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_types::{ColMat, SlsqpbData};

    // -- linmin (Brent's line-search) ----------------------------------------

    #[test]
    fn linmin_initialisation_returns_mode1() {
        let mut ldat = LinminData::default();
        let (x, mode) = linmin(0, 0.0, 1.0, 0.0, 1e-8, &mut ldat);
        assert_eq!(mode, 1); // requests first function evaluation
        assert!(x >= 0.0 && x <= 1.0);
    }

    #[test]
    fn linmin_finds_minimum_of_quadratic() {
        // Minimise f(x) = (x - 0.3)² on [0, 1]
        // Feed linmin with reverse-communication protocol
        let mut ldat = LinminData::default();
        let tol = 1e-8;

        // Mode 0: initialise
        let (mut x, mut mode) = linmin(0, 0.0, 1.0, 0.0, tol, &mut ldat);
        assert_eq!(mode, 1);

        // Mode 1: first evaluation
        let f = (x - 0.3) * (x - 0.3);
        let result = linmin(mode, 0.0, 1.0, f, tol, &mut ldat);
        x = result.0;
        mode = result.1;

        // Iterate until convergence (mode == 3)
        let mut iters = 0;
        while mode == 2 && iters < 100 {
            let f = (x - 0.3) * (x - 0.3);
            let result = linmin(mode, 0.0, 1.0, f, tol, &mut ldat);
            x = result.0;
            mode = result.1;
            iters += 1;
        }

        assert_eq!(mode, 3); // converged
        assert!((x - 0.3).abs() < 1e-5, "x = {}, expected ≈ 0.3", x);
    }

    #[test]
    fn linmin_finds_minimum_of_quartic() {
        // Minimise f(x) = (x - 0.7)⁴ on [0, 1]
        let mut ldat = LinminData::default();
        let tol = 1e-8;

        let (mut x, mut mode) = linmin(0, 0.0, 1.0, 0.0, tol, &mut ldat);

        let mut iters = 0;
        loop {
            let f = (x - 0.7_f64).powi(4);
            let result = linmin(mode, 0.0, 1.0, f, tol, &mut ldat);
            x = result.0;
            mode = result.1;
            iters += 1;
            if mode == 3 || iters > 200 {
                break;
            }
        }

        assert_eq!(mode, 3);
        assert!((x - 0.7).abs() < 1e-4, "x = {}, expected ≈ 0.7", x);
    }

    // -- slsqp_step (full reverse-communication integration test) -------------

    #[test]
    fn slsqp_step_unconstrained_rosenbrock_2d() {
        // Minimise Rosenbrock: f(x) = (1-x₁)² + 100(x₂-x₁²)²
        // Optimal: x* = [1, 1], f* = 0
        let n = 2usize;
        let m = 0usize;
        let meq = 0usize;
        let la = 1usize.max(m + 1);

        let mut x = vec![-1.0, 1.0];
        let xl = vec![-10.0; n];
        let xu = vec![10.0; n];
        let mut g = vec![0.0; n + 1];
        let mut c = vec![0.0; la];
        let mut a = ColMat::zeros(la, n + 1);
        let mut ws = SlsqpWorkspace::new(n, m, meq);

        let acc = 1e-10;
        let max_iter = 200usize;
        let mut mode = 0_i32;
        let mut iter_ = max_iter;

        for _outer in 0..500 {
            // Evaluate objective and gradient
            if mode == 0 || mode == 1 || mode == -1 {
                let x1: f64 = x[0];
                let x2: f64 = x[1];
                let f: f64 = (1.0 - x1).powi(2) + 100.0 * (x2 - x1 * x1).powi(2);
                g[0] = -2.0 * (1.0 - x1) + 200.0 * (x2 - x1 * x1) * (-2.0 * x1);
                g[1] = 200.0 * (x2 - x1 * x1);

                let result = slsqp_step(
                    m,
                    meq,
                    la,
                    n,
                    &mut x,
                    &xl,
                    &xu,
                    f,
                    &mut c,
                    &mut g,
                    &mut a,
                    acc,
                    iter_,
                    mode,
                    &mut ws,
                    0.1,
                    1.0, // alphamin, alphamax
                    -1.0,
                    -1.0,
                    -1.0, // tolf, toldf, toldx (disabled)
                    0,
                    NnlsMode::Nnls, // max_iter_ls, nnls_mode
                    1e20,           // infinite_bound
                );
                let new_acc = result.0;
                iter_ = result.1;
                mode = result.2;
                let _ = new_acc;
            } else {
                break;
            }
        }

        assert_eq!(mode, 0, "SLSQP did not converge, mode = {}", mode);
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0] = {}, expected ≈ 1.0", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1] = {}, expected ≈ 1.0", x[1]);
    }

    #[test]
    fn slsqp_step_equality_constrained() {
        // min f(x) = x₁² + x₂²  s.t. x₁ + x₂ = 1
        // Optimal: x* = [0.5, 0.5], f* = 0.5
        let n = 2usize;
        let m = 1usize;
        let meq = 1usize;
        let la = m + 1;

        let mut x = vec![0.0, 0.0];
        let xl = vec![-10.0; n];
        let xu = vec![10.0; n];
        let mut g = vec![0.0; n + 1];
        let mut c = vec![0.0; la];
        let mut a = ColMat::zeros(la, n + 1);
        let mut ws = SlsqpWorkspace::new(n, m, meq);

        let acc = 1e-10;
        let mut mode = 0_i32;
        let mut iter_: usize = 100;

        for _outer in 0..300 {
            if mode == 0 || mode == 1 || mode == -1 {
                let x1 = x[0];
                let x2 = x[1];
                let f = x1 * x1 + x2 * x2;
                g[0] = 2.0 * x1;
                g[1] = 2.0 * x2;

                // Equality constraint: x1 + x2 - 1 = 0
                c[0] = x1 + x2 - 1.0;
                // Jacobian of constraint
                a[(0, 0)] = 1.0;
                a[(0, 1)] = 1.0;

                let result = slsqp_step(
                    m,
                    meq,
                    la,
                    n,
                    &mut x,
                    &xl,
                    &xu,
                    f,
                    &mut c,
                    &mut g,
                    &mut a,
                    acc,
                    iter_,
                    mode,
                    &mut ws,
                    0.1,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0,
                    NnlsMode::Nnls,
                    1e20,
                );
                iter_ = result.1;
                mode = result.2;
            } else {
                break;
            }
        }

        assert_eq!(mode, 0, "SLSQP did not converge, mode = {}", mode);
        assert!((x[0] - 0.5).abs() < 1e-4, "x[0] = {}, expected ≈ 0.5", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-4, "x[1] = {}, expected ≈ 0.5", x[1]);
    }

    #[test]
    fn slsqp_step_inequality_constrained_rosenbrock() {
        // Minimise Rosenbrock subject to x₁² + x₂² ≤ 1
        // c[0] = 1 - x₁² - x₂² ≥ 0
        // Optimal: x* ≈ [0.7864, 0.6177]
        let n = 2usize;
        let m = 1usize;
        let meq = 0usize;
        let la = 1usize.max(m);

        let mut x = vec![0.1, 0.1];
        let xl = vec![-1.0; n];
        let xu = vec![1.0; n];
        let mut g = vec![0.0; n + 1];
        let mut c = vec![0.0; la];
        let mut a = ColMat::zeros(la, n + 1);
        let mut ws = SlsqpWorkspace::new(n, m, meq);

        let acc = 1e-8;
        let mut mode = 0_i32;
        let mut iter_: usize = 100;
        let mut f = 0.0_f64;

        for _outer in 0..1000 {
            if mode == 0 || mode == 1 {
                let dx = x[1] - x[0] * x[0];
                let omx = 1.0 - x[0];
                f = 100.0 * dx * dx + omx * omx;
                c[0] = 1.0 - x[0] * x[0] - x[1] * x[1];
            }
            if mode == 0 || mode == -1 {
                g[0] = -400.0 * (x[1] - x[0] * x[0]) * x[0] - 2.0 * (1.0 - x[0]);
                g[1] = 200.0 * (x[1] - x[0] * x[0]);
                a[(0, 0)] = -2.0 * x[0];
                a[(0, 1)] = -2.0 * x[1];
            }

            let result = slsqp_step(
                m,
                meq,
                la,
                n,
                &mut x,
                &xl,
                &xu,
                f,
                &mut c,
                &mut g,
                &mut a,
                acc,
                iter_,
                mode,
                &mut ws,
                0.1,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                0,
                NnlsMode::Nnls,
                0.0, // infinite_bound = 0 → uses f64::MAX
            );
            iter_ = result.1;
            mode = result.2;

            if mode != 1 && mode != -1 {
                break;
            }
        }

        assert_eq!(mode, 0, "SLSQP did not converge, mode = {}", mode);
        assert!(
            (x[0] - 0.7864).abs() < 0.01,
            "x[0] = {}, expected ≈ 0.7864",
            x[0]
        );
        assert!(
            (x[1] - 0.6177).abs() < 0.01,
            "x[1] = {}, expected ≈ 0.6177",
            x[1]
        );
    }

    // -- reset_bfgs ----------------------------------------------------------

    #[test]
    fn reset_bfgs_sets_identity() {
        let n = 3_usize;
        let n1 = n + 1;
        let n2 = n1 * n / 2;
        let mut l = vec![99.0; n2 + 1];
        let s = vec![0.0; n];
        let x = vec![0.0; n];
        let x0 = vec![0.0; n];
        let c = vec![];
        let mu = vec![];
        let mut sdat = SlsqpbData::default();
        sdat.ireset = 0;

        let (_acc, _iter, mode) = reset_bfgs(
            n, n1, n2, &mut l, &s, 1e-6, &mut sdat, -1.0, -1.0, -1.0, 0.0, &x, &x0, &c, 0, 0, &mu,
            false, 0,
        );
        assert_eq!(mode, -1); // request Jacobian re-evaluation
        assert_eq!(sdat.ireset, 1);

        // Check identity in packed format: diagonals are 1, off-diagonals are 0
        // For n=3: packed indices for diagonals: 0, 3, 5
        assert_eq!(l[0], 1.0); // d0
        assert_eq!(l[1], 0.0); // l10
        assert_eq!(l[2], 0.0); // l20
        assert_eq!(l[3], 1.0); // d1
        assert_eq!(l[4], 0.0); // l21
        assert_eq!(l[5], 1.0); // d2
    }
}
