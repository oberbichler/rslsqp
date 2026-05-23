//! Constrained least-squares routines: NNLS, LDP, LSI, LSEI, LSQ.
//!
//! Port of the second half of `slsqp_core.f90`.  These routines form the
//! constraint-handling chain used by the QP sub-problem solver:
//!
//! ```text
//!     lsq → lsei → lsi → ldp → nnls (or bvls)
//! ```
//!
//! - [`nnls`]  — Non-Negative Least Squares (Lawson–Hanson algorithm).
//! - [`ldp`]   — Least Distance Programming (`min ½‖x‖² s.t. Gx ≥ h`).
//! - [`lsi`]   — Least Squares with Inequality constraints (`min ‖Ex−f‖ s.t. Gx ≥ h`).
//! - [`lsei`]  — Least Squares with Equality and Inequality constraints.
//! - [`lsq`]   — QP sub-problem solver (top of the chain, called from `slsqpb`).
//!
//! ## Workspace pre-allocation
//!
//! Each routine has a `_ws` variant that accepts the appropriate nested
//! workspace type to reuse pre-allocated buffers across SQP iterations.
//! The public (non-`_ws`) functions create a temporary workspace for
//! backward compatibility with the Python bindings.

use crate::bvls;
use crate::core_basic::*;
use crate::core_types::{
    ColMat, ColMatView, LdpChainWs, LsWorkspace, LseiChainWs, LsiChainWs, NnlsMode, NnlsWs,
    ensure_vec_len,
};
use crate::support::*;

// ===========================================================================
// NNLS — Non-Negative Least Squares
// ===========================================================================

// ---- NNLS helper functions ----

/// Solve the upper-triangular system arising from the NNLS active-set QR.
fn nnls_solve_triangular_cm(a: &ColMat, index: &[usize], nsetp: usize, zz: &mut [f64]) {
    let mut jj = 0_usize;
    for l in 0..nsetp {
        let ip = nsetp - 1 - l;
        if l != 0 {
            let col = a.col(jj);
            for ii in 0..=ip {
                zz[ii] -= col[ii] * zz[ip + 1];
            }
        }
        jj = index[ip];
        zz[ip] /= a[(ip, jj)];
    }
}

/// Package the final NNLS result (success path).
fn nnls_terminate_cm(
    b: &[f64],
    _x: &[f64],
    w: &mut [f64],
    m_val: usize,
    n: usize,
    npp1: usize,
) -> (f64, i32) {
    let mut sm = ZERO;
    if npp1 <= m_val.saturating_sub(1) {
        for i in npp1..m_val {
            sm += b[i] * b[i];
        }
    } else {
        for i in 0..n {
            w[i] = ZERO;
        }
    }
    (sm.sqrt(), 1)
}

/// Compute the residual norm `‖b[npp1..m]‖_2`.
fn nnls_compute_rnorm(b: &[f64], npp1: usize, m_val: usize) -> f64 {
    let mut sm = ZERO;
    for i in npp1..m_val {
        sm += b[i] * b[i];
    }
    sm.sqrt()
}

/// NNLS with pre-allocated workspace — avoids per-call heap allocations.
///
/// Accepts a **column-major** `ColMat` directly, eliminating any
/// row↔column-major conversion overhead.
pub fn nnls_ws(
    a: &mut ColMat,
    m_val: usize,
    n: usize,
    b: &mut [f64],
    max_iter: usize,
    ws: &mut NnlsWs,
) -> (f64, i32) {
    profile_section!("nnls_ws");

    if m_val == 0 || n == 0 {
        ws.x[..n].fill(ZERO);
        ws.w[..n].fill(ZERO);
        return (ZERO, 2);
    }

    nnls_colmajor_ws(a, m_val, n, b, max_iter, ws)
}

/// NNLS core operating on column-major storage, with workspace.
///
/// The `ColMat` is passed separately from `NnlsWs` so that callers can
/// supply their own column-major matrix (e.g. `LdpWs.e_cm`) directly.
fn nnls_colmajor_ws(
    a: &mut ColMat,
    m: usize,
    nu: usize,
    b: &mut [f64],
    max_iter: usize,
    ws: &mut NnlsWs,
) -> (f64, i32) {
    let factor = 0.01;

    // Prepare workspace vectors
    ensure_vec_len(&mut ws.x, nu);
    ensure_vec_len(&mut ws.w, nu);
    ensure_vec_len(&mut ws.zz, m);
    if ws.index.len() < nu {
        ws.index.resize(nu, 0);
    }
    for i in 0..nu {
        ws.index[i] = i;
    }

    let itmax = if max_iter > 0 {
        (3 * nu).max(max_iter)
    } else {
        3 * nu
    };

    let iz2 = nu - 1;
    let mut iz1: usize = 0;
    let mut nsetp: usize = 0;
    let mut npp1: usize = 0;
    let mut iter_count: usize = 0;

    // Alias refs into workspace for convenience
    let x = &mut ws.x[..nu];
    let w = &mut ws.w[..nu];
    let zz = &mut ws.zz[..m];
    let index = &mut ws.index[..nu];

    // -----------------------------------------------------------------------
    // Main loop — add variables from Z to P
    // -----------------------------------------------------------------------
    loop {
        if iz1 > iz2 || nsetp >= m {
            return nnls_terminate_cm(b, x, w, m, nu, npp1);
        }

        // Compute dual vector w[j] = A[:,j]^T * b[npp1..] for each j in Z.
        {
            let tail = m - npp1;
            for iz in iz1..=iz2 {
                let j = index[iz];
                let col = a.col(j);
                w[j] = crate::lapack::accel_ddot(tail, &col[npp1..], 1, &b[npp1..], 1);
            }
        }

        // Find the Z-set variable with the largest positive dual.
        loop {
            let mut wmax = ZERO;
            let mut izmax = iz1;
            for iz in iz1..=iz2 {
                let j = index[iz];
                if w[j] > wmax {
                    wmax = w[j];
                    izmax = iz;
                }
            }

            if wmax <= ZERO {
                return nnls_terminate_cm(b, x, w, m, nu, npp1);
            }

            let iz = izmax;
            let j = index[iz];

            // Householder on column j — column is contiguous (stride=1).
            let asave = a[(npp1, j)];
            let col_j = a.col_mut(j);
            let up = h12_construct(npp1, npp1 + 1, m, col_j, 1);

            // Compute norm of the already-factored part of column j.
            let col_j = a.col(j);
            let unorm = crate::lapack::accel_dnrm2(nsetp, col_j, 1);

            if a[(npp1, j)].abs() * factor >= unorm * f64::EPSILON {
                // Column is sufficiently independent — trial solve.
                zz[..m].copy_from_slice(&b[..m]);
                h12_apply(npp1, npp1 + 1, m, a.col(j), 1, up, zz, 1, 1, 1);
                let ztest = zz[npp1] / a[(npp1, j)];

                if ztest > ZERO {
                    // Move j from Z to P.
                    b[..m].copy_from_slice(&zz[..m]);
                    index[iz] = index[iz1];
                    index[iz1] = j;
                    iz1 += 1;
                    nsetp = npp1 + 1;
                    npp1 += 1;

                    // Apply Householder to remaining Z-set columns.
                    if iz1 <= iz2 {
                        let rows = a.rows;
                        let lpivot = nsetp - 1;
                        let l1 = npp1;
                        let j_off = j * rows;
                        let b_val = up * a.data[j_off + lpivot];
                        if b_val < ZERO {
                            let inv_b = ONE / b_val;
                            let tail = m - l1;
                            for jz in iz1..=iz2 {
                                let jj = index[jz];
                                let jj_off = jj * rows;
                                // sm = up * c[pivot] + u_tail · c_tail
                                let mut sm = up * a.data[jj_off + lpivot];
                                sm += crate::lapack::accel_ddot(
                                    tail,
                                    &a.data[j_off + l1..],
                                    1,
                                    &a.data[jj_off + l1..],
                                    1,
                                );
                                if sm.abs() > ZERO {
                                    let sm_scaled = sm * inv_b;
                                    a.data[jj_off + lpivot] += sm_scaled * up;
                                    // c_tail += sm_scaled * u_tail
                                    // SAFETY: j != jj so column slices don't overlap.
                                    unsafe {
                                        let dx = std::slice::from_raw_parts(
                                            a.data.as_ptr().add(j_off + l1),
                                            tail,
                                        );
                                        let dy = std::slice::from_raw_parts_mut(
                                            a.data.as_mut_ptr().add(jj_off + l1),
                                            tail,
                                        );
                                        crate::lapack::accel_daxpy(tail, sm_scaled, dx, 1, dy, 1);
                                    }
                                }
                            }
                        }
                    }

                    // Zero out sub-diagonal of column j.
                    let col_j = a.col_mut(j);
                    for l in npp1..m {
                        col_j[l] = ZERO;
                    }
                    w[j] = ZERO;

                    nnls_solve_triangular_cm(a, index, nsetp, zz);
                    break;
                }
            }
            // Reject column j.
            a[(npp1, j)] = asave;
            w[j] = ZERO;
        }

        // -------------------------------------------------------------------
        // Secondary loop — enforce non-negativity on P-set variables
        // -------------------------------------------------------------------
        loop {
            iter_count += 1;
            if iter_count > itmax {
                let rnorm = nnls_compute_rnorm(b, npp1, m);
                return (rnorm, 3);
            }

            let mut alpha_val = TWO;
            let mut jj_pivot: usize = 0;
            for ip in 0..nsetp {
                let l = index[ip];
                if zz[ip] <= ZERO {
                    let t = -x[l] / (zz[ip] - x[l]);
                    if alpha_val > t {
                        alpha_val = t;
                        jj_pivot = ip;
                    }
                }
            }

            if (alpha_val - TWO).abs() <= ZERO {
                for ip in 0..nsetp {
                    let i = index[ip];
                    x[i] = zz[ip];
                }
                break;
            }

            for ip in 0..nsetp {
                let l = index[ip];
                x[l] += alpha_val * (zz[ip] - x[l]);
            }

            // Move the most-infeasible variable from P to Z.
            let mut i = index[jj_pivot];
            loop {
                x[i] = ZERO;

                if jj_pivot != nsetp - 1 {
                    let rows = a.rows;
                    for j in jj_pivot..(nsetp - 1) {
                        let ii = index[j + 1];
                        index[j] = ii;
                        let (cc, ss, sig) = g1(a[(j, ii)], a[(j + 1, ii)]);
                        a[(j, ii)] = sig;
                        a[(j + 1, ii)] = ZERO;
                        // Apply Givens rotation to rows j, j+1 across all
                        // columns except ii (already handled above).
                        crate::lapack::drot_cm_rows(&mut a.data, rows, nu, j, cc, ss, ii);
                        let temp = b[j];
                        b[j] = cc * temp + ss * b[j + 1];
                        b[j + 1] = -ss * temp + cc * b[j + 1];
                    }
                }

                npp1 = nsetp - 1;
                nsetp -= 1;
                iz1 -= 1;
                index[iz1] = i;

                let mut found_inf = false;
                for jj2 in 0..nsetp {
                    let i2 = index[jj2];
                    if x[i2] <= ZERO {
                        jj_pivot = jj2;
                        i = i2;
                        found_inf = true;
                        break;
                    }
                }
                if !found_inf {
                    break;
                }
            }

            zz[..m].copy_from_slice(&b[..m]);
            nnls_solve_triangular_cm(a, index, nsetp, zz);
        }
    }
}

// ===========================================================================
// LDP — Least Distance Programming
// ===========================================================================

/// LDP with pre-allocated workspace.
///
/// Takes `&mut LdpChainWs` which contains LDP-level temporaries and
/// the NNLS sub-workspace, allowing disjoint borrows.
///
/// Accepts G as a **column-major** `ColMat`, matching the native storage
/// of the upstream LSI solver.  The dual NNLS matrix E is also built
/// directly in column-major format, eliminating row↔column-major
/// conversions at both the LDP→NNLS and LSI→LDP boundaries.
pub fn ldp_ws(
    g: &ColMat,
    m_val: usize,
    n: usize,
    h: &[f64],
    max_iter_ls: usize,
    nnls_mode: NnlsMode,
    ws: &mut LdpChainWs,
) -> (f64, i32) {
    profile_section!("ldp_ws");

    if n == 0 {
        return (ZERO, 2);
    }

    ws.ldp.prepare(m_val, n);

    if m_val == 0 {
        ws.ldp.x_out[..n].fill(ZERO);
        return (ZERO, 1);
    }

    // Build the dual NNLS problem:
    //   E = [G^T; h^T]  (n+1 × m),  f = [0; ...; 0; 1]  (n+1)
    let n1 = n + 1;
    ws.ldp.e_cm.resize_zero(n1, m_val);
    for j in 0..m_val {
        for i in 0..n {
            ws.ldp.e_cm[(i, j)] = g[(j, i)]; // transpose G
        }
        ws.ldp.e_cm[(n, j)] = h[j];
    }
    ensure_vec_len(&mut ws.ldp.f_vec, n1);
    ws.ldp.f_vec[n] = ONE;

    // Solve the NNLS sub-problem.
    // We pass `&mut ws.ldp.e_cm` and `&mut ws.ldp.f_vec` as the matrix/rhs,
    // and `&mut ws.nnls` as the NNLS workspace — all disjoint fields.
    let (rnorm, mode) = match nnls_mode {
        NnlsMode::Nnls => nnls_ws(
            &mut ws.ldp.e_cm,
            n1,
            m_val,
            &mut ws.ldp.f_vec[..n1],
            max_iter_ls,
            &mut ws.nnls,
        ),
        NnlsMode::Bvls => {
            let (y_bvls, rn, _wdual, md) = bvls_wrapper_nnls(
                &mut ws.ldp.e_cm,
                n1,
                m_val,
                &mut ws.ldp.f_vec[..n1].to_vec(),
                max_iter_ls,
            );
            ensure_vec_len(&mut ws.nnls.x, m_val);
            ws.nnls.x[..m_val].copy_from_slice(&y_bvls[..m_val]);
            (rn, md)
        }
    };

    if mode == 1 {
        let y = &ws.nnls.x;
        let mut mode = 4;
        if rnorm > ZERO {
            let fac = ONE - ddot(m_val, h, 1, y, 1);
            if fac.is_nan() {
                ws.ldp.x_out[..n].fill(ZERO);
                ws.ldp.w_out[..m_val].fill(ZERO);
                return (ZERO, mode);
            }
            if fac >= f64::EPSILON {
                mode = 1;
                let fac = ONE / fac;
                // x = fac * G^T * y — use BLAS dgemv on macOS, manual on others.
                // G is ColMat (column-major), so G.col(j) = [G[0,j], ..., G[m-1,j]].
                // (G^T * y)_j = sum_i G[i,j] * y[i] = ddot(m, G.col(j), 1, y, 1).
                #[cfg(feature = "blas")]
                {
                    crate::lapack::colmajor_dgemv(
                        true, // transpose
                        m_val,
                        n,
                        fac, // alpha
                        &g.data,
                        g.rows,
                        y,
                        1,
                        ZERO, // beta
                        &mut ws.ldp.x_out,
                        1,
                    );
                }
                #[cfg(not(feature = "blas"))]
                {
                    for j in 0..n {
                        ws.ldp.x_out[j] = fac * ddot(m_val, g.col(j), 1, y, 1);
                    }
                }
                let xnorm = dnrm2(n, &ws.ldp.x_out, 1);
                for i in 0..m_val {
                    ws.ldp.w_out[i] = fac * y[i];
                }
                return (xnorm, mode);
            }
        }
        ws.ldp.x_out[..n].fill(ZERO);
        ws.ldp.w_out[..m_val].fill(ZERO);
        return (ZERO, mode);
    }

    ws.ldp.x_out[..n].fill(ZERO);
    ws.ldp.w_out[..m_val].fill(ZERO);
    (ZERO, mode)
}

/// Wrapper that calls [`bvls::bvls_wrapper`] and returns the NNLS-compatible
/// 4-tuple `(x, rnorm, w, mode)`.
///
/// Accepts a column-major `ColMat` and passes it directly to the BVLS solver
/// which now uses column-major storage natively.
fn bvls_wrapper_nnls(
    a: &mut ColMat,
    m_val: usize,
    n: usize,
    b: &mut Vec<f64>,
    max_iter: usize,
) -> (Vec<f64>, f64, Vec<f64>, i32) {
    let mut b_sub = b[..m_val].to_vec();
    let (x_out, rnorm, mode) = bvls::bvls_wrapper(a, m_val, n, &mut b_sub, max_iter);
    let w_out = vec![0.0; m_val];
    (x_out, rnorm, w_out, mode)
}

// ===========================================================================
// LSI — Least Squares with Inequality constraints
// ===========================================================================

/// LSI with pre-allocated workspace.
///
/// Takes `&mut LsiChainWs` which contains LSI-level temporaries and
/// the LDP+NNLS sub-workspace chain.
///
/// Accepts E and G as column-major `ColMat`, matching the native storage
/// of the upstream LSEI solver.
pub fn lsi_ws(
    e: &ColMat,
    f: &[f64],
    g: &ColMat,
    h: &[f64],
    _le: usize,
    me: usize,
    _lg: usize,
    mg: usize,
    n: usize,
    max_iter_ls: usize,
    nnls_mode: NnlsMode,
    ws: &mut LsiChainWs,
) -> (f64, i32) {
    profile_section!("lsi_ws");

    ws.lsi.prepare(me, mg, n);

    // Copy input ColMat data into workspace ColMat buffers.
    // Both source and destination are column-major, so copy is direct.
    {
        profile_section!("lsi_matrix_copy");
        // If e_cm was pre-populated by the caller (lsq_ws), skip the E copy.
        if ws.lsi.e_cm_ready {
            ws.lsi.e_cm_ready = false;
        } else {
            ws.lsi.e_cm.resize_uninit(me, n);
            for j in 0..n {
                // Both source and dest are column-major: direct column copy
                ws.lsi.e_cm.col_mut(j)[..me].copy_from_slice(&e.col(j)[..me]);
            }
        }

        // Copy G column-major → column-major directly.
        if ws.lsi.g_cm_ready {
            ws.lsi.g_cm_ready = false;
        } else {
            ws.lsi.g_cm.resize_uninit(mg, n);
            for j in 0..n {
                ws.lsi.g_cm.col_mut(j)[..mg].copy_from_slice(&g.col(j)[..mg]);
            }
        }

        ws.lsi.f_arr[..me].copy_from_slice(&f[..me]);
        ws.lsi.h_arr[..mg].copy_from_slice(&h[..mg]);
    }

    // -----------------------------------------------------------------------
    // QR-factor E via Householder and apply to f (column-major: stride=1)
    // -----------------------------------------------------------------------
    profile_section!("lsi_qr_factor");
    let e_rows = ws.lsi.e_cm.rows;

    // Use LAPACK dgeqr2 + dorm2r for blocked, cache-optimised QR when BLAS is available.
    // Falls back to pure-Rust Householder without BLAS or on failure.
    #[cfg(feature = "blas")]
    let lapack_ok = crate::lapack::lsi_qr_factor_lapack(
        me,
        n,
        &mut ws.lsi.e_cm.data,
        e_rows,
        &mut ws.lsi.f_arr,
        &mut ws.lsi.tau,
        &mut ws.lsi.lapack_work,
        &mut ws.lsi.lapack_lwork,
        &mut ws.lsi.lapack_cached_me,
        &mut ws.lsi.lapack_cached_n,
    );
    #[cfg(not(feature = "blas"))]
    let lapack_ok = false;

    if !lapack_ok {
        // Pure-Rust fallback: element-wise Householder QR
        for i_f in 0..n {
            let col = ws.lsi.e_cm.col_mut(i_f);
            let up = h12_construct(i_f, i_f + 1, me, col, 1);

            let b_val = up * ws.lsi.e_cm.data[i_f * e_rows + i_f];
            if b_val < ZERO {
                let inv_b = ONE / b_val;
                let tail = i_f + 1;
                let tail_len = if me > tail { me - tail } else { 0 };
                for jj in (i_f + 1)..n {
                    let j_off = jj * e_rows;
                    let (left, right) = ws.lsi.e_cm.data.split_at_mut(j_off);
                    let u_col = &left[i_f * e_rows..i_f * e_rows + e_rows];
                    let c_col = &mut right[..e_rows];
                    let mut sm = up * c_col[i_f];
                    let u_tail = &u_col[tail..tail + tail_len];
                    let c_tail = &c_col[tail..tail + tail_len];
                    for (u_v, c_v) in u_tail.iter().zip(c_tail.iter()) {
                        sm += *u_v * *c_v;
                    }
                    if sm.abs() > ZERO {
                        let sm = sm * inv_b;
                        c_col[i_f] += sm * up;
                        let c_tail = &mut c_col[tail..tail + tail_len];
                        for (c_v, u_v) in c_tail.iter_mut().zip(u_tail.iter()) {
                            *c_v += sm * *u_v;
                        }
                    }
                }
            }

            let col = ws.lsi.e_cm.col(i_f);
            h12_apply(i_f, i_f + 1, me, col, 1, up, &mut ws.lsi.f_arr, 1, 1, 1);
        }
    }

    // -----------------------------------------------------------------------
    // Check rank of E and transform G, h into the reduced space
    // -----------------------------------------------------------------------
    let mode = 5; // rank-deficient E

    for j_f in 0..n {
        let diag = ws.lsi.e_cm.data[j_f * e_rows + j_f];
        if diag.abs() < EPMACH || diag.is_nan() {
            ws.lsi.x_out[..n].fill(ZERO);
            return (ZERO, mode);
        }
        ws.lsi.inv_diag[j_f] = ONE / diag;
    }

    if mg > 0 {
        profile_section!("lsi_g_transform");

        // Solve G_reduced = G * R^{-1} via dtrsm on macOS, pure-Rust on other platforms.
        // g_cm is already in column-major format on all platforms.
        #[cfg(feature = "blas")]
        {
            crate::lapack::colmajor_dtrsm_right_upper(
                mg,
                n,
                &ws.lsi.e_cm.data,
                e_rows,
                &mut ws.lsi.g_cm.data,
                mg,
            );
        }

        #[cfg(not(feature = "blas"))]
        {
            for j_f in 0..n {
                let inv_d = ws.lsi.inv_diag[j_f];
                let g_col_j = ws.lsi.g_cm.col_mut(j_f);
                for v in g_col_j.iter_mut() {
                    *v *= inv_d;
                }

                for k_f in (j_f + 1)..n {
                    let r_jk = ws.lsi.e_cm[(j_f, k_f)];
                    if r_jk.abs() > ZERO {
                        let k_off = k_f * mg;
                        let (left, right) = ws.lsi.g_cm.data.split_at_mut(k_off);
                        let g_col_j_sl = &left[j_f * mg..j_f * mg + mg];
                        let g_col_k_sl = &mut right[..mg];
                        for (gk, gj) in g_col_k_sl.iter_mut().zip(g_col_j_sl.iter()) {
                            *gk -= *gj * r_jk;
                        }
                    }
                }
            }
        }

        // h -= G * f  (column-major dgemv — BLAS L2 on macOS)
        crate::lapack::colmajor_gemv_sub(
            mg,
            n,
            &ws.lsi.g_cm.data,
            ws.lsi.g_cm.rows,
            &ws.lsi.f_arr,
            &mut ws.lsi.h_arr,
        );
    }

    // -----------------------------------------------------------------------
    // Solve the reduced inequality-constrained problem via LDP.
    // g_cm (column-major) is passed directly — no back-conversion needed.
    // -----------------------------------------------------------------------
    let (mut xnorm, mode) = ldp_ws(
        &ws.lsi.g_cm,
        mg,
        n,
        &ws.lsi.h_arr[..mg],
        max_iter_ls,
        nnls_mode,
        &mut ws.inner,
    );

    if mode == 1 {
        ws.lsi.x_out[..n].copy_from_slice(&ws.inner.ldp.x_out[..n]);
        daxpy(n, ONE, &ws.lsi.f_arr, 1, &mut ws.lsi.x_out, 1);

        // Solve R * x = rhs via back-substitution (R is upper-triangular in e_cm).
        #[cfg(feature = "blas")]
        {
            crate::lapack::colmajor_dtrsv_upper(
                n,
                &ws.lsi.e_cm.data,
                e_rows,
                &mut ws.lsi.x_out[..n],
            );
        }
        #[cfg(not(feature = "blas"))]
        {
            for i_f in (0..n).rev() {
                let mut dot_val = ZERO;
                for k in (i_f + 1)..n {
                    dot_val += ws.lsi.e_cm[(i_f, k)] * ws.lsi.x_out[k];
                }
                let diag = ws.lsi.e_cm[(i_f, i_f)];
                ws.lsi.x_out[i_f] = (ws.lsi.x_out[i_f] - dot_val) / diag;
            }
        }

        let t = dnrm2(me - n, &ws.lsi.f_arr[n..], 1);
        xnorm = (xnorm * xnorm + t * t).sqrt();

        return (xnorm, mode);
    }

    ws.lsi.x_out[..n].fill(ZERO);
    (ZERO, mode)
}

// ===========================================================================
// LSEI — Least Squares with Equality and Inequality constraints
// ===========================================================================

/// LSEI with pre-allocated workspace.
///
/// Takes `&mut LseiChainWs` which contains LSEI-level temporaries and
/// the LSI+LDP+NNLS sub-workspace chain.
///
/// Accepts C, E, G as column-major `ColMat`.  Internally converts to
/// row-major workspace copies (`Mat`, crate-private) for the Householder
/// triangularisation of C (which requires contiguous row access).
/// Sub-matrices passed to `lsi_ws` are `ColMat`.
pub fn lsei_ws(
    c: &ColMat,
    d: &[f64],
    e: &ColMat,
    f: &[f64],
    g: &ColMat,
    h: &[f64],
    _lc: usize,
    mc: usize,
    _le: usize,
    me: usize,
    _lg: usize,
    mg: usize,
    n: usize,
    max_iter_ls: usize,
    nnls_mode: NnlsMode,
    ws: &mut LseiChainWs,
) -> (f64, i32) {
    profile_section!("lsei_ws");

    ws.lsei.prepare(mc, me, mg, n);

    let mut mode = 2;
    let mut xnorm = ZERO;

    if mc > n {
        ws.lsei.x_out[..n].fill(ZERO);
        ws.lsei.w_out[..mc + mg].fill(ZERO);
        return (ZERO, mode);
    }

    let l = n - mc;

    // -----------------------------------------------------------------------
    // Fast path when mc == 0 (no equality constraints).
    // Pass E, G (ColMat) directly to lsi_ws — no copies needed.
    // -----------------------------------------------------------------------
    if mc == 0 {
        if mg > 0 {
            ws.lsei.h_arr[..mg].copy_from_slice(&h[..mg]);
            let (xn, m2) = lsi_ws(
                e,
                f,
                g,
                &ws.lsei.h_arr,
                me,
                me,
                mg,
                mg,
                n,
                max_iter_ls,
                nnls_mode,
                &mut ws.inner,
            );
            mode = m2;
            xnorm = xn;
            ws.lsei.x_out[..n].copy_from_slice(&ws.inner.lsi.x_out[..n]);
            ws.lsei.w_out[..mg].copy_from_slice(&ws.inner.inner.ldp.w_out[..mg]);
            return (xnorm, mode);
        } else {
            // mg == 0, mc == 0: unconstrained HFTI solve
            mode = 7;
            let tau_val = EPMACH.sqrt();
            let hfti_rows = me.max(n);
            ws.lsei.hfti_a_tmp.resize_zero(hfti_rows, n);
            let e_rows = me.min(e.rows);
            let e_cols = n.min(e.cols);
            for j in 0..e_cols {
                let src = e.col(j);
                let dst = ws.lsei.hfti_a_tmp.col_mut(j);
                dst[..e_rows].copy_from_slice(&src[..e_rows]);
            }
            let rows = me.max(n);
            ws.lsei.b_hfti.resize_zero(rows, 1);
            let b_col = ws.lsei.b_hfti.col_mut(0);
            b_col[..me].copy_from_slice(&f[..me]);
            let (krank, rnorm_arr) = hfti(
                &mut ws.lsei.hfti_a_tmp,
                me,
                n,
                &mut ws.lsei.b_hfti,
                1,
                tau_val,
            );
            xnorm = rnorm_arr[0];
            let b_col = ws.lsei.b_hfti.col(0);
            ws.lsei.x_out[..n].copy_from_slice(&b_col[..n]);
            if krank != n {
                return (xnorm, mode);
            }
            mode = 1;
            return (xnorm, mode);
        }
    }

    // Fill mutable row-major copies from input ColMat matrices.
    // The Householder triangularisation of C operates row-major.
    {
        let mc1 = mc.max(1);
        ws.lsei.c_mat.resize_zero(mc1, n);
        let c_rows = mc.min(c.rows);
        let c_cols = n.min(c.cols);
        for i in 0..c_rows {
            for j in 0..c_cols {
                ws.lsei.c_mat[(i, j)] = c[(i, j)];
            }
        }
    }

    {
        let me1 = me.max(1);
        ws.lsei.e_mat.resize_zero(me1, n);
        let e_rows = me.min(e.rows);
        let e_cols = n.min(e.cols);
        for i in 0..e_rows {
            for j in 0..e_cols {
                ws.lsei.e_mat[(i, j)] = e[(i, j)];
            }
        }
    }

    {
        let mg1 = mg.max(1);
        ws.lsei.g_mat.resize_zero(mg1, n);
        let g_rows = mg.min(g.rows);
        let g_cols = n.min(g.cols);
        for i in 0..g_rows {
            for j in 0..g_cols {
                ws.lsei.g_mat[(i, j)] = g[(i, j)];
            }
        }
    }

    if mc > 0 {
        ws.lsei.d_arr[..mc].copy_from_slice(&d[..mc]);
    }
    ws.lsei.f_arr[..me].copy_from_slice(&f[..me]);
    if mg > 0 {
        ws.lsei.h_arr[..mg].copy_from_slice(&h[..mg]);
    }

    let c_stride = ws.lsei.c_mat.stride;

    // -----------------------------------------------------------------------
    // Step 1: Triangularise C and apply Householder factors to E and G
    // (operates on row-major workspace copies)
    // -----------------------------------------------------------------------
    for i_f in 0..mc {
        let j_f = (i_f + 1).min(ws.lsei.c_mat.rows - 1);
        let ncv_remaining = mc.saturating_sub(1 + i_f);

        let up = h12_construct(
            i_f,
            i_f + 1,
            n,
            &mut ws.lsei.c_mat.data[i_f * c_stride..],
            1,
        );
        ws.lsei.w_hh[i_f] = up;

        if ncv_remaining > 0 {
            ws.lsei.u_buf[..n]
                .copy_from_slice(&ws.lsei.c_mat.data[i_f * c_stride..i_f * c_stride + n]);
            h12_apply(
                i_f,
                i_f + 1,
                n,
                &ws.lsei.u_buf,
                1,
                up,
                &mut ws.lsei.c_mat.data[j_f * c_stride..],
                1,
                c_stride,
                ncv_remaining,
            );
        }

        let e_stride_val = ws.lsei.e_mat.stride;
        h12_apply(
            i_f,
            i_f + 1,
            n,
            &ws.lsei.c_mat.data[i_f * c_stride..],
            1,
            ws.lsei.w_hh[i_f],
            &mut ws.lsei.e_mat.data[..],
            1,
            e_stride_val,
            me,
        );

        let g_stride_val = ws.lsei.g_mat.stride;
        h12_apply(
            i_f,
            i_f + 1,
            n,
            &ws.lsei.c_mat.data[i_f * c_stride..],
            1,
            ws.lsei.w_hh[i_f],
            &mut ws.lsei.g_mat.data[..],
            1,
            g_stride_val,
            mg,
        );
    }

    // -----------------------------------------------------------------------
    // Step 2: Solve C x = d (upper-triangular)
    // -----------------------------------------------------------------------
    mode = 6;
    for i_f in 0..mc {
        let diag = ws.lsei.c_mat[(i_f, i_f)];
        if diag.abs() < EPMACH {
            return (ZERO, mode);
        }
        let dot_val = ddot(
            i_f,
            &ws.lsei.c_mat.data[i_f * c_stride..],
            1,
            &ws.lsei.x_out,
            1,
        );
        ws.lsei.x_out[i_f] = (ws.lsei.d_arr[i_f] - dot_val) / diag;
    }
    mode = 1;

    ws.lsei.w_out[mc..mc + mg].fill(ZERO);

    // -----------------------------------------------------------------------
    // Step 3: Solve the reduced problem in the null-space of C
    // -----------------------------------------------------------------------
    if mc != n {
        let e_stride_val = ws.lsei.e_mat.stride;
        for i_f in 0..me {
            let dot_val = ddot(
                mc,
                &ws.lsei.e_mat.data[i_f * e_stride_val..],
                1,
                &ws.lsei.x_out,
                1,
            );
            ws.lsei.f_reduced[i_f] = ws.lsei.f_arr[i_f] - dot_val;
        }

        // Extract e_sub (me × l) from row-major e_mat columns [mc..n] → ColMat.
        ws.lsei.e_sub.resize_uninit(me, l);
        {
            let e_stride = ws.lsei.e_mat.stride;
            for j in 0..l {
                for i in 0..me {
                    ws.lsei.e_sub[(i, j)] = ws.lsei.e_mat.data[i * e_stride + mc + j];
                }
            }
        }

        // Extract g_sub (mg × l) from row-major g_mat columns [mc..n] → ColMat.
        ws.lsei.g_sub.resize_zero(mg.max(1), l);
        if mg > 0 {
            let g_stride = ws.lsei.g_mat.stride;
            for j in 0..l {
                for i in 0..mg {
                    ws.lsei.g_sub[(i, j)] = ws.lsei.g_mat.data[i * g_stride + mc + j];
                }
            }
        }

        if mg > 0 {
            let g_stride_val = ws.lsei.g_mat.stride;
            for i_f in 0..mg {
                let dot_val = ddot(
                    mc,
                    &ws.lsei.g_mat.data[i_f * g_stride_val..],
                    1,
                    &ws.lsei.x_out,
                    1,
                );
                ws.lsei.h_arr[i_f] -= dot_val;
            }

            // Call lsi_ws with ColMat sub-matrices.
            let (xn, m2) = lsi_ws(
                &ws.lsei.e_sub,
                &ws.lsei.f_reduced,
                &ws.lsei.g_sub,
                &ws.lsei.h_arr,
                me,
                me,
                mg,
                mg,
                l,
                max_iter_ls,
                nnls_mode,
                &mut ws.inner,
            );
            mode = m2;
            xnorm = xn;
            for i in 0..l {
                ws.lsei.x_out[mc + i] = ws.inner.lsi.x_out[i];
            }
            for i in 0..mg {
                ws.lsei.w_out[mc + i] = ws.inner.inner.ldp.w_out[i];
            }

            if mc == 0 {
                return (xnorm, mode);
            }
            let t = dnrm2(mc, &ws.lsei.x_out, 1);
            xnorm = (xnorm * xnorm + t * t).sqrt();
            if mode != 1 {
                return (ZERO, mode);
            }
        } else {
            // mg == 0: unconstrained HFTI on the reduced problem.
            // e_sub is already a ColMat workspace — pass directly.
            mode = 7;
            let tau_val = EPMACH.sqrt();
            let rows = me.max(l);
            ws.lsei.b_hfti.resize_zero(rows, 1);
            let b_col = ws.lsei.b_hfti.col_mut(0);
            b_col[..me].copy_from_slice(&ws.lsei.f_reduced[..me]);
            let (krank, rnorm_arr) =
                hfti(&mut ws.lsei.e_sub, me, l, &mut ws.lsei.b_hfti, 1, tau_val);
            xnorm = rnorm_arr[0];
            let b_col = ws.lsei.b_hfti.col(0);
            ws.lsei.x_out[mc..mc + l].copy_from_slice(&b_col[..l]);
            if krank != l {
                return (xnorm, mode);
            }
            mode = 1;
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Compute Lagrange multipliers and back-transform x
    // -----------------------------------------------------------------------
    let e_stride_val = ws.lsei.e_mat.stride;
    for i_f in 0..me {
        let dot_val = ddot(
            n,
            &ws.lsei.e_mat.data[i_f * e_stride_val..],
            1,
            &ws.lsei.x_out,
            1,
        );
        ws.lsei.f_arr[i_f] = dot_val - ws.lsei.f_arr[i_f];
    }

    for i_f in 0..mc {
        let mut dot_e = ZERO;
        for k in 0..me {
            dot_e += ws.lsei.e_mat[(k, i_f)] * ws.lsei.f_arr[k];
        }
        let mut dot_g = ZERO;
        for k in 0..mg {
            dot_g += ws.lsei.g_mat[(k, i_f)] * ws.lsei.w_out[mc + k];
        }
        ws.lsei.d_arr[i_f] = dot_e - dot_g;
    }

    for i_f in (0..mc).rev() {
        h12_apply(
            i_f,
            i_f + 1,
            n,
            &ws.lsei.c_mat.data[i_f * c_stride..],
            1,
            ws.lsei.w_hh[i_f],
            &mut ws.lsei.x_out,
            1,
            1,
            1,
        );
    }

    for i_f in (0..mc).rev() {
        let j_f = (i_f + 1).min(mc - 1);
        let dot_val = ddot(
            mc.saturating_sub(1 + i_f),
            &ws.lsei.c_mat.data[j_f * c_stride + i_f..],
            c_stride as i32,
            &ws.lsei.w_out[j_f..],
            1,
        );
        let diag = ws.lsei.c_mat[(i_f, i_f)];
        ws.lsei.w_out[i_f] = (ws.lsei.d_arr[i_f] - dot_val) / diag;
    }

    (xnorm, mode)
}

// ===========================================================================
// LSQ — QP sub-problem solver
// ===========================================================================

/// LSQ with pre-allocated workspace — the hot-path variant called from `slsqpb`.
///
/// Takes `&mut LsWorkspace` which contains LSQ-level temporaries and
/// the full LSEI+LSI+LDP+NNLS sub-workspace chain.
///
/// Accepts `a` as a column-major `ColMatView`, matching the native storage
/// of the upstream SQP driver.
pub fn lsq_ws(
    m: usize,
    meq: usize,
    n: usize,
    nl: usize,
    _la: usize,
    l: &[f64],
    g: &[f64],
    a: ColMatView<'_>,
    b: &[f64],
    xl: &[f64],
    xu: &[f64],
    s: &mut [f64],
    y: &mut [f64],
    max_iter_ls: usize,
    nnls_mode: NnlsMode,
    infbnd: f64,
    ws: &mut LsWorkspace,
) -> i32 {
    profile_section!("lsq_ws");
    let n1 = n + 1;
    let mineq = m - meq;

    let n2 = if n1 * n / 2 + 1 == nl {
        0_usize
    } else {
        1_usize
    };
    let n3 = n - n2;

    // Count active bound constraints to determine mg_actual up front.
    let mut mg_actual = mineq;
    for i in 0..n {
        if !xl[i].is_nan() && xl[i] > -infbnd {
            mg_actual += 1;
        }
    }
    for i in 0..n {
        if !xu[i].is_nan() && xu[i] < infbnd {
            mg_actual += 1;
        }
    }

    ws.lsq.prepare(n, meq, mg_actual);

    // -----------------------------------------------------------------------
    // Recover E (n×n Cholesky-like factor) and f (n,) from packed l and g
    // -----------------------------------------------------------------------
    let e_flat = &mut ws.lsq.e_flat[..n * n1];
    for v in e_flat.iter_mut() {
        *v = 0.0;
    }
    let f_arr = &mut ws.lsq.f_arr[..n];
    for v in f_arr.iter_mut() {
        *v = 0.0;
    }

    let mut i2: usize = 0;
    let mut i3: usize = 0;
    let mut i4: usize = 0;

    for i in 0..n3 {
        let i1 = n1 - (i + 1);
        let diag = l[i2].sqrt();

        for k in 0..i1 {
            e_flat[i3 + k] = ZERO;
        }
        for k in 0..(i1 - n2) {
            e_flat[i3 + k * n] = l[i2 + k];
        }
        for k in 0..(i1 - n2) {
            e_flat[i3 + k * n] *= diag;
        }
        e_flat[i3] = diag;

        let mut dot_val = ZERO;
        for k in 0..i {
            dot_val += e_flat[i4 + k] * f_arr[k];
        }
        f_arr[i] = (g[i] - dot_val) / diag;

        i2 += i1 - n2;
        i3 += n1;
        i4 += n;
    }

    if n2 == 1 {
        e_flat[i3] = l[nl - 1];
        for k in 0..n3 {
            e_flat[i4 + k] = ZERO;
        }
        f_arr[n - 1] = ZERO;
    }

    for v in f_arr.iter_mut() {
        *v = -*v;
    }

    // When meq==0 and there are inequality/bound constraints (mg_actual > 0),
    // pre-populate the LSI workspace's e_cm directly from e_flat (which is
    // already in column-major order with leading dim = nu).
    // This avoids any transpose.
    // When mg_actual == 0, the HFTI fallback in lsei_ws needs a ColMat e,
    // so we build e_cm at the LSQ level.
    if meq == 0 && mg_actual > 0 {
        let e_cm = &mut ws.inner.inner.lsi.e_cm;
        e_cm.resize_uninit(n, n);
        e_cm.data[..n * n].copy_from_slice(&e_flat[..n * n]);
        ws.inner.inner.lsi.e_cm_ready = true;
    } else {
        // Build column-major e_cm from e_flat (which is already column-major
        // with leading dim = nu, but stride = n+1). Copy the n×n sub-block.
        ws.lsq.e_cm.resize_uninit(n, n);
        ws.lsq.e_cm.data[..n * n].copy_from_slice(&e_flat[..n * n]);
    }

    // -----------------------------------------------------------------------
    // Build C, d (equality constraints) — column-major ColMat
    // -----------------------------------------------------------------------
    let meq_max = meq.max(1);
    ws.lsq.c_cm.resize_zero(meq_max, n);
    ws.lsq.d_arr[..meq_max].fill(0.0);
    if meq > 0 {
        for i in 0..meq {
            for j in 0..n {
                ws.lsq.c_cm[(i, j)] = a[(i, j)];
            }
            ws.lsq.d_arr[i] = -b[i];
        }
    }

    // -----------------------------------------------------------------------
    // Build G, h (inequality constraints + variable bounds)
    // -----------------------------------------------------------------------
    ws.lsq.h_arr[..mg_actual.max(1)].fill(0.0);

    // When mequ == 0 and mg_actual > 0, build g_cm directly in column-major
    // format in the LSI workspace, skipping the intermediate g_cm copy.
    let build_g_cm_direct = meq == 0 && mg_actual > 0;

    if build_g_cm_direct {
        let mg_max = mg_actual.max(1);
        let g_cm = &mut ws.inner.inner.lsi.g_cm;
        g_cm.resize_uninit(mg_max, n);
        // Zero entire g_cm buffer
        g_cm.data[..mg_max * n].fill(0.0);

        let mut gi = 0usize;

        // Inequality constraints from a — copy column-by-column.
        for j in 0..n {
            let col = g_cm.col_mut(j);
            for i in 0..mineq {
                col[i] = a[(meq + i, j)];
            }
        }
        for i in 0..mineq {
            ws.lsq.h_arr[gi] = -b[meq + i];
            gi += 1;
        }

        // Lower bound constraints: row gi has +1 in column i.
        for i in 0..n {
            if !xl[i].is_nan() && xl[i] > -infbnd {
                g_cm[(gi, i)] = ONE;
                ws.lsq.h_arr[gi] = xl[i];
                gi += 1;
            }
        }

        // Upper bound constraints: row gi has -1 in column i.
        for i in 0..n {
            if !xu[i].is_nan() && xu[i] < infbnd {
                g_cm[(gi, i)] = -ONE;
                ws.lsq.h_arr[gi] = -xu[i];
                gi += 1;
            }
        }

        debug_assert_eq!(gi, mg_actual);
        ws.inner.inner.lsi.g_cm_ready = true;

        // g_cm still needs valid dimensions for the lsei_ws API,
        // but its data won't be read since g_cm_ready is set.
        ws.lsq.g_cm.resize_uninit(mg_max, n);
    } else {
        ws.lsq.g_cm.resize_zero(mg_actual.max(1), n);

        let mut gi = 0; // running index into G rows

        // Inequality constraints from a.
        for i in 0..mineq {
            for j in 0..n {
                ws.lsq.g_cm[(gi, j)] = a[(meq + i, j)];
            }
            ws.lsq.h_arr[gi] = -b[meq + i];
            gi += 1;
        }

        // Lower bound constraints.
        for i in 0..n {
            if !xl[i].is_nan() && xl[i] > -infbnd {
                // g_cm is already zeroed by resize_zero
                ws.lsq.g_cm[(gi, i)] = ONE;
                ws.lsq.h_arr[gi] = xl[i];
                gi += 1;
            }
        }

        // Upper bound constraints.
        for i in 0..n {
            if !xu[i].is_nan() && xu[i] < infbnd {
                // g_cm is already zeroed by resize_zero
                ws.lsq.g_cm[(gi, i)] = -ONE;
                ws.lsq.h_arr[gi] = -xu[i];
                gi += 1;
            }
        }

        debug_assert_eq!(gi, mg_actual);
    }

    // -----------------------------------------------------------------------
    // Solve via LSEI.
    // ws.lsq.* matrices (ColMat) are passed as immutable refs to lsei_ws;
    // ws.inner (LseiChainWs) is passed as mutable — disjoint from ws.lsq.
    // -----------------------------------------------------------------------
    let (_xnorm, mode) = lsei_ws(
        &ws.lsq.c_cm,
        &ws.lsq.d_arr[..meq_max],
        &ws.lsq.e_cm,
        &ws.lsq.f_arr[..n],
        &ws.lsq.g_cm,
        &ws.lsq.h_arr[..mg_actual.max(1)],
        meq_max,
        meq,
        n,
        n,
        mg_actual.max(1),
        mg_actual,
        n,
        max_iter_ls,
        nnls_mode,
        &mut ws.inner,
    );

    if mode == 1 {
        s[..n].copy_from_slice(&ws.inner.lsei.x_out[..n]);
        let wlen = ws.inner.lsei.w_out.len().min(meq + mineq + 2 * mineq);
        if wlen >= m {
            y[..m].copy_from_slice(&ws.inner.lsei.w_out[..m]);
        } else {
            y[..wlen].copy_from_slice(&ws.inner.lsei.w_out[..wlen]);
            for i in wlen..m {
                y[i] = ZERO;
            }
        }
        if n3 > 0 {
            let end = (m + 2 * n3).min(y.len());
            for i in m..end {
                y[i] = f64::NAN;
            }
        }
        enforce_bounds(&mut s[..n], xl, xu, infbnd);
    }

    mode
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_types::ColMat;

    // -- NNLS tests ----------------------------------------------------------

    #[test]
    fn nnls_identity_positive_rhs() {
        // A = I₃, b = [1, 2, 3] → x* = [1, 2, 3]
        let mut a = ColMat::from_vv(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let mut b = vec![1.0, 2.0, 3.0];
        let mut ws = NnlsWs::new();
        let (_rnorm, mode) = nnls_ws(&mut a, 3, 3, &mut b, 0, &mut ws);
        assert_eq!(mode, 1);
        assert!((ws.x[0] - 1.0).abs() < 1e-10);
        assert!((ws.x[1] - 2.0).abs() < 1e-10);
        assert!((ws.x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn nnls_identity_negative_rhs() {
        // A = I₃, b = [-1, 2, -3] → x* = [0, 2, 0]
        let mut a = ColMat::from_vv(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let mut b = vec![-1.0, 2.0, -3.0];
        let mut ws = NnlsWs::new();
        let (_rnorm, mode) = nnls_ws(&mut a, 3, 3, &mut b, 0, &mut ws);
        assert_eq!(mode, 1);
        assert!((ws.x[0]).abs() < 1e-10);
        assert!((ws.x[1] - 2.0).abs() < 1e-10);
        assert!((ws.x[2]).abs() < 1e-10);
    }

    #[test]
    fn nnls_overdetermined() {
        // 4×2 system, positive solution
        let mut a = ColMat::from_vv(&[
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, -1.0],
        ]);
        let mut b = vec![1.0, 2.0, 3.0, -1.0];
        let mut ws = NnlsWs::new();
        let (_rnorm, mode) = nnls_ws(&mut a, 4, 2, &mut b, 0, &mut ws);
        assert_eq!(mode, 1);
        assert!(ws.x[0] >= -1e-10);
        assert!(ws.x[1] >= -1e-10);
    }

    #[test]
    fn nnls_empty_returns_mode2() {
        let mut a = ColMat::zeros(0, 0);
        let mut b: Vec<f64> = vec![];
        let mut ws = NnlsWs::new();
        let (_rnorm, mode) = nnls_ws(&mut a, 0, 0, &mut b, 0, &mut ws);
        assert_eq!(mode, 2);
    }

    #[test]
    fn nnls_known_2x2() {
        // A = [[1, 1], [1, -1]], b = [2, 0] → unconstrained sol = [1, 1], both ≥0 → same
        let mut a = ColMat::from_vv(&[vec![1.0, 1.0], vec![1.0, -1.0]]);
        let mut b = vec![2.0, 0.0];
        let mut ws = NnlsWs::new();
        let (_rnorm, mode) = nnls_ws(&mut a, 2, 2, &mut b, 0, &mut ws);
        assert_eq!(mode, 1);
        assert!((ws.x[0] - 1.0).abs() < 1e-8);
        assert!((ws.x[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn nnls_2x2_positive_rnorm() {
        // A = I₂, b = [1, 2] → x* = [1, 2], rnorm = 0
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![1.0, 2.0];
        let mut ws = NnlsWs::new();
        let (rnorm, mode) = nnls_ws(&mut a, 2, 2, &mut b, 0, &mut ws);
        assert_eq!(mode, 1);
        assert!((ws.x[0] - 1.0).abs() < 1e-10);
        assert!((ws.x[1] - 2.0).abs() < 1e-10);
        assert!(rnorm.abs() < 1e-10);
    }

    #[test]
    fn nnls_2x2_negative_rhs_all_zero() {
        // A = I₂, b = [-1, -2] → x* = [0, 0] (non-negativity forces zero)
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mut b = vec![-1.0, -2.0];
        let mut ws = NnlsWs::new();
        let (_rnorm, mode) = nnls_ws(&mut a, 2, 2, &mut b, 0, &mut ws);
        assert_eq!(mode, 1);
        assert!((ws.x[0]).abs() < 1e-10);
        assert!((ws.x[1]).abs() < 1e-10);
    }

    #[test]
    fn nnls_3x2_overdetermined() {
        // 3×2 overdetermined system with positive solution
        let mut a = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        let mut b = vec![1.0, 2.0, 4.0];
        let mut ws = NnlsWs::new();
        let (_rnorm, mode) = nnls_ws(&mut a, 3, 2, &mut b, 0, &mut ws);
        assert_eq!(mode, 1);
        assert!(ws.x[0] > 0.0);
        assert!(ws.x[1] > 0.0);
    }

    #[test]
    fn nnls_bad_dims_returns_mode2() {
        // n=0, m=0 → mode 2
        let mut a = ColMat::zeros(1, 1);
        let mut b = vec![0.0];
        let mut ws = NnlsWs::new();
        let (_rnorm, mode) = nnls_ws(&mut a, 0, 0, &mut b, 0, &mut ws);
        assert_eq!(mode, 2);
    }

    #[test]
    fn nnls_vs_scipy_reference() {
        // Cross-check against scipy.optimize.nnls reference values.
        // Matrix generated with numpy rng seed 42: rng.standard_normal((5, 3)).
        let mut a = ColMat::from_vv(&[
            vec![0.30471707975443135, -1.0399841062404955, 0.7504511958064572],
            vec![0.9405647163912139, -1.9510351886538364, -1.302179506862318],
            vec![
                0.12784040316728537,
                -0.3162425923435822,
                -0.016801157504288795,
            ],
            vec![-0.85304392757358, 0.8793979748628286, 0.7777919354289483],
            vec![0.06603069756121605, 1.1272412069680329, 0.4675093422520456],
        ]);
        let mut b = vec![
            -0.8592924628832382,
            0.36875078408249884,
            -0.9588826008289989,
            0.8784503013072725,
            -0.049925910986252896,
        ];
        // scipy reference: x = [0.0, 0.16974564627857777, 0.0], rnorm = 1.5379737415231234
        let mut ws = NnlsWs::new();
        let (rnorm, mode) = nnls_ws(&mut a, 5, 3, &mut b, 0, &mut ws);
        assert_eq!(mode, 1);
        assert!((ws.x[0]).abs() < 1e-8, "x[0] = {}", ws.x[0]);
        assert!(
            (ws.x[1] - 0.16974564627857777).abs() < 1e-8,
            "x[1] = {}",
            ws.x[1]
        );
        assert!((ws.x[2]).abs() < 1e-8, "x[2] = {}", ws.x[2]);
        assert!(
            (rnorm - 1.5379737415231234).abs() < 1e-8,
            "rnorm = {}",
            rnorm
        );
    }

    // -- LDP tests -----------------------------------------------------------

    #[test]
    fn ldp_unconstrained_origin() {
        // No constraints → solution is the origin
        let g = ColMat::zeros(1, 2);
        let h = vec![0.0];
        let mut ws = LdpChainWs::new();
        let (_xnorm, mode) = ldp_ws(&g, 0, 2, &h, 0, NnlsMode::Nnls, &mut ws);
        assert_eq!(mode, 1);
        assert!((ws.ldp.x_out[0]).abs() < 1e-10);
        assert!((ws.ldp.x_out[1]).abs() < 1e-10);
    }

    #[test]
    fn ldp_single_active_constraint() {
        // min ½‖x‖² s.t. x₁ ≥ 1  → x* = [1, 0]
        let g = ColMat::from_vv(&[vec![1.0, 0.0]]);
        let h = vec![1.0];
        let mut ws = LdpChainWs::new();
        let (_xnorm, mode) = ldp_ws(&g, 1, 2, &h, 0, NnlsMode::Nnls, &mut ws);
        assert_eq!(mode, 1);
        assert!(
            (ws.ldp.x_out[0] - 1.0).abs() < 1e-8,
            "x[0] = {}",
            ws.ldp.x_out[0]
        );
        assert!((ws.ldp.x_out[1]).abs() < 1e-8, "x[1] = {}", ws.ldp.x_out[1]);
    }

    #[test]
    fn ldp_zero_variables() {
        let g = ColMat::zeros(0, 0);
        let h: Vec<f64> = vec![];
        let mut ws = LdpChainWs::new();
        let (_xnorm, mode) = ldp_ws(&g, 0, 0, &h, 0, NnlsMode::Nnls, &mut ws);
        assert_eq!(mode, 2);
    }

    #[test]
    fn ldp_no_constraints() {
        let g = ColMat::zeros(1, 3);
        let h = vec![0.0];
        let mut ws = LdpChainWs::new();
        let (_xnorm, mode) = ldp_ws(&g, 0, 3, &h, 0, NnlsMode::Nnls, &mut ws);
        assert_eq!(mode, 1);
    }

    // -- LSI tests -----------------------------------------------------------

    #[test]
    fn lsi_unconstrained_least_squares() {
        // E = I₂, f = [1, 2], no inequality → x* = [1, 2]
        let e = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let f = vec![1.0, 2.0];
        let g = ColMat::zeros(1, 2);
        let h = vec![0.0];
        let mut ws = LsiChainWs::new();
        let (_xnorm, mode) = lsi_ws(&e, &f, &g, &h, 2, 2, 1, 0, 2, 0, NnlsMode::Nnls, &mut ws);
        assert_eq!(mode, 1);
        assert!((ws.lsi.x_out[0] - 1.0).abs() < 1e-8);
        assert!((ws.lsi.x_out[1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn lsi_with_active_inequality() {
        // min ‖x - [2,2]‖ s.t. x₁+x₂ ≥ 5  → x* = [2.5, 2.5]
        let e = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let f = vec![2.0, 2.0];
        let g = ColMat::from_vv(&[vec![1.0, 1.0]]);
        let h = vec![5.0];
        let mut ws = LsiChainWs::new();
        let (_xnorm, mode) = lsi_ws(&e, &f, &g, &h, 2, 2, 1, 1, 2, 0, NnlsMode::Nnls, &mut ws);
        assert_eq!(mode, 1);
        assert!(
            (ws.lsi.x_out[0] - 2.5).abs() < 1e-6,
            "x[0] = {}",
            ws.lsi.x_out[0]
        );
        assert!(
            (ws.lsi.x_out[1] - 2.5).abs() < 1e-6,
            "x[1] = {}",
            ws.lsi.x_out[1]
        );
    }

    // -- LSEI tests ----------------------------------------------------------

    #[test]
    fn lsei_equality_only() {
        // min ‖x - [3,4]‖ s.t. x₁+x₂ = 5, no inequality  → x* = [2, 3]  (proj onto line)
        let c = ColMat::from_vv(&[vec![1.0, 1.0]]);
        let d = vec![5.0];
        let e = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let f_arr = vec![3.0, 4.0];
        let g = ColMat::zeros(1, 2);
        let h = vec![0.0];
        let mut ws = LseiChainWs::new();
        let (_xnorm, mode) = lsei_ws(
            &c,
            &d,
            &e,
            &f_arr,
            &g,
            &h,
            1,
            1,
            2,
            2,
            1,
            0,
            2,
            0,
            NnlsMode::Nnls,
            &mut ws,
        );
        assert_eq!(mode, 1);
        assert!(
            (ws.lsei.x_out[0] + ws.lsei.x_out[1] - 5.0).abs() < 1e-8,
            "constraint violated"
        );
        // Projected solution: x = [3,4] - ((3+4-5)/2)*[1,1] = [3,4] - [1,1] = [2,3]
        assert!(
            (ws.lsei.x_out[0] - 2.0).abs() < 1e-6,
            "x[0] = {}",
            ws.lsei.x_out[0]
        );
        assert!(
            (ws.lsei.x_out[1] - 3.0).abs() < 1e-6,
            "x[1] = {}",
            ws.lsei.x_out[1]
        );
    }

    #[test]
    fn lsei_too_many_equalities() {
        // mc > n → mode 2
        let c = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        let d = vec![1.0, 1.0, 2.0];
        let e = ColMat::from_vv(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let f_arr = vec![0.0, 0.0];
        let g = ColMat::zeros(1, 2);
        let h = vec![0.0];
        let mut ws = LseiChainWs::new();
        let (_xnorm, mode) = lsei_ws(
            &c,
            &d,
            &e,
            &f_arr,
            &g,
            &h,
            3,
            3,
            2,
            2,
            1,
            0,
            2,
            0,
            NnlsMode::Nnls,
            &mut ws,
        );
        assert_eq!(mode, 2);
    }
}
