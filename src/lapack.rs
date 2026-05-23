//! Platform-optimised BLAS/LAPACK bindings.
//!
//! When the `blas` Cargo feature is enabled, this module links to the
//! platform-native BLAS/LAPACK library:
//!
//! - **macOS**: Apple Accelerate framework (AMX/NEON on Apple Silicon, AVX on Intel)
//! - **Linux**: OpenBLAS (`apt install libopenblas-dev` / `dnf install openblas-devel`)
//! - **Windows**: OpenBLAS (if available; set `OPENBLAS_PATH` env var)
//!
//! Without the `blas` feature, all functions fall through to pure-Rust
//! implementations in [`support`](crate::support) — correct but slower.
//!
//! ## Provided BLAS routines
//!
//! | Level | Function | Purpose |
//! |-------|----------|---------|
//! | 1 | `accel_daxpy` | y += α·x |
//! | 1 | `accel_dcopy` | y := x |
//! | 1 | `accel_ddot` | x·y |
//! | 1 | `accel_dnrm2` | ‖x‖₂ |
//! | 1 | `accel_dscal` | x *= α |
//! | 2 | `colmajor_dgemv` | y := α·A·x + β·y (column-major) |
//! | 2 | `colmajor_dtrsv_upper` | Solve T·x = b (upper triangular, column-major) |
//!
//! ## Design
//!
//! All functions accept the same Rust-friendly signatures used throughout
//! the SLSQP solver.  The `blas` feature flag controls whether the FFI
//! path is compiled in.  When not available, every function falls through
//! to the pure-Rust implementation.
//!
//! ## Building
//!
//! ```sh
//! # macOS (Accelerate is always available):
//! cargo build --features blas
//!
//! # Linux (requires libopenblas-dev):
//! cargo build --features blas
//!
//! # Any platform (pure-Rust fallback, no system dependencies):
//! cargo build
//! ```

// ===========================================================================
// FFI declarations for CBLAS/LAPACK (feature-gated)
// ===========================================================================

#[cfg(feature = "blas")]
#[allow(dead_code)]
mod ffi {
    //! Raw C function declarations for CBLAS and LAPACK.
    //!
    //! These symbols are provided by all standard BLAS/LAPACK implementations:
    //! - Apple Accelerate (macOS)
    //! - OpenBLAS (Linux, Windows)
    //! - Intel MKL
    //! - Netlib reference BLAS/LAPACK

    use std::os::raw::c_int;

    #[cfg_attr(target_os = "macos", link(name = "Accelerate", kind = "framework"))]
    #[cfg_attr(target_os = "linux", link(name = "openblas", kind = "dylib"))]
    #[cfg_attr(target_os = "windows", link(name = "openblas", kind = "static"))]
    unsafe extern "C" {
        // BLAS Level 1
        pub fn cblas_daxpy(
            n: c_int,
            alpha: f64,
            x: *const f64,
            incx: c_int,
            y: *mut f64,
            incy: c_int,
        );
        pub fn cblas_dcopy(n: c_int, x: *const f64, incx: c_int, y: *mut f64, incy: c_int);
        pub fn cblas_ddot(n: c_int, x: *const f64, incx: c_int, y: *const f64, incy: c_int) -> f64;
        pub fn cblas_dnrm2(n: c_int, x: *const f64, incx: c_int) -> f64;
        pub fn cblas_dscal(n: c_int, alpha: f64, x: *mut f64, incx: c_int);
        pub fn cblas_drot(
            n: c_int,
            x: *mut f64,
            incx: c_int,
            y: *mut f64,
            incy: c_int,
            c: f64,
            s: f64,
        );

        // BLAS Level 2
        // CblasRowMajor=101, CblasColMajor=102
        // CblasNoTrans=111, CblasTrans=112
        // CblasUpper=121, CblasLower=122
        // CblasNonUnit=131, CblasUnit=132
        pub fn cblas_dgemv(
            order: c_int,
            trans: c_int,
            m: c_int,
            n: c_int,
            alpha: f64,
            a: *const f64,
            lda: c_int,
            x: *const f64,
            incx: c_int,
            beta: f64,
            y: *mut f64,
            incy: c_int,
        );
        pub fn cblas_dtrsv(
            order: c_int,
            uplo: c_int,
            trans: c_int,
            diag: c_int,
            n: c_int,
            a: *const f64,
            lda: c_int,
            x: *mut f64,
            incx: c_int,
        );
        pub fn cblas_dger(
            order: c_int,
            m: c_int,
            n: c_int,
            alpha: f64,
            x: *const f64,
            incx: c_int,
            y: *const f64,
            incy: c_int,
            a: *mut f64,
            lda: c_int,
        );
        pub fn cblas_dtpmv(
            order: c_int,
            uplo: c_int,
            trans: c_int,
            diag: c_int,
            n: c_int,
            ap: *const f64,
            x: *mut f64,
            incx: c_int,
        );

        // BLAS Level 3 — triangular solve with multiple RHS
        // CblasLeft=141, CblasRight=142
        pub fn cblas_dtrsm(
            order: c_int,
            side: c_int,
            uplo: c_int,
            trans_a: c_int,
            diag: c_int,
            m: c_int,
            n: c_int,
            alpha: f64,
            a: *const f64,
            lda: c_int,
            b: *mut f64,
            ldb: c_int,
        );

        // LAPACK — unpivoted QR factorization (blocked, BLAS L3)
        pub fn dgeqrf_(
            m: *const c_int,
            n: *const c_int,
            a: *mut f64,
            lda: *const c_int,
            tau: *mut f64,
            work: *mut f64,
            lwork: *const c_int,
            info: *mut c_int,
        );

        // LAPACK — unpivoted QR factorization (unblocked, BLAS L2)
        pub fn dgeqr2_(
            m: *const c_int,
            n: *const c_int,
            a: *mut f64,
            lda: *const c_int,
            tau: *mut f64,
            work: *mut f64,
            info: *mut c_int,
        );

        // LAPACK — column-pivoted QR (blocked, BLAS L3)
        pub fn dgeqp3_(
            m: *const c_int,
            n: *const c_int,
            a: *mut f64,
            lda: *const c_int,
            jpvt: *mut c_int,
            tau: *mut f64,
            work: *mut f64,
            lwork: *const c_int,
            info: *mut c_int,
        );

        // LAPACK — apply Q from QR to a matrix (blocked, BLAS L3)
        pub fn dormqr_(
            side: *const u8,
            trans: *const u8,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            a: *const f64,
            lda: *const c_int,
            tau: *const f64,
            c: *mut f64,
            ldc: *const c_int,
            work: *mut f64,
            lwork: *const c_int,
            info: *mut c_int,
        );

        // LAPACK — apply Q from QR to a matrix (unblocked, BLAS L2)
        pub fn dorm2r_(
            side: *const u8,
            trans: *const u8,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            a: *const f64,
            lda: *const c_int,
            tau: *const f64,
            c: *mut f64,
            ldc: *const c_int,
            work: *mut f64,
            info: *mut c_int,
        );

        // LAPACK — solve triangular system with multiple RHS
        pub fn dtrtrs_(
            uplo: *const u8,
            trans: *const u8,
            diag: *const u8,
            n: *const c_int,
            nrhs: *const c_int,
            a: *const f64,
            lda: *const c_int,
            b: *mut f64,
            ldb: *const c_int,
            info: *mut c_int,
        );
    }

    // CBLAS enum constants
    pub const CBLAS_ROW_MAJOR: c_int = 101;
    pub const CBLAS_COL_MAJOR: c_int = 102;
    pub const CBLAS_NO_TRANS: c_int = 111;
    pub const CBLAS_TRANS: c_int = 112;
    pub const CBLAS_UPPER: c_int = 121;
    pub const CBLAS_LOWER: c_int = 122;
    pub const CBLAS_NON_UNIT: c_int = 131;
    pub const CBLAS_UNIT: c_int = 132;
    pub const CBLAS_RIGHT: c_int = 142;
}

// ===========================================================================
// Minimum problem size for BLAS dispatch
// ===========================================================================

/// Below this threshold, the pure-Rust implementations are faster due to
/// lower call overhead (no C ABI boundary, better inlining).
/// Native BLAS shines for n ≥ 32 where SIMD pipelines are fully utilised.
#[cfg(feature = "blas")]
const ACCEL_THRESHOLD: usize = 32;

// ===========================================================================
// BLAS Level 1 — optimised wrappers
// ===========================================================================

/// Optimised `daxpy`: `dy := dy + da * dx`.
///
/// Dispatches to native BLAS for unit-stride vectors with n ≥ 32.
/// Falls back to the pure-Rust implementation otherwise.
#[inline]
pub fn accel_daxpy(n: usize, da: f64, dx: &[f64], incx: i32, dy: &mut [f64], incy: i32) {
    if n == 0 || da == 0.0 {
        return;
    }

    #[cfg(feature = "blas")]
    {
        if n >= ACCEL_THRESHOLD && incx >= 1 && incy >= 1 {
            unsafe {
                ffi::cblas_daxpy(n as i32, da, dx.as_ptr(), incx, dy.as_mut_ptr(), incy);
            }
            return;
        }
    }

    // Fallback: pure Rust
    crate::support::daxpy(n, da, dx, incx, dy, incy);
}

/// Optimised `dcopy`: `dy := dx`.
///
/// Dispatches to native BLAS for unit-stride vectors with n ≥ 32.
#[inline]
#[allow(dead_code)]
pub fn accel_dcopy(n: usize, dx: &[f64], incx: i32, dy: &mut [f64], incy: i32) {
    if n == 0 {
        return;
    }

    #[cfg(feature = "blas")]
    {
        if n >= ACCEL_THRESHOLD && incx >= 1 && incy >= 1 {
            unsafe {
                ffi::cblas_dcopy(n as i32, dx.as_ptr(), incx, dy.as_mut_ptr(), incy);
            }
            return;
        }
    }

    crate::support::dcopy(n, dx, incx, dy, incy);
}

/// Optimised `ddot`: inner product `dx · dy`.
///
/// Dispatches to native BLAS for vectors with n ≥ 32.
#[inline]
pub fn accel_ddot(n: usize, dx: &[f64], incx: i32, dy: &[f64], incy: i32) -> f64 {
    if n == 0 {
        return 0.0;
    }

    #[cfg(feature = "blas")]
    {
        if n >= ACCEL_THRESHOLD && incx >= 1 && incy >= 1 {
            return unsafe { ffi::cblas_ddot(n as i32, dx.as_ptr(), incx, dy.as_ptr(), incy) };
        }
    }

    crate::support::ddot(n, dx, incx, dy, incy)
}

/// Optimised `dnrm2`: Euclidean norm `‖x‖₂`.
///
/// Dispatches to native BLAS for n ≥ 32.
/// BLAS `dnrm2` uses the same scaled-sum-of-squares algorithm,
/// so numerical stability is preserved.
#[inline]
pub fn accel_dnrm2(n: usize, x: &[f64], incx: i32) -> f64 {
    if n == 0 || incx < 1 {
        return 0.0;
    }

    #[cfg(feature = "blas")]
    {
        if n >= ACCEL_THRESHOLD {
            return unsafe { ffi::cblas_dnrm2(n as i32, x.as_ptr(), incx) };
        }
    }

    crate::support::dnrm2(n, x, incx)
}

/// Optimised `dscal`: `dx := da * dx`.
///
/// Dispatches to native BLAS for n ≥ 32.
#[inline]
#[allow(dead_code)]
pub fn accel_dscal(n: usize, da: f64, dx: &mut [f64], incx: i32) {
    if n == 0 || incx <= 0 {
        return;
    }

    #[cfg(feature = "blas")]
    {
        if n >= ACCEL_THRESHOLD {
            unsafe {
                ffi::cblas_dscal(n as i32, da, dx.as_mut_ptr(), incx);
            }
            return;
        }
    }

    crate::support::dscal(n, da, dx, incx);
}

/// Optimised `drot`: apply Givens rotation `[c s; -s c]` to vectors x, y.
///
/// Dispatches to native BLAS for n ≥ 32.
#[inline]
#[allow(dead_code)]
pub fn accel_drot(n: usize, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, c: f64, s: f64) {
    if n == 0 {
        return;
    }

    #[cfg(feature = "blas")]
    {
        if n >= ACCEL_THRESHOLD {
            unsafe {
                ffi::cblas_drot(n as i32, x.as_mut_ptr(), incx, y.as_mut_ptr(), incy, c, s);
            }
            return;
        }
    }

    // Pure-Rust fallback
    let incx = incx as usize;
    let incy = incy as usize;
    for i in 0..n {
        let xi = x[i * incx];
        let yi = y[i * incy];
        x[i * incx] = c * xi + s * yi;
        y[i * incy] = -s * xi + c * yi;
    }
}

// ===========================================================================
// Givens rotation on column-major rows
// ===========================================================================

/// Apply a Givens rotation `[c s; -s c]` to rows `j` and `j+1` across all
/// columns of a column-major matrix, **skipping** column `skip_col`.
///
/// This replaces the scalar loop:
/// ```text
/// for l in 0..ncols {
///     if l != skip_col {
///         let temp = data[l*rows + j];
///         data[l*rows + j]     =  c * temp + s * data[l*rows + j + 1];
///         data[l*rows + j + 1] = -s * temp + c * data[l*rows + j + 1];
///     }
/// }
/// ```
/// with BLAS `drot` calls (two segments: before and after `skip_col`).
#[inline]
pub fn drot_cm_rows(
    data: &mut [f64],
    rows: usize,
    ncols: usize,
    j: usize,
    cc: f64,
    ss: f64,
    skip_col: usize,
) {
    // Segment 1: columns 0..skip_col
    if skip_col > 0 {
        drot_cm_segment(data, rows, j, 0, skip_col, cc, ss);
    }
    // Segment 2: columns skip_col+1..ncols
    if skip_col + 1 < ncols {
        drot_cm_segment(data, rows, j, skip_col + 1, ncols - skip_col - 1, cc, ss);
    }
}

/// Apply Givens rotation to a contiguous segment of columns starting at
/// `col_start`, covering `n` columns.  Rows `j` and `j+1` are modified.
#[inline]
fn drot_cm_segment(
    data: &mut [f64],
    rows: usize,
    j: usize,
    col_start: usize,
    n: usize,
    cc: f64,
    ss: f64,
) {
    if n == 0 {
        return;
    }
    let start_x = col_start * rows + j;
    let start_y = col_start * rows + j + 1;

    #[cfg(feature = "blas")]
    {
        if n >= ACCEL_THRESHOLD {
            unsafe {
                let xp = data.as_mut_ptr().add(start_x);
                let yp = data.as_mut_ptr().add(start_y);
                ffi::cblas_drot(n as i32, xp, rows as i32, yp, rows as i32, cc, ss);
            }
            return;
        }
    }

    // Pure-Rust fallback — reads both values before writing to avoid aliasing.
    for i in 0..n {
        let xi = start_x + i * rows;
        let yi = start_y + i * rows;
        let temp_x = data[xi];
        let temp_y = data[yi];
        data[xi] = cc * temp_x + ss * temp_y;
        data[yi] = -ss * temp_x + cc * temp_y;
    }
}

// ===========================================================================
// BLAS Level 2 — matrix-vector operations (column-major)
// ===========================================================================

/// Column-major BLAS `dgemv`: `y := alpha * op(A) * x + beta * y`
/// where A is stored column-major with leading dimension `lda`.
#[cfg(feature = "blas")]
#[inline]
pub fn colmajor_dgemv(
    trans: bool,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    let trans_flag = if trans {
        ffi::CBLAS_TRANS
    } else {
        ffi::CBLAS_NO_TRANS
    };
    unsafe {
        ffi::cblas_dgemv(
            ffi::CBLAS_COL_MAJOR,
            trans_flag,
            m as i32,
            n as i32,
            alpha,
            a.as_ptr(),
            lda as i32,
            x.as_ptr(),
            incx,
            beta,
            y.as_mut_ptr(),
            incy,
        );
    }
}

/// Solve `T * x = b` for upper triangular T stored in **column-major** layout.
///
/// Dispatches to BLAS `cblas_dtrsv` with `CblasColMajor`.
/// `x` is both the RHS on entry and the solution on exit.
#[cfg(feature = "blas")]
#[inline]
pub fn colmajor_dtrsv_upper(n: usize, a: &[f64], lda: usize, x: &mut [f64]) {
    unsafe {
        ffi::cblas_dtrsv(
            ffi::CBLAS_COL_MAJOR,
            ffi::CBLAS_UPPER,
            ffi::CBLAS_NO_TRANS,
            ffi::CBLAS_NON_UNIT,
            n as i32,
            a.as_ptr(),
            lda as i32,
            x.as_mut_ptr(),
            1,
        );
    }
}

// ===========================================================================
// Packed triangular matrix-vector multiply (BLAS Level 2)
// ===========================================================================

/// Compute `v = L * D * L^T * s` using BLAS packed triangular multiply,
/// where L*D*L^T is stored in row-major upper-triangular packed format `ap`.
///
/// The diagonal of `ap` holds D; off-diagonal holds L^T (= U with unit diag).
/// Three operations:
///   1. `v = U * s`  (unit-diagonal upper triangular)
///   2. `v *= D`      (diagonal scale)
///   3. `v = U^T * v` (unit-diagonal, i.e. lower triangular)
///
/// Returns `true` if BLAS was used, `false` if caller should use scalar fallback.
#[cfg(feature = "blas")]
pub fn bfgs_ldlt_multiply_blas(n: usize, ap: &[f64], s: &[f64], v: &mut [f64]) -> bool {
    if n < ACCEL_THRESHOLD {
        return false;
    }

    use std::os::raw::c_int;
    let ni = n as c_int;

    // v = s (copy)
    v[..n].copy_from_slice(&s[..n]);

    // Pass 1: v = U * v (unit diagonal, no-trans)
    unsafe {
        ffi::cblas_dtpmv(
            ffi::CBLAS_ROW_MAJOR,
            ffi::CBLAS_UPPER,
            ffi::CBLAS_NO_TRANS,
            ffi::CBLAS_UNIT,
            ni,
            ap.as_ptr(),
            v.as_mut_ptr(),
            1,
        );
    }

    // Pass 2: v[i] *= D[i] — diagonal elements at packed positions
    let mut k = 0usize;
    for i in 0..n {
        v[i] *= ap[k];
        k += n - i; // skip to next diagonal
    }

    // Pass 3: v = U^T * v (unit diagonal, trans)
    unsafe {
        ffi::cblas_dtpmv(
            ffi::CBLAS_ROW_MAJOR,
            ffi::CBLAS_UPPER,
            ffi::CBLAS_TRANS,
            ffi::CBLAS_UNIT,
            ni,
            ap.as_ptr(),
            v.as_mut_ptr(),
            1,
        );
    }

    true
}

// ===========================================================================
// Batch Householder application (using BLAS Level 2)
// ===========================================================================

#[allow(dead_code)]
/// Apply a Householder reflector `H = I - (1/b) u u^T` to multiple
/// contiguous column vectors simultaneously using BLAS Level 2 operations.
///
/// This is equivalent to calling [`h12_apply`](crate::core_basic::h12_apply)
/// with `ice=1, icv=stride, iue=1` but batches the dot products and updates
/// into `dgemv` + `dger` calls that the native BLAS can vectorise efficiently.
///
/// # Requirements
/// - `u` must be contiguous (stride 1).
/// - Target vectors in `c` must be contiguous columns with stride `icv`.
/// - `lpivot`, `l1`, `m` use the same 0-based convention as `h12_apply`.
///
/// # Arguments
/// * `lpivot` — Pivot index.
/// * `l1`     — First tail index.
/// * `m_val`  — Exclusive upper bound on reflector elements.
/// * `u`      — Householder vector (contiguous, length ≥ m).
/// * `up`     — Scalar from `h12_construct`.
/// * `c`      — Target column-major data (modified in-place).
/// * `c_rows` — Number of rows per column (stride between columns).
/// * `ncv`    — Number of target vectors to transform.
#[inline]
pub fn h12_apply_batch(
    lpivot: usize,
    l1: usize,
    m_val: usize,
    u: &[f64],
    up: f64,
    c: &mut [f64],
    c_rows: usize,
    ncv: usize,
) {
    if lpivot >= l1 || l1 >= m_val || ncv == 0 {
        return;
    }

    let b = up * u[lpivot];
    if b >= 0.0 || u[lpivot].abs() == 0.0 {
        return;
    }
    let inv_b = 1.0 / b;
    let tail_len = m_val - l1;

    // For each target vector j (column j of c), we need:
    //   sm_j = up * c[lpivot + j*c_rows] + u[l1..m]^T · c[l1..m, j]
    //   c[lpivot + j*c_rows] += sm_j * inv_b * up
    //   c[l1..m, j] += sm_j * inv_b * u[l1..m]
    //
    // We can batch the dot products and rank-1 update if we treat the
    // target columns as a matrix C_tail (tail_len × ncv, col-major with stride c_rows).

    // Step 1: Compute sm[j] for each target column.
    // This is: sm = up * c_pivot_row + u_tail^T * C_tail
    // where c_pivot_row is the row of c at index lpivot across all ncv columns.
    //
    // For small ncv (typical case in SLSQP: ncv < 50), a loop is efficient.
    // For large ncv, we could use dgemv but the memory layout requires careful handling.

    // Allocate sm on the stack for small ncv, heap for large.
    // In practice ncv < 500 for SLSQP.
    let mut sm_buf = [0.0_f64; 512];
    let sm: &mut [f64] = if ncv <= 512 {
        &mut sm_buf[..ncv]
    } else {
        // Fallback for very large ncv — shouldn't happen in practice
        return fallback_h12_apply_loop(lpivot, l1, m_val, u, up, c, c_rows, ncv);
    };

    let u_tail = &u[l1..l1 + tail_len];

    for j in 0..ncv {
        let c_base = j * c_rows;
        let mut s = up * c[c_base + lpivot];
        let c_tail = &c[c_base + l1..c_base + l1 + tail_len];

        // Dot product: u_tail · c_tail
        // Use native BLAS for large tail
        #[cfg(feature = "blas")]
        {
            if tail_len >= ACCEL_THRESHOLD {
                s += unsafe {
                    ffi::cblas_ddot(tail_len as i32, u_tail.as_ptr(), 1, c_tail.as_ptr(), 1)
                };
                sm[j] = s;
                continue;
            }
        }

        // Pure-Rust fast path for contiguous data
        for k in 0..tail_len {
            s += u_tail[k] * c_tail[k];
        }
        sm[j] = s;
    }

    // Step 2: Apply update: c -= (sm * inv_b) * u
    for j in 0..ncv {
        if sm[j].abs() == 0.0 {
            continue;
        }
        let factor = sm[j] * inv_b;
        let c_base = j * c_rows;
        c[c_base + lpivot] += factor * up;

        let c_tail = &mut c[c_base + l1..c_base + l1 + tail_len];

        #[cfg(feature = "blas")]
        {
            if tail_len >= ACCEL_THRESHOLD {
                unsafe {
                    ffi::cblas_daxpy(
                        tail_len as i32,
                        factor,
                        u_tail.as_ptr(),
                        1,
                        c_tail.as_mut_ptr(),
                        1,
                    );
                }
                continue;
            }
        }

        // Pure-Rust fast path
        for k in 0..tail_len {
            c_tail[k] += factor * u_tail[k];
        }
    }
}

/// Fallback loop-based Householder apply for ncv > 512 (extremely rare).
#[cold]
#[allow(dead_code)]
fn fallback_h12_apply_loop(
    lpivot: usize,
    l1: usize,
    m_val: usize,
    u: &[f64],
    up: f64,
    c: &mut [f64],
    c_rows: usize,
    ncv: usize,
) {
    let b = up * u[lpivot];
    if b >= 0.0 {
        return;
    }
    let inv_b = 1.0 / b;
    let tail_len = m_val - l1;
    let u_tail = &u[l1..l1 + tail_len];

    for j in 0..ncv {
        let c_base = j * c_rows;
        let mut sm = up * c[c_base + lpivot];
        for k in 0..tail_len {
            sm += u_tail[k] * c[c_base + l1 + k];
        }
        if sm.abs() > 0.0 {
            let factor = sm * inv_b;
            c[c_base + lpivot] += factor * up;
            for k in 0..tail_len {
                c[c_base + l1 + k] += factor * u_tail[k];
            }
        }
    }
}

/// Column-major matrix-vector product: `h[0..m] -= G[0..m, 0..n] * f[0..n]`.
///
/// G is stored column-major in `g_data` with `g_rows` rows per column.
/// Dispatches to native BLAS `cblas_dgemv` when available; pure-Rust fallback otherwise.
pub fn colmajor_gemv_sub(
    m: usize,
    n: usize,
    g_data: &[f64],
    g_rows: usize,
    f: &[f64],
    h: &mut [f64],
) {
    #[cfg(feature = "blas")]
    if m >= ACCEL_THRESHOLD || n >= ACCEL_THRESHOLD {
        // h = h - G*f  ⟹  h = (-1)*G*f + 1*h
        unsafe {
            ffi::cblas_dgemv(
                ffi::CBLAS_COL_MAJOR,
                ffi::CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                -1.0,
                g_data.as_ptr(),
                g_rows as i32,
                f.as_ptr(),
                1,
                1.0,
                h.as_mut_ptr(),
                1,
            );
        }
        return;
    }

    // Pure-Rust fallback
    for i in 0..m {
        let mut dot = 0.0;
        for k in 0..n {
            dot += g_data[k * g_rows + i] * f[k];
        }
        h[i] -= dot;
    }
}

// ===========================================================================
// BLAS Level 3 — column-major right-side triangular solve for LSI
// ===========================================================================

/// Compute `B := B * R⁻¹` where R is upper triangular (column-major).
///
/// This wraps `cblas_dtrsm(ColMajor, Right, Upper, NoTrans, NonUnit, ...)`
/// and is used by `lsi_g_transform` to replace the hand-written forward
/// substitution loop.
///
/// # Arguments
/// * `m` — number of rows of B (= mg, number of inequality constraints)
/// * `n` — number of columns of B / order of R (= n, number of variables)
/// * `r_data` — column-major R data (upper triangle of QR result)
/// * `r_lda` — leading dimension of R (= me, number of rows of E)
/// * `b_data` — column-major B data (G matrix), modified in-place
/// * `b_ldb` — leading dimension of B (= mg)
#[cfg(feature = "blas")]
pub fn colmajor_dtrsm_right_upper(
    m: usize,
    n: usize,
    r_data: &[f64],
    r_lda: usize,
    b_data: &mut [f64],
    b_ldb: usize,
) {
    unsafe {
        ffi::cblas_dtrsm(
            ffi::CBLAS_COL_MAJOR,
            ffi::CBLAS_RIGHT,    // B * R^{-1}
            ffi::CBLAS_UPPER,    // R is upper triangular
            ffi::CBLAS_NO_TRANS, // no transpose on R
            ffi::CBLAS_NON_UNIT, // diagonal is not unit
            m as i32,            // rows of B
            n as i32,            // cols of B / order of R
            1.0,                 // alpha = 1
            r_data.as_ptr(),
            r_lda as i32,
            b_data.as_mut_ptr(),
            b_ldb as i32,
        );
    }
}

// ===========================================================================
// LAPACK-based LSI QR factorization (unblocked dgeqr2 + dorm2r)
// ===========================================================================

/// LAPACK-accelerated QR factorization for the LSI sub-problem.
///
/// Uses unblocked `dgeqr2` + `dorm2r` (matching scipy's implementation)
/// which requires no workspace queries — only a fixed-size work buffer.
///
/// After this call:
/// - `e_cm_data[0..me*n]` contains R in the upper triangle (column-major,
///   leading dimension `me`). The lower triangle holds Householder vectors.
/// - `f[0..me]` is overwritten with `Q^T * f`.
///
/// # Arguments
/// * `me` — number of rows of E
/// * `n`  — number of columns of E
/// * `e_cm_data` — column-major E data, length ≥ me*n (modified in-place)
/// * `e_cm_rows` — leading dimension of E (= me)
/// * `f` — RHS vector, length ≥ me (modified in-place: becomes Q^T f)
/// * `tau` — workspace for Householder scalars, length ≥ min(me,n)
/// * `work` — LAPACK workspace (needs only max(n, 1) elements)
///
/// # Returns
/// `true` if successful, `false` if LAPACK returned an error.
#[cfg(feature = "blas")]
pub fn lsi_qr_factor_lapack(
    me: usize,
    n: usize,
    e_cm_data: &mut [f64],
    e_cm_rows: usize,
    f: &mut [f64],
    tau: &mut Vec<f64>,
    work: &mut Vec<f64>,
    _cached_lwork: &mut i32,
    _cached_me: &mut usize,
    _cached_n: &mut usize,
) -> bool {
    use std::os::raw::c_int;

    let mi = me as c_int;
    let ni = n as c_int;
    let lda = e_cm_rows as c_int;
    let k = me.min(n);
    let mut info: c_int = 0;

    // Ensure tau is large enough
    if tau.len() < k {
        tau.resize(k, 0.0);
    }

    // dgeqr2 needs work of size n; dorm2r needs work of size 1.
    let work_needed = n.max(1);
    if work.len() < work_needed {
        work.resize(work_needed, 0.0);
    }

    let ki = k as c_int;
    let one_i: c_int = 1;
    let ldf = me as c_int;

    // ── Compute QR factorisation (unblocked) ─────────────────────────
    unsafe {
        ffi::dgeqr2_(
            &mi,
            &ni,
            e_cm_data.as_mut_ptr(),
            &lda,
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        return false;
    }

    // ── Apply Q^T to f (unblocked) ──────────────────────────────────
    unsafe {
        ffi::dorm2r_(
            b"L\0".as_ptr(),
            b"T\0".as_ptr(),
            &mi,
            &one_i,
            &ki,
            e_cm_data.as_ptr(),
            &lda,
            tau.as_ptr(),
            f.as_mut_ptr(),
            &ldf,
            work.as_mut_ptr(),
            &mut info,
        );
    }

    info == 0
}

// ===========================================================================
// LAPACK-based HFTI (blocked QR via dgeqp3 + dormqr + dtrtrs)
// ===========================================================================

/// LAPACK-accelerated rank-deficient least squares via column-pivoted QR.
///
/// This is functionally equivalent to [`hfti`](crate::core_basic::hfti) but
/// uses LAPACK's blocked algorithms (`dgeqp3`, `dormqr`, `dtrtrs`) which
/// internally call `dgemm` (BLAS Level 3) for optimal cache utilisation.
///
/// When the `blas` feature is enabled, this is dispatched from `hfti()` for
/// problems larger than a threshold.  Otherwise falls through to the pure-Rust `hfti()`.
///
/// Input data is **column-major** (`ColMat` layout): element `(i, j)` is at
/// `data[j * stride + i]`.
///
/// # Arguments
/// * `a_data`   — Column-major m×n matrix data (overwritten).
/// * `a_stride` — Column stride (leading dimension) of A.
/// * `m`        — Number of rows.
/// * `n`        — Number of columns.
/// * `b_data`   — Column-major max(m,n)×nb matrix data (overwritten with solution).
/// * `b_stride` — Column stride (leading dimension) of B.
/// * `nb`       — Number of RHS columns.
/// * `tau_tol`  — Pivot tolerance for pseudo-rank determination.
///
/// # Returns
/// `(krank, rnorm)` — pseudo-rank and per-column residual norms.
#[cfg(feature = "blas")]
pub fn hfti_lapack(
    a_data: &mut [f64],
    a_stride: usize,
    m: usize,
    n: usize,
    b_data: &mut [f64],
    b_stride: usize,
    nb: usize,
    tau_tol: f64,
) -> (usize, Vec<f64>) {
    use std::os::raw::c_int;

    let ldiag = m.min(n);
    let mut rnorm = vec![0.0; nb.max(1)];

    if ldiag == 0 {
        return (0, rnorm);
    }

    let lda = a_stride as c_int; // Column-major leading dim = a_stride (≥ m)
    let mi = m as c_int;
    let ni = n as c_int;

    // Data is already column-major — use a_data directly (no conversion needed).

    // ── Step 1: Column-pivoted QR via dgeqp3 ─────────────────────────
    let mut jpvt = vec![0_i32; n]; // 0 = free column (all pivotable)
    let mut tau = vec![0.0_f64; ldiag];
    let mut info: c_int = 0;

    // Workspace query
    let mut work_query = [0.0_f64; 1];
    let lwork_query: c_int = -1;
    unsafe {
        ffi::dgeqp3_(
            &mi,
            &ni,
            a_data.as_mut_ptr(),
            &lda,
            jpvt.as_mut_ptr(),
            tau.as_mut_ptr(),
            work_query.as_mut_ptr(),
            &lwork_query,
            &mut info,
        );
    }
    let lwork = work_query[0] as c_int;
    let mut work = vec![0.0_f64; lwork as usize];

    // Actual factorisation
    unsafe {
        ffi::dgeqp3_(
            &mi,
            &ni,
            a_data.as_mut_ptr(),
            &lda,
            jpvt.as_mut_ptr(),
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }

    if info != 0 {
        // dgeqp3 failed — return zero rank
        return (0, rnorm);
    }

    // ── Step 2: Determine pseudo-rank ────────────────────────────────
    let mut krank = 0_usize;
    for i in 0..ldiag {
        // R[i,i] in column-major = a_data[i * a_stride + i]
        if a_data[i * a_stride + i].abs() > tau_tol {
            krank = i + 1;
        } else {
            break;
        }
    }

    if krank == 0 {
        // Zero rank — zero out solution (column-major)
        for jb in 0..nb {
            for i in 0..n {
                b_data[jb * b_stride + i] = 0.0;
            }
        }
        return (0, rnorm);
    }

    // B is already column-major — use b_data directly (no conversion needed).
    let max_mn = m.max(n);
    let ldb = b_stride as c_int;

    // ── Step 3: Apply Q^T to B: B := Q^T * B ─────────────────────────
    // dormqr('L', 'T', m, nb, k, A, lda, tau, B, ldb, work, lwork, info)
    let ki = krank as c_int;
    let nbi = nb as c_int;

    // Workspace query for dormqr
    let lwork_query2: c_int = -1;
    unsafe {
        ffi::dormqr_(
            b"L\0".as_ptr(),
            b"T\0".as_ptr(),
            &mi,
            &nbi,
            &ki,
            a_data.as_ptr(),
            &lda,
            tau.as_ptr(),
            b_data.as_mut_ptr(),
            &ldb,
            work_query.as_mut_ptr(),
            &lwork_query2,
            &mut info,
        );
    }
    let lwork2 = work_query[0] as c_int;
    if (lwork2 as usize) > work.len() {
        work.resize(lwork2 as usize, 0.0);
    }

    unsafe {
        ffi::dormqr_(
            b"L\0".as_ptr(),
            b"T\0".as_ptr(),
            &mi,
            &nbi,
            &ki,
            a_data.as_ptr(),
            &lda,
            tau.as_ptr(),
            b_data.as_mut_ptr(),
            &ldb,
            work.as_mut_ptr(),
            &lwork2,
            &mut info,
        );
    }

    // ── Step 4: Residual norms from B[krank..m, :] ───────────────────
    for jb in 0..nb {
        let mut res_sq = 0.0;
        for i in krank..m {
            let v = b_data[jb * b_stride + i];
            res_sq += v * v;
        }
        rnorm[jb] = res_sq.sqrt();
    }

    // ── Step 5: Back-substitution: R[0:k,0:k] * x = B[0:k,:] ────────
    unsafe {
        ffi::dtrtrs_(
            b"U\0".as_ptr(),
            b"N\0".as_ptr(),
            b"N\0".as_ptr(),
            &ki,
            &nbi,
            a_data.as_ptr(),
            &lda,
            b_data.as_mut_ptr(),
            &ldb,
            &mut info,
        );
    }

    // ── Step 6: If rank-deficient, apply Q for minimum-norm ──────────
    if krank < n {
        // Zero out entries krank..max_mn in solution
        for jb in 0..nb {
            for i in krank..max_mn {
                b_data[jb * b_stride + i] = 0.0;
            }
        }

        // Apply Q (not Q^T): B := Q * B
        // dormqr('L', 'N', max(m,n), nb, k, ...)
        // We need to apply to max_mn rows to get the full n-dimensional solution
        let max_mn_i = max_mn as c_int;
        let lwork_query3: c_int = -1;
        unsafe {
            ffi::dormqr_(
                b"L\0".as_ptr(),
                b"N\0".as_ptr(),
                &max_mn_i,
                &nbi,
                &ki,
                a_data.as_ptr(),
                &lda,
                tau.as_ptr(),
                b_data.as_mut_ptr(),
                &ldb,
                work_query.as_mut_ptr(),
                &lwork_query3,
                &mut info,
            );
        }
        let lwork3 = work_query[0] as c_int;
        if (lwork3 as usize) > work.len() {
            work.resize(lwork3 as usize, 0.0);
        }

        unsafe {
            ffi::dormqr_(
                b"L\0".as_ptr(),
                b"N\0".as_ptr(),
                &max_mn_i,
                &nbi,
                &ki,
                a_data.as_ptr(),
                &lda,
                tau.as_ptr(),
                b_data.as_mut_ptr(),
                &ldb,
                work.as_mut_ptr(),
                &lwork3,
                &mut info,
            );
        }
    }

    // ── Step 7: Inverse column permutation ───────────────────────────
    // jpvt is 1-based from Fortran
    let mut tmp = vec![0.0_f64; max_mn];
    for jb in 0..nb {
        for j in 0..n {
            let perm_col = (jpvt[j] - 1) as usize; // 1-based → 0-based
            tmp[perm_col] = b_data[jb * b_stride + j];
        }
        // Copy back
        for j in 0..n {
            b_data[jb * b_stride + j] = tmp[j];
        }
    }

    // Results are already in b_data (column-major) — no conversion needed.

    (krank, rnorm)
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn accel_daxpy_basic() {
        let dx = vec![1.0; 100];
        let mut dy = vec![2.0; 100];
        accel_daxpy(100, 3.0, &dx, 1, &mut dy, 1);
        for &v in &dy {
            assert!((v - 5.0).abs() < TOL);
        }
    }

    #[test]
    fn accel_ddot_basic() {
        let dx = vec![1.0; 100];
        let dy = vec![2.0; 100];
        let result = accel_ddot(100, &dx, 1, &dy, 1);
        assert!((result - 200.0).abs() < TOL);
    }

    #[test]
    fn accel_dnrm2_basic() {
        let x = vec![1.0; 100];
        let result = accel_dnrm2(100, &x, 1);
        assert!((result - 10.0).abs() < TOL); // sqrt(100)
    }

    #[test]
    fn accel_dscal_basic() {
        let mut dx = vec![3.0; 100];
        accel_dscal(100, 2.0, &mut dx, 1);
        for &v in &dx {
            assert!((v - 6.0).abs() < TOL);
        }
    }

    #[test]
    fn accel_dcopy_basic() {
        let dx = vec![7.0; 100];
        let mut dy = vec![0.0; 100];
        accel_dcopy(100, &dx, 1, &mut dy, 1);
        for &v in &dy {
            assert!((v - 7.0).abs() < TOL);
        }
    }

    #[test]
    fn h12_apply_batch_basic() {
        // Construct a Householder from u = [2, 1, 1] (contiguous)
        let mut u = vec![2.0, 1.0, 1.0];
        let up = crate::core_basic::h12_construct(0, 1, 3, &mut u, 1);

        // Apply to two column vectors stored column-major
        // c = [c0_0, c0_1, c0_2, c1_0, c1_1, c1_2]
        let mut c = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        h12_apply_batch(0, 1, 3, &u, up, &mut c, 3, 2);

        // Verify norms are preserved (Householder is orthogonal)
        let n0 = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let n1 = (c[3] * c[3] + c[4] * c[4] + c[5] * c[5]).sqrt();
        assert!((n0 - 1.0).abs() < TOL, "n0 = {n0}");
        assert!((n1 - 1.0).abs() < TOL, "n1 = {n1}");

        // Apply twice — should recover original (H is involution)
        h12_apply_batch(0, 1, 3, &u, up, &mut c, 3, 2);
        assert!((c[0] - 1.0).abs() < TOL);
        assert!(c[1].abs() < TOL);
        assert!(c[2].abs() < TOL);
        assert!(c[3].abs() < TOL);
        assert!((c[4] - 1.0).abs() < TOL);
        assert!(c[5].abs() < TOL);
    }

    #[test]
    fn h12_apply_batch_matches_scalar() {
        // Build a Householder vector
        let mut u = vec![3.0, 1.0, 2.0, 0.5];
        let up = crate::core_basic::h12_construct(0, 1, 4, &mut u, 1);

        // Apply via batch
        let mut c_batch = vec![
            1.0, 2.0, 3.0, 4.0, // col 0
            5.0, 6.0, 7.0, 8.0, // col 1
            9.0, 10.0, 11.0, 12.0, // col 2
        ];
        h12_apply_batch(0, 1, 4, &u, up, &mut c_batch, 4, 3);

        // Apply via scalar h12_apply (ice=1, icv=4)
        let mut c_scalar = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        crate::core_basic::h12_apply(0, 1, 4, &u, 1, up, &mut c_scalar, 1, 4, 3);

        // Results should match
        for i in 0..12 {
            assert!(
                (c_batch[i] - c_scalar[i]).abs() < TOL,
                "mismatch at {i}: batch={} scalar={}",
                c_batch[i],
                c_scalar[i]
            );
        }
    }

    // Test small-n fallback path
    #[test]
    fn accel_daxpy_small() {
        let dx = vec![1.0, 2.0, 3.0];
        let mut dy = vec![10.0, 20.0, 30.0];
        accel_daxpy(3, 2.0, &dx, 1, &mut dy, 1);
        assert!((dy[0] - 12.0).abs() < TOL);
        assert!((dy[1] - 24.0).abs() < TOL);
        assert!((dy[2] - 36.0).abs() < TOL);
    }

    #[test]
    fn accel_ddot_small() {
        let result = accel_ddot(3, &[1.0, 2.0, 3.0], 1, &[4.0, 5.0, 6.0], 1);
        assert!((result - 32.0).abs() < TOL);
    }
}
