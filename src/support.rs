//! BLAS Level 1 routines and numerical constants.
//!
//! Port of `slsqp_support.f90` — provides the small set of BLAS-1 operations
//! used throughout the SLSQP solver: vector scaling, copy, dot product, norm,
//! and axpy.
//!
//! When the `blas` Cargo feature is enabled, vectors with n ≥ 32 are
//! dispatched to the platform-native BLAS library via the
//! [`lapack`](crate::lapack) module for hardware-optimised SIMD.
//! Smaller vectors and non-unit-stride cases use the pure-Rust
//! implementations below.
//!
//! The unit-stride fast paths use idiomatic iterator/zip patterns that LLVM
//! reliably auto-vectorises (SSE/AVX on x86-64, NEON on aarch64).  All
//! functions are `#[inline]` so the optimiser can specialise at each call site.

// ---------------------------------------------------------------------------
// Numerical constants (Fortran-style names kept for traceability)
// ---------------------------------------------------------------------------

/// Machine epsilon for `f64` — smallest value such that `1.0 + EPMACH > 1.0`.
pub const EPMACH: f64 = f64::EPSILON;

/// Constant `0.0`.
pub const ZERO: f64 = 0.0;

/// Constant `1.0`.
pub const ONE: f64 = 1.0;

/// Constant `2.0`.
pub const TWO: f64 = 2.0;

/// Constant `4.0`.
pub const FOUR: f64 = 4.0;

/// Constant `10.0`.
pub const TEN: f64 = 10.0;

/// Constant `100.0`.
pub const HUN: f64 = 100.0;

// ---------------------------------------------------------------------------
// BLAS Level 1 routines
// ---------------------------------------------------------------------------

/// BLAS `daxpy`: `dy := dy + da * dx` (constant times a vector plus a vector).
///
/// # Arguments
/// * `n`    — Number of elements to process.
/// * `da`   — Scalar multiplier.
/// * `dx`   — Source vector.
/// * `incx` — Stride for `dx`.
/// * `dy`   — Destination vector (modified in-place).
/// * `incy` — Stride for `dy`.
///
/// If `n == 0` or `da == 0.0` the function returns immediately.
#[inline]
pub fn daxpy(n: usize, da: f64, dx: &[f64], incx: i32, dy: &mut [f64], incy: i32) {
    if n == 0 || da == 0.0 {
        return;
    }

    // Native BLAS dispatch for large vectors with positive strides.
    #[cfg(feature = "blas")]
    if n >= 32 && incx >= 1 && incy >= 1 {
        crate::lapack::accel_daxpy(n, da, dx, incx, dy, incy);
        return;
    }


    // Fast path: unit strides — iterator zip enables LLVM auto-vectorisation.
    if incx == 1 && incy == 1 {
        dy[..n]
            .iter_mut()
            .zip(&dx[..n])
            .for_each(|(y, &x)| *y += da * x);
    } else {
        // General strided access (handles negative strides).
        let n_s = n as isize;
        let incx_s = incx as isize;
        let incy_s = incy as isize;
        let mut ix = if incx >= 0 { 0 } else { (-n_s + 1) * incx_s };
        let mut iy = if incy >= 0 { 0 } else { (-n_s + 1) * incy_s };
        for _ in 0..n {
            dy[iy as usize] += da * dx[ix as usize];
            ix += incx_s;
            iy += incy_s;
        }
    }
}

/// BLAS `dcopy`: `dy := dx` (copy a vector).
///
/// # Special behaviour
/// When `incx == 0`, fills `dy` with `dx[0]` (broadcast scalar).
///
/// # Arguments
/// * `n`    — Number of elements to copy.
/// * `dx`   — Source vector.
/// * `incx` — Stride for `dx` (0 = broadcast `dx[0]`).
/// * `dy`   — Destination vector (modified in-place).
/// * `incy` — Stride for `dy`.
#[inline]
pub fn dcopy(n: usize, dx: &[f64], incx: i32, dy: &mut [f64], incy: i32) {
    if n == 0 {
        return;
    }

    // Native BLAS dispatch for strided (non-broadcast) copies.
    #[cfg(feature = "blas")]
    if n >= 32 && incx >= 1 && incy >= 1 {
        crate::lapack::accel_dcopy(n, dx, incx, dy, incy);
        return;
    }


    if incx == 1 && incy == 1 {
        // Fast path: contiguous memcpy.
        dy[..n].copy_from_slice(&dx[..n]);
    } else if incx == 0 {
        // Broadcast: fill dy with dx[0].
        let val = dx[0];
        if incy == 1 {
            dy[..n].fill(val);
        } else {
            let n_s = n as isize;
            let incy_s = incy as isize;
            let mut iy = if incy >= 0 { 0 } else { (-n_s + 1) * incy_s };
            for _ in 0..n {
                dy[iy as usize] = val;
                iy += incy_s;
            }
        }
    } else {
        // General strided copy.
        let n_s = n as isize;
        let incx_s = incx as isize;
        let incy_s = incy as isize;
        let mut ix = if incx >= 0 { 0 } else { (-n_s + 1) * incx_s };
        let mut iy = if incy >= 0 { 0 } else { (-n_s + 1) * incy_s };
        for _ in 0..n {
            dy[iy as usize] = dx[ix as usize];
            ix += incx_s;
            iy += incy_s;
        }
    }
}

/// Fill `n` elements of `dx` with `val`, starting at index 0 with stride `incx`.
///
/// This is a safe alternative to `dcopy` with `incx == 0` when source and
/// destination would alias the same buffer.
///
/// # Arguments
/// * `n`    — Number of elements to fill.
/// * `val`  — Value to write.
/// * `dx`   — Target buffer (modified in-place).
/// * `incx` — Stride (must be > 0).
#[inline]
pub fn dfill(n: usize, val: f64, dx: &mut [f64], incx: i32) {
    if n == 0 || incx <= 0 {
        return;
    }
    let incx = incx as usize;
    if incx == 1 {
        dx[..n].fill(val);
    } else {
        let mut ix = 0usize;
        for _ in 0..n {
            dx[ix] = val;
            ix += incx;
        }
    }
}

/// BLAS `ddot`: inner (dot) product of two vectors.
///
/// Returns `sum_{i=0}^{n-1} dx[i*incx] * dy[i*incy]`.
///
/// # Arguments
/// * `n`    — Number of elements.
/// * `dx`   — First vector.
/// * `incx` — Stride for `dx`.
/// * `dy`   — Second vector.
/// * `incy` — Stride for `dy`.
#[inline]
pub fn ddot(n: usize, dx: &[f64], incx: i32, dy: &[f64], incy: i32) -> f64 {
    if n == 0 {
        return 0.0;
    }

    // Native BLAS dispatch for large vectors with positive strides.
    #[cfg(feature = "blas")]
    if n >= 32 && incx >= 1 && incy >= 1 {
        return crate::lapack::accel_ddot(n, dx, incx, dy, incy);
    }


    if incx == 1 && incy == 1 {
        // Fast path: iterator zip — LLVM auto-vectorises this into SIMD
        // multiply-accumulate (FMA on supporting targets).
        dx[..n].iter().zip(&dy[..n]).map(|(&x, &y)| x * y).sum()
    } else {
        // General strided access.
        let n_s = n as isize;
        let incx_s = incx as isize;
        let incy_s = incy as isize;
        let mut ix = if incx >= 0 { 0 } else { (-n_s + 1) * incx_s };
        let mut iy = if incy >= 0 { 0 } else { (-n_s + 1) * incy_s };
        let mut s = 0.0;
        for _ in 0..n {
            s += dx[ix as usize] * dy[iy as usize];
            ix += incx_s;
            iy += incy_s;
        }
        s
    }
}

/// Convenience wrapper: Euclidean norm of a contiguous slice.
///
/// Equivalent to `dnrm2(x.len(), x, 1)`.
#[inline]
pub fn nrm2(x: &[f64]) -> f64 {
    dnrm2(x.len(), x, 1)
}

/// BLAS `dnrm2`: Euclidean norm with stride, using a numerically stable
/// scaled-sum-of-squares algorithm that avoids overflow / underflow.
///
/// Returns `|| x ||_2 = sqrt(sum x[i*incx]^2)`.
///
/// # Algorithm
///
/// Maintains running `scale` and `ssq` such that
/// `norm = scale * sqrt(ssq)`.  Each new element either rescales upward
/// (if it exceeds the current `scale`) or accumulates into `ssq`.
///
/// # Arguments
/// * `n`    — Number of elements.
/// * `x`    — Source vector.
/// * `incx` — Stride (must be ≥ 1; returns 0 otherwise).
#[inline]
pub fn dnrm2(n: usize, x: &[f64], incx: i32) -> f64 {
    if n == 0 || incx < 1 {
        return 0.0;
    }
    if n == 1 {
        return x[0].abs();
    }

    // Native BLAS dispatch for large vectors.
    #[cfg(feature = "blas")]
    if n >= 32 {
        return crate::lapack::accel_dnrm2(n, x, incx);
    }

    let incx = incx as usize;

    // Scaled-sum-of-squares — intentionally NOT auto-vectorised because
    // the branch-heavy rescaling logic has data-dependent control flow.
    // This is the correct reference-BLAS algorithm for numerical stability.
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;

    let mut ix = 0usize;
    for _ in 0..n {
        let absxi = x[ix].abs();
        if absxi > 0.0 {
            if scale < absxi {
                // Rescale: new dominant magnitude.
                ssq = 1.0 + ssq * (scale / absxi).powi(2);
                scale = absxi;
            } else {
                // Accumulate into existing scale.
                ssq += (absxi / scale).powi(2);
            }
        }
        ix += incx;
    }
    scale * ssq.sqrt()
}

/// BLAS `dscal`: `dx := da * dx` (scale a vector by a constant).
///
/// # Arguments
/// * `n`    — Number of elements.
/// * `da`   — Scalar multiplier.
/// * `dx`   — Vector to scale (modified in-place).
/// * `incx` — Stride (must be > 0).
#[inline]
pub fn dscal(n: usize, da: f64, dx: &mut [f64], incx: i32) {
    if n == 0 || incx <= 0 {
        return;
    }

    // Native BLAS dispatch for large vectors.
    #[cfg(feature = "blas")]
    if n >= 32 {
        crate::lapack::accel_dscal(n, da, dx, incx);
        return;
    }

    let incx = incx as usize;

    if incx == 1 {
        // Fast path: iterator enables LLVM auto-vectorisation.
        dx[..n].iter_mut().for_each(|v| *v *= da);
    } else {
        // General strided scaling.
        let mut ix = 0usize;
        for _ in 0..n {
            dx[ix] *= da;
            ix += incx;
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-14;

    // -- daxpy ---------------------------------------------------------------

    #[test]
    fn daxpy_unit_stride() {
        let dx = vec![1.0, 2.0, 3.0];
        let mut dy = vec![10.0, 20.0, 30.0];
        daxpy(3, 2.0, &dx, 1, &mut dy, 1);
        assert_eq!(dy, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    fn daxpy_non_unit_stride() {
        let dx = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        let mut dy = vec![10.0, 0.0, 20.0, 0.0, 30.0];
        daxpy(3, 2.0, &dx, 2, &mut dy, 2);
        assert_eq!(dy[0], 12.0);
        assert_eq!(dy[2], 24.0);
        assert_eq!(dy[4], 36.0);
    }

    #[test]
    fn daxpy_zero_alpha_noop() {
        let dx = vec![1.0, 2.0];
        let mut dy = vec![10.0, 20.0];
        daxpy(2, 0.0, &dx, 1, &mut dy, 1);
        assert_eq!(dy, vec![10.0, 20.0]);
    }

    #[test]
    fn daxpy_n_zero_noop() {
        let dx = vec![1.0];
        let mut dy = vec![10.0];
        daxpy(0, 5.0, &dx, 1, &mut dy, 1);
        assert_eq!(dy, vec![10.0]);
    }

    #[test]
    fn daxpy_negative_stride() {
        // With negative incx=-1 and n=3: ix starts at (-3+1)*(-1)=2
        // So dx is accessed as dx[2], dx[1], dx[0] = 1.0, 2.0, 3.0
        // dy is accessed forwards: dy[0], dy[1], dy[2]
        let dx = vec![3.0, 2.0, 1.0];
        let mut dy = vec![10.0, 20.0, 30.0];
        daxpy(3, 1.0, &dx, -1, &mut dy, 1);
        assert_eq!(dy, vec![11.0, 22.0, 33.0]);
    }

    // -- dcopy ---------------------------------------------------------------

    #[test]
    fn dcopy_unit_stride() {
        let dx = vec![1.0, 2.0, 3.0];
        let mut dy = vec![0.0; 3];
        dcopy(3, &dx, 1, &mut dy, 1);
        assert_eq!(dy, dx);
    }

    #[test]
    fn dcopy_non_unit_stride() {
        let dx = vec![1.0, 99.0, 2.0, 99.0, 3.0];
        let mut dy = vec![0.0; 5];
        dcopy(3, &dx, 2, &mut dy, 2);
        assert_eq!(dy[0], 1.0);
        assert_eq!(dy[2], 2.0);
        assert_eq!(dy[4], 3.0);
    }

    #[test]
    fn dcopy_broadcast_incx_zero() {
        let dx = vec![42.0];
        let mut dy = vec![0.0; 4];
        dcopy(4, &dx, 0, &mut dy, 1);
        assert_eq!(dy, vec![42.0, 42.0, 42.0, 42.0]);
    }

    #[test]
    fn dcopy_broadcast_strided_output() {
        let dx = vec![7.0];
        let mut dy = vec![0.0; 5];
        dcopy(3, &dx, 0, &mut dy, 2);
        assert_eq!(dy[0], 7.0);
        assert_eq!(dy[1], 0.0);
        assert_eq!(dy[2], 7.0);
        assert_eq!(dy[3], 0.0);
        assert_eq!(dy[4], 7.0);
    }

    #[test]
    fn dcopy_n_zero_noop() {
        let dx = vec![1.0];
        let mut dy = vec![99.0];
        dcopy(0, &dx, 1, &mut dy, 1);
        assert_eq!(dy, vec![99.0]);
    }

    // -- ddot ----------------------------------------------------------------

    #[test]
    fn ddot_unit_stride() {
        let dx = vec![1.0, 2.0, 3.0];
        let dy = vec![4.0, 5.0, 6.0];
        let result = ddot(3, &dx, 1, &dy, 1);
        assert!((result - 32.0).abs() < TOL); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn ddot_non_unit_stride() {
        let dx = vec![1.0, 0.0, 2.0]; // stride 2: elements 1.0, 2.0
        let dy = vec![3.0, 0.0, 4.0]; // stride 2: elements 3.0, 4.0
        let result = ddot(2, &dx, 2, &dy, 2);
        assert!((result - 11.0).abs() < TOL); // 1*3 + 2*4 = 11
    }

    #[test]
    fn ddot_n_zero_returns_zero() {
        let result = ddot(0, &[1.0], 1, &[1.0], 1);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn ddot_orthogonal_vectors() {
        let dx = vec![1.0, 0.0];
        let dy = vec![0.0, 1.0];
        assert_eq!(ddot(2, &dx, 1, &dy, 1), 0.0);
    }

    // -- dnrm2 / nrm2 -------------------------------------------------------

    #[test]
    fn dnrm2_unit_stride() {
        let x = vec![3.0, 4.0];
        assert!((dnrm2(2, &x, 1) - 5.0).abs() < TOL);
    }

    #[test]
    fn dnrm2_single_element() {
        assert!((dnrm2(1, &[-7.0], 1) - 7.0).abs() < TOL);
    }

    #[test]
    fn dnrm2_strided() {
        let x = vec![3.0, 0.0, 4.0]; // stride 2: elements 3.0, 4.0
        assert!((dnrm2(2, &x, 2) - 5.0).abs() < TOL);
    }

    #[test]
    fn dnrm2_zero_vector() {
        let x = vec![0.0, 0.0, 0.0];
        assert_eq!(dnrm2(3, &x, 1), 0.0);
    }

    #[test]
    fn dnrm2_n_zero_returns_zero() {
        assert_eq!(dnrm2(0, &[1.0], 1), 0.0);
    }

    #[test]
    fn dnrm2_negative_incx_returns_zero() {
        assert_eq!(dnrm2(2, &[3.0, 4.0], -1), 0.0);
    }

    #[test]
    fn dnrm2_large_values_no_overflow() {
        // Values near sqrt(f64::MAX) — naive sum-of-squares would overflow
        let big = 1e154;
        let x = vec![big, big];
        let result = dnrm2(2, &x, 1);
        let expected = big * 2.0_f64.sqrt();
        assert!((result - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn dnrm2_small_values_no_underflow() {
        // Values near sqrt(f64::MIN_POSITIVE) — naive squaring would underflow
        let small = 1e-170;
        let x = vec![small, small, small];
        let result = dnrm2(3, &x, 1);
        let expected = small * 3.0_f64.sqrt();
        assert!((result - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn nrm2_convenience_wrapper() {
        let x = vec![3.0, 4.0];
        assert!((nrm2(&x) - 5.0).abs() < TOL);
    }

    // -- dscal ---------------------------------------------------------------

    #[test]
    fn dscal_unit_stride() {
        let mut dx = vec![1.0, 2.0, 3.0];
        dscal(3, 2.0, &mut dx, 1);
        assert_eq!(dx, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn dscal_non_unit_stride() {
        let mut dx = vec![1.0, 0.0, 2.0, 0.0, 3.0];
        dscal(3, 3.0, &mut dx, 2);
        assert_eq!(dx[0], 3.0);
        assert_eq!(dx[1], 0.0); // untouched
        assert_eq!(dx[2], 6.0);
        assert_eq!(dx[4], 9.0);
    }

    #[test]
    fn dscal_zero_alpha() {
        let mut dx = vec![1.0, 2.0, 3.0];
        dscal(3, 0.0, &mut dx, 1);
        assert_eq!(dx, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn dscal_n_zero_noop() {
        let mut dx = vec![5.0];
        dscal(0, 100.0, &mut dx, 1);
        assert_eq!(dx, vec![5.0]);
    }

    // -- dfill ---------------------------------------------------------------

    #[test]
    fn dfill_unit_stride() {
        let mut dx = vec![0.0; 4];
        dfill(4, 3.14, &mut dx, 1);
        assert_eq!(dx, vec![3.14, 3.14, 3.14, 3.14]);
    }

    #[test]
    fn dfill_non_unit_stride() {
        let mut dx = vec![0.0; 6];
        dfill(3, 1.0, &mut dx, 2);
        assert_eq!(dx[0], 1.0);
        assert_eq!(dx[1], 0.0);
        assert_eq!(dx[2], 1.0);
        assert_eq!(dx[3], 0.0);
        assert_eq!(dx[4], 1.0);
        assert_eq!(dx[5], 0.0);
    }

    #[test]
    fn dfill_n_zero_noop() {
        let mut dx = vec![99.0; 2];
        dfill(0, 0.0, &mut dx, 1);
        assert_eq!(dx, vec![99.0, 99.0]);
    }

    #[test]
    fn dfill_negative_incx_noop() {
        let mut dx = vec![99.0; 2];
        dfill(2, 0.0, &mut dx, -1);
        assert_eq!(dx, vec![99.0, 99.0]);
    }

    // -- Additional daxpy tests (ported from Python) -------------------------

    // daxpy_n_negative_noop removed: n is now usize (cannot be negative)

    #[test]
    fn daxpy_partial_n() {
        // Only the first n elements are affected
        let dx = vec![1.0, 2.0, 3.0, 4.0];
        let mut dy = vec![10.0, 20.0, 30.0, 40.0];
        daxpy(2, 1.0, &dx, 1, &mut dy, 1);
        assert_eq!(dy, vec![11.0, 22.0, 30.0, 40.0]);
    }

    #[test]
    fn daxpy_negative_da() {
        let dx = vec![1.0, 2.0, 3.0];
        let mut dy = vec![10.0, 20.0, 30.0];
        daxpy(3, -1.0, &dx, 1, &mut dy, 1);
        assert_eq!(dy, vec![9.0, 18.0, 27.0]);
    }

    // -- Additional dcopy tests (ported from Python) -------------------------

    #[test]
    fn dcopy_partial_n() {
        let dx = vec![1.0, 2.0, 3.0, 4.0];
        let mut dy = vec![0.0; 4];
        dcopy(2, &dx, 1, &mut dy, 1);
        assert_eq!(dy, vec![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn dcopy_different_strides() {
        let dx = vec![1.0, 2.0, 3.0];
        let mut dy = vec![0.0; 5];
        dcopy(3, &dx, 1, &mut dy, 2);
        assert_eq!(dy, vec![1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn dcopy_negative_stride() {
        let dx = vec![1.0, 2.0, 3.0];
        let mut dy = vec![0.0; 3];
        dcopy(3, &dx, -1, &mut dy, -1);
        // Both start at index 2 and go backwards → preserves order
        assert_eq!(dy, vec![1.0, 2.0, 3.0]);
    }

    // -- Additional ddot tests (ported from Python) --------------------------

    // ddot_n_negative_returns_zero removed: n is now usize (cannot be negative)

    #[test]
    fn ddot_partial_n() {
        let dx = vec![1.0, 2.0, 3.0];
        let dy = vec![4.0, 5.0, 6.0];
        let result = ddot(2, &dx, 1, &dy, 1);
        assert!((result - 14.0).abs() < TOL); // 1*4 + 2*5 = 14
    }

    #[test]
    fn ddot_single_element() {
        let result = ddot(1, &[3.0], 1, &[7.0], 1);
        assert!((result - 21.0).abs() < TOL);
    }

    #[test]
    fn ddot_negative_stride() {
        let dx = vec![1.0, 2.0, 3.0];
        let dy = vec![4.0, 5.0, 6.0];
        // incx=-1 reverses dx → 3*4 + 2*5 + 1*6 = 28
        let result = ddot(3, &dx, -1, &dy, 1);
        assert!((result - 28.0).abs() < TOL);
    }

    // -- Additional dnrm2 tests (ported from Python) -------------------------

    // dnrm2_n_negative_returns_zero removed: n is now usize (cannot be negative)

    #[test]
    fn dnrm2_incx_zero_returns_zero() {
        assert_eq!(dnrm2(3, &[1.0, 2.0, 3.0], 0), 0.0);
    }

    #[test]
    fn dnrm2_unit_vector() {
        let x = vec![1.0, 0.0, 0.0];
        assert!((dnrm2(3, &x, 1) - 1.0).abs() < TOL);
    }

    // -- Additional dscal tests (ported from Python) -------------------------

    // dscal_n_negative_noop removed: n is now usize (cannot be negative)

    #[test]
    fn dscal_incx_zero_noop() {
        let mut dx = vec![1.0, 2.0];
        dscal(2, 5.0, &mut dx, 0);
        assert_eq!(dx, vec![1.0, 2.0]);
    }

    #[test]
    fn dscal_incx_negative_noop() {
        let mut dx = vec![1.0, 2.0];
        dscal(2, 5.0, &mut dx, -1);
        assert_eq!(dx, vec![1.0, 2.0]);
    }

    #[test]
    fn dscal_partial_n() {
        let mut dx = vec![1.0, 2.0, 3.0, 4.0];
        dscal(2, 10.0, &mut dx, 1);
        assert_eq!(dx, vec![10.0, 20.0, 3.0, 4.0]);
    }

    // -- Integration tests (ported from Python) ------------------------------

    #[test]
    fn integration_copy_and_axpy_pattern() {
        // Pattern: dcopy(n,xl,1,u,1); daxpy(n,-one,x,1,u,1) → u = xl - x
        let xl = vec![0.0, -1.0, 0.5];
        let x = vec![0.3, 0.2, 0.8];
        let mut u = vec![0.0; 3];
        dcopy(3, &xl, 1, &mut u, 1);
        daxpy(3, -1.0, &x, 1, &mut u, 1);
        assert!((u[0] - (-0.3)).abs() < TOL);
        assert!((u[1] - (-1.2)).abs() < TOL);
        assert!((u[2] - (-0.3)).abs() < TOL);
    }

    #[test]
    fn integration_fill_pattern() {
        // Pattern: s[0] = 0; dcopy(n, s, 0, s, 1) → s[:] = 0
        let mut s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        s[0] = ZERO;
        dcopy(5, &s.clone(), 0, &mut s, 1);
        assert_eq!(s, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn integration_scale_copy_axpy_pattern() {
        // Pattern from inexact line search: x = x0 + alpha * s
        let mut s = vec![1.0, 2.0, 3.0];
        let x0 = vec![10.0, 20.0, 30.0];
        let mut x = vec![0.0; 3];
        let alpha = 0.5;
        dscal(3, alpha, &mut s, 1);
        dcopy(3, &x0, 1, &mut x, 1);
        daxpy(3, 1.0, &s, 1, &mut x, 1);
        assert!((x[0] - 10.5).abs() < TOL);
        assert!((x[1] - 21.0).abs() < TOL);
        assert!((x[2] - 31.5).abs() < TOL);
    }

    #[test]
    fn integration_dot_and_norm_consistency() {
        // dnrm2(x) == sqrt(ddot(x, x))
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let norm_val = dnrm2(5, &x, 1);
        let dot_val = ddot(5, &x, 1, &x, 1);
        assert!((norm_val - dot_val.sqrt()).abs() < TOL);
    }

    // -- nrm2 convenience edge cases (ported from Python TestNrm2) -----------

    #[test]
    fn nrm2_empty_slice() {
        assert_eq!(nrm2(&[]), 0.0);
    }

    #[test]
    fn nrm2_single_negative() {
        assert!((nrm2(&[-3.0]) - 3.0).abs() < TOL);
    }

    // -- Constants -----------------------------------------------------------

    #[test]
    fn constants_have_expected_values() {
        assert_eq!(ZERO, 0.0);
        assert_eq!(ONE, 1.0);
        assert_eq!(TWO, 2.0);
        assert_eq!(FOUR, 4.0);
        assert_eq!(TEN, 10.0);
        assert_eq!(HUN, 100.0);
        assert_eq!(EPMACH, f64::EPSILON);
    }
}
