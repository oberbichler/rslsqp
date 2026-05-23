//! PyO3 bindings for the SLSQP solver.
//!
//! This crate exposes the Rust SLSQP numerical core to Python via PyO3/maturin.
//! The Python module `_core` provides:
//!
//! - **Core solver**: `slsqp` (single reverse-communication step).
//! - **High-level optimize**: runs the entire SLSQP loop in Rust.
//! - **Workspace**: `SlsqpWorkspace` for reuse across iterations.
//!
//! # Zero-copy strategy
//!
//! Multiple `readwrite()` guards on **distinct** `PyArray` objects can coexist
//! safely because each guard borrows its own `Bound<PyArray>` reference.  We
//! obtain `&mut [f64]` slices directly from the NumPy arrays and pass them
//! straight into the Rust core, eliminating intermediate `Vec` allocations.
//!
//! The only remaining copies are:
//! - **2-D arrays** (`PyArray2` ↔ [`ColMat`](core_types::ColMat)): the Rust core uses
//!   the custom `ColMat` struct (column-major), so data must be copied in and out.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[macro_use]
mod profiling;
mod bvls;
mod core_basic;
mod core_ls;
mod core_slsqp;
mod core_types;
mod lapack;
mod support;

use core_slsqp as cs;
use core_types::*;

// ===========================================================================
// 2-D array helpers — conversion between PyArray2 and ColMat
// ===========================================================================
//
// Two flavours:
//
//   • `read_2d()` — copies from a mutable `&Bound<PyArray2>` into `ColMat`.
//   • `read_2d_inner()` — copies from an `ndarray::ArrayView2` into `ColMat`.
//
// NumPy arrays are row-major (C-order); ColMat is column-major.  We always
// do an element-wise copy to transpose the layout.

/// Read a 2-D NumPy array (via mutable `&Bound`) into a [`ColMat`].
///
/// Always copies, since the solver may mutate the data and we write it
/// back via [`write_2d()`] afterwards.
fn read_2d(a: &Bound<'_, PyArray2<f64>>) -> ColMat {
    let binding = a.readonly();
    let arr = binding.as_array();
    read_2d_inner(&arr)
}

/// Copy an `ndarray::ArrayView2` into a `ColMat` (column-major).
fn read_2d_inner(arr: &ndarray::ArrayView2<'_, f64>) -> ColMat {
    let rows = arr.nrows();
    let cols = arr.ncols();
    if rows == 0 || cols == 0 {
        return ColMat::zeros(rows, cols);
    }

    // Element-wise copy from NumPy (row-major) to ColMat (column-major).
    let mut mat = ColMat::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            mat[(i, j)] = arr[(i, j)];
        }
    }
    mat
}

/// Write a [`ColMat`] back into a 2-D NumPy array.
///
/// Copies from column-major `ColMat` into row-major NumPy array,
/// element-wise.
fn write_2d(a: &Bound<'_, PyArray2<f64>>, mat: &ColMat) {
    let rows = mat.rows;
    if rows == 0 {
        return;
    }
    let cols = mat.cols;
    let mut arr = a.readwrite();
    let mut arr = arr.as_array_mut();
    for i in 0..rows {
        for j in 0..cols {
            arr[(i, j)] = mat[(i, j)];
        }
    }
}

// ===========================================================================
// SlsqpWorkspace Python wrapper + SLSQP top-level binding
// ===========================================================================

/// Python wrapper for [`SlsqpWorkspace`](core_slsqp::SlsqpWorkspace).
///
/// Owns all sub-arrays and the `LsWorkspace` so that `slsqp_step` can
/// reuse them across reverse-communication iterations without allocation.
#[pyclass(name = "SlsqpWorkspace")]
struct PySlsqpWorkspace {
    inner: cs::SlsqpWorkspace,
}

#[pymethods]
impl PySlsqpWorkspace {
    /// Create a workspace for a problem with `n` variables, `m` total
    /// constraints, and `meq` equality constraints.
    #[new]
    fn new(n: usize, m: usize, meq: usize) -> Self {
        PySlsqpWorkspace {
            inner: cs::SlsqpWorkspace::new(n, m, meq),
        }
    }

    /// Reset all workspace arrays to zero for reuse across optimisations.
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Python binding for [`slsqp_step`](core_slsqp::slsqp_step) — one step of SLSQP.
///
/// Uses a [`SlsqpWorkspace`] that owns all sub-arrays, eliminating the
/// flat-`w` copy-in/copy-out overhead and providing proper workspace reuse.
///
/// Mutable 1-D arrays (`x`, `c`, `g`) use zero-copy `readwrite()` guards.
/// The 2-D constraint Jacobian `a` requires copy-in/copy-out.
#[pyfunction]
#[pyo3(name = "slsqp")]
fn py_slsqp<'py>(
    _py: Python<'py>,
    m: usize,
    meq: usize,
    la: usize,
    n: usize,
    x: &Bound<'py, PyArray1<f64>>,
    xl: PyReadonlyArray1<'py, f64>,
    xu: PyReadonlyArray1<'py, f64>,
    f: f64,
    c: &Bound<'py, PyArray1<f64>>,
    g: &Bound<'py, PyArray1<f64>>,
    a: &Bound<'py, PyArray2<f64>>,
    acc: f64,
    iter_: usize,
    mode: i32,
    ws: &mut PySlsqpWorkspace,
    alphamin: f64,
    alphamax: f64,
    tolf: f64,
    toldf: f64,
    toldx: f64,
    max_iter_ls: usize,
    nnls_mode: NnlsMode,
    infinite_bound: f64,
) -> PyResult<(f64, usize, i32)> {
    let mut x_rw = x.readwrite();
    let mut c_rw = c.readwrite();
    let mut g_rw = g.readwrite();

    let x_s = x_rw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("x must be a contiguous array"))?;
    let xl_s = xl
        .as_slice()
        .map_err(|_| PyValueError::new_err("xl must be a contiguous array"))?;
    let xu_s = xu
        .as_slice()
        .map_err(|_| PyValueError::new_err("xu must be a contiguous array"))?;
    let c_s = c_rw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("c must be a contiguous array"))?;
    let g_s = g_rw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("g must be a contiguous array"))?;

    let mut a_mat = read_2d(a);

    let result = cs::slsqp_step(
        m,
        meq,
        la,
        n,
        x_s,
        xl_s,
        xu_s,
        f,
        c_s,
        g_s,
        &mut a_mat,
        acc,
        iter_,
        mode,
        &mut ws.inner,
        alphamin,
        alphamax,
        tolf,
        toldf,
        toldx,
        max_iter_ls,
        nnls_mode,
        infinite_bound,
    );

    // Drop readwrite guards before writing 2-D array back.
    drop(x_rw);
    drop(c_rw);
    drop(g_rw);
    write_2d(a, &a_mat);

    Ok((result.0, result.1, result.2))
}

// ===========================================================================
// High-level optimize — entire loop in Rust
// ===========================================================================

/// Result returned by [`py_optimize`].
#[pyclass(name = "RustOptimizeResult")]
struct PyOptResult {
    #[pyo3(get)]
    x: Py<PyArray1<f64>>,
    #[pyo3(get)]
    fun: f64,
    #[pyo3(get)]
    constraints: Py<PyArray1<f64>>,
    #[pyo3(get)]
    status: SlsqpStatus,
    #[pyo3(get)]
    message: String,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    success: bool,
    #[pyo3(get)]
    nfev: usize,
    #[pyo3(get)]
    njev: usize,
}

/// Run the entire SLSQP optimisation loop in Rust.
///
/// Only crosses the Python boundary for function/gradient evaluations.
/// This eliminates:
/// - Per-iteration copy-in/copy-out (2D Jacobian stays in Rust)
/// - Per-iteration workspace Vec allocations
/// - Per-iteration Python→Rust call overhead for the core solver step
///
/// # Arguments
/// * `func` — Python callable `(x_array) -> (f: float, c: array)`.
/// * `grad` — Python callable `(x_array) -> (g: array, a: array_2d)`, or None for FD.
/// * `x0`   — Initial guess.
/// * `xl`, `xu` — Bounds (use NaN for unbounded).
/// * `m`, `meq` — Total and equality constraint counts.
/// * `max_iter` — Maximum iterations.
/// * `acc` — Convergence tolerance.
/// * `gradient_mode` — [`GradientMode`] enum (USER, BACKWARD, FORWARD, CENTRAL).
/// * `gradient_delta` — Step for FD.
/// * `linesearch_mode` — [`LinesearchMode`] enum (INEXACT, EXACT).
/// * `alphamin`, `alphamax` — Step-length bounds.
/// * `tolf`, `toldf`, `toldx` — Convergence tolerances.
/// * `max_iter_ls` — Max NNLS iterations.
/// * `nnls_mode` — [`NnlsMode`] enum (NNLS, BVLS).
/// * `infinite_bound` — Threshold for infinite bounds.
#[pyfunction]
#[pyo3(name = "optimize", signature = (
    func, grad, x0, xl, xu,
    m, meq, max_iter, acc,
    gradient_mode, gradient_delta,
    linesearch_mode, alphamin, alphamax,
    tolf, toldf, toldx,
    max_iter_ls, nnls_mode, infinite_bound,
    workspace = None,
))]
fn py_optimize<'py>(
    py: Python<'py>,
    func: &Bound<'py, pyo3::types::PyAny>,
    grad: &Bound<'py, pyo3::types::PyAny>,
    x0: PyReadonlyArray1<'py, f64>,
    xl: PyReadonlyArray1<'py, f64>,
    xu: PyReadonlyArray1<'py, f64>,
    m: usize,
    meq: usize,
    max_iter: usize,
    acc: f64,
    gradient_mode: GradientMode,
    gradient_delta: f64,
    linesearch_mode: LinesearchMode,
    alphamin: f64,
    alphamax: f64,
    tolf: f64,
    toldf: f64,
    toldx: f64,
    max_iter_ls: usize,
    nnls_mode: NnlsMode,
    infinite_bound: f64,
    workspace: Option<&mut PySlsqpWorkspace>,
) -> PyResult<PyOptResult> {
    // Validate: USER mode requires a grad callback
    if gradient_mode == GradientMode::User && grad.is_none() {
        return Err(PyValueError::new_err(
            "grad callback must be provided when gradient_mode is USER",
        ));
    }

    let x0_slice = x0
        .as_slice()
        .map_err(|_| PyValueError::new_err("x0 must be a contiguous array"))?;
    let n = x0_slice.len();
    let la = m.max(1);

    // Copy inputs into owned Rust arrays
    let xl_vec = xl
        .as_slice()
        .map_err(|_| PyValueError::new_err("xl must be a contiguous array"))?
        .to_vec();
    let xu_vec = xu
        .as_slice()
        .map_err(|_| PyValueError::new_err("xu must be a contiguous array"))?
        .to_vec();
    let mut x = x0_slice.to_vec();

    // Allocate solver working arrays
    let mut g = vec![0.0; n + 1];
    let mut c = vec![0.0; la];
    let mut a = ColMat::zeros(la, n + 1);

    // Reuse or allocate workspace
    let mut ws_owned: Option<cs::SlsqpWorkspace>;
    let ws_ref: &mut cs::SlsqpWorkspace = match workspace {
        Some(py_ws) => {
            py_ws.inner.reset();
            &mut py_ws.inner
        }
        None => {
            ws_owned = Some(cs::SlsqpWorkspace::new(n, m, meq));
            ws_owned.as_mut().unwrap()
        }
    };

    let has_grad = !grad.is_none();

    // Line-search sign convention
    let acc_val = match linesearch_mode {
        LinesearchMode::Exact => -acc.abs(),
        LinesearchMode::Inexact => acc.abs(),
    };

    let mut mode = 0i32;
    let mut iter_ = max_iter;
    let mut f_val: f64 = 0.0;
    let mut cvec: Vec<f64> = vec![0.0; m];
    let mut acc_mut = acc_val;
    let mut nfev: usize = 0;
    let mut njev: usize = 0;

    // --- Iteration loop (entirely in Rust) ---
    loop {
        // --- Function evaluation (only when solver requests it) ---
        if mode == 0 || mode == 1 {
            // Call Python func(x) -> (f, c_vec)
            let x_arr = PyArray1::from_slice(py, &x);
            let result = func.call1((x_arr,))?;
            nfev += 1;
            let tup = result.cast::<pyo3::types::PyTuple>()?;
            f_val = tup.get_item(0)?.extract::<f64>()?;
            let c_item = tup.get_item(1)?;
            let c_py = c_item.cast::<PyArray1<f64>>()?;
            let c_ro = c_py.readonly();
            cvec = c_ro
                .as_slice()
                .map_err(|_| PyValueError::new_err("constraint array from func must be contiguous"))?
                .to_vec();
            if m > 0 && !cvec.is_empty() {
                let copy_len = m.min(cvec.len()).min(c.len());
                c[..copy_len].copy_from_slice(&cvec[..copy_len]);
            }
        }
        // When mode == -1: keep previous f_val and cvec (gradient-only request)

        // --- Gradient evaluation ---
        if mode == 0 || mode == -1 {
            njev += 1;
            if has_grad {
                // User-provided gradient: grad(x) -> (g_vec, a_mat)
                let x_arr = PyArray1::from_slice(py, &x);
                let result = grad.call1((x_arr,))?;
                let tup = result.cast::<pyo3::types::PyTuple>()?;

                // Extract gradient vector — bind temporaries to extend lifetimes
                let g_item = tup.get_item(0)?;
                let g_py = g_item.cast::<PyArray1<f64>>()?;
                let g_ro = g_py.readonly();
                let g_slice = g_ro
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("gradient array from grad must be contiguous"))?;
                g[..n].copy_from_slice(&g_slice[..n]);

                // Extract Jacobian matrix
                let a_item = tup.get_item(1)?;
                let a_py = a_item.cast::<PyArray2<f64>>()?;
                let a_ro = a_py.readonly();
                let a_arr = a_ro.as_array();
                let a_rows = a_arr.nrows();
                let a_cols = a_arr.ncols();
                for r in 0..a_rows.min(la) {
                    for cc in 0..a_cols.min(n) {
                        a[(r, cc)] = a_arr[[r, cc]];
                    }
                }
            } else {
                // Finite-difference gradient — helper closure to call func
                let eval = |x_buf: &[f64]| -> PyResult<(f64, Vec<f64>)> {
                    let x_arr = PyArray1::from_slice(py, x_buf);
                    let result = func.call1((x_arr,))?;
                    let tup = result.cast::<pyo3::types::PyTuple>()?;
                    let fv = tup.get_item(0)?.extract::<f64>()?;
                    let c_item = tup.get_item(1)?;
                    let c_py = c_item.cast::<PyArray1<f64>>()?;
                    let cv: Vec<f64> = c_py
                        .readonly()
                        .as_slice()
                        .map_err(|_| PyValueError::new_err("constraint array from func must be contiguous"))?
                        .to_vec();
                    Ok((fv, cv))
                };

                let delta = gradient_delta;
                let fact = match gradient_mode {
                    GradientMode::Central => 2.0,
                    _ => 1.0,
                };
                for ig in 0..n {
                    let (fr, cvecr, fl, cvecl) = match gradient_mode {
                        GradientMode::Backward => {
                            let (fr, cvecr) = eval(&x)?;
                            nfev += 1;
                            x[ig] -= delta;
                            let (fl, cvecl) = eval(&x)?;
                            nfev += 1;
                            x[ig] += delta;
                            (fr, cvecr, fl, cvecl)
                        }
                        GradientMode::Forward => {
                            x[ig] += delta;
                            let (fr, cvecr) = eval(&x)?;
                            nfev += 1;
                            x[ig] -= delta;
                            let (fl, cvecl) = eval(&x)?;
                            nfev += 1;
                            (fr, cvecr, fl, cvecl)
                        }
                        GradientMode::Central => {
                            x[ig] += delta;
                            let (fr, cvecr) = eval(&x)?;
                            nfev += 1;
                            x[ig] -= 2.0 * delta;
                            let (fl, cvecl) = eval(&x)?;
                            nfev += 1;
                            x[ig] += delta; // restore
                            (fr, cvecr, fl, cvecl)
                        }
                        GradientMode::User => unreachable!("has_grad is false"),
                    };

                    g[ig] = (fr - fl) / (fact * delta);
                    if m > 0 {
                        for j in 0..m.min(cvecr.len()).min(cvecl.len()) {
                            a[(j, ig)] = (cvecr[j] - cvecl[j]) / (fact * delta);
                        }
                    }
                }
            }
        }

        // --- Core solver step ---
        let result = cs::slsqp_step(
            m,
            meq,
            la,
            n,
            &mut x,
            &xl_vec,
            &xu_vec,
            f_val,
            &mut c,
            &mut g,
            &mut a,
            acc_mut,
            iter_,
            mode,
            ws_ref,
            alphamin,
            alphamax,
            tolf,
            toldf,
            toldx,
            max_iter_ls,
            nnls_mode,
            infinite_bound,
        );
        acc_mut = result.0;
        iter_ = result.1;
        mode = result.2;

        if mode == 1 || mode == -1 {
            continue; // next evaluation
        } else {
            break; // converged or error
        }
    }

    let status = SlsqpStatus::from(mode);

    Ok(PyOptResult {
        x: PyArray1::from_vec(py, x).unbind(),
        fun: f_val,
        constraints: PyArray1::from_vec(py, cvec).unbind(),
        status,
        message: status.to_string(),
        iterations: iter_,
        success: status == SlsqpStatus::Converged,
        nfev,
        njev,
    })
}

// ===========================================================================
// Finite-difference gradients — standalone function for the Python-loop path
// ===========================================================================

/// Compute finite-difference gradients entirely in Rust.
///
/// This eliminates the Python-level `for ig in range(n)` loop in
/// `SlsqpSolver._compute_gradients()` (see `slsqp_module.py`),
/// keeping all perturbation/differencing arithmetic in Rust and only
/// crossing the Python boundary for the actual function evaluations.
///
/// # Arguments
/// * `func` — Python callable `(x_array) -> (f: float, c: array)`.
/// * `x` — Current point, shape `(n,)`.
/// * `gradient_mode` — [`GradientMode`] enum (BACKWARD, FORWARD, CENTRAL).
/// * `gradient_delta` — FD step size.
/// * `m` — Number of constraints.
///
/// # Returns
/// `(g, a)` — objective gradient `(n,)` and constraint Jacobian `(m, n)`.
#[pyfunction]
#[pyo3(name = "compute_fd_gradients")]
fn py_compute_fd_gradients<'py>(
    py: Python<'py>,
    func: &Bound<'py, pyo3::types::PyAny>,
    x: PyReadonlyArray1<'py, f64>,
    gradient_mode: GradientMode,
    gradient_delta: f64,
    m: usize,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)> {
    let x_slice = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be a contiguous array"))?;
    let n = x_slice.len();
    let delta = gradient_delta;

    // Copy x so we can perturb it in-place
    let mut xw = x_slice.to_vec();

    let mut g = vec![0.0f64; n];
    // Store Jacobian in row-major order: a[j * n + i] = da_j / dx_i
    let mut a_flat = vec![0.0f64; m * n];

    let fact = match gradient_mode {
        GradientMode::Central => 2.0,
        _ => 1.0,
    };

    // Helper closure: call Python func(x) -> (f, c_vec)
    let eval = |xb: &[f64]| -> PyResult<(f64, Vec<f64>)> {
        let x_arr = PyArray1::from_slice(py, xb);
        let result = func.call1((x_arr,))?;
        let tup = result.cast::<pyo3::types::PyTuple>()?;
        let fv = tup.get_item(0)?.extract::<f64>()?;
        let c_item = tup.get_item(1)?;
        let c_py = c_item.cast::<PyArray1<f64>>()?;
        let cv: Vec<f64> = c_py
            .readonly()
            .as_slice()
            .map_err(|_| PyValueError::new_err("constraint array from func must be contiguous"))?
            .to_vec();
        Ok((fv, cv))
    };

    for ig in 0..n {
        let (fr, cvecr, fl, cvecl) = match gradient_mode {
            GradientMode::Backward => {
                let (fr, cvecr) = eval(&xw)?;
                xw[ig] -= delta;
                let (fl, cvecl) = eval(&xw)?;
                xw[ig] += delta; // restore
                (fr, cvecr, fl, cvecl)
            }
            GradientMode::Forward => {
                xw[ig] += delta;
                let (fr, cvecr) = eval(&xw)?;
                xw[ig] -= delta; // restore
                let (fl, cvecl) = eval(&xw)?;
                (fr, cvecr, fl, cvecl)
            }
            GradientMode::Central => {
                xw[ig] += delta;
                let (fr, cvecr) = eval(&xw)?;
                xw[ig] -= 2.0 * delta;
                let (fl, cvecl) = eval(&xw)?;
                xw[ig] += delta; // restore
                (fr, cvecr, fl, cvecl)
            }
            GradientMode::User => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "compute_fd_gradients cannot be used with GradientMode.USER",
                ));
            }
        };

        g[ig] = (fr - fl) / (fact * delta);
        if m > 0 {
            let len = m.min(cvecr.len()).min(cvecl.len());
            for j in 0..len {
                a_flat[j * n + ig] = (cvecr[j] - cvecl[j]) / (fact * delta);
            }
        }
    }

    // Build NumPy arrays
    let g_arr = PyArray1::from_vec(py, g).unbind();
    // Build 2-D Jacobian (m, n) row-major
    let a_arr = if m == 0 {
        // from_vec2 with empty input gives (0,0); we need (0, n)
        PyArray2::zeros(py, [0, n], false).unbind()
    } else {
        PyArray2::from_vec2(
            py,
            &(0..m)
                .map(|j| a_flat[j * n..(j + 1) * n].to_vec())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .unbind()
    };

    Ok((g_arr, a_arr))
}

// ===========================================================================
// Profiling bindings (only available with --features profiling)
// ===========================================================================

#[cfg(feature = "profiling")]
#[pyfunction]
#[pyo3(name = "get_profiling_stats")]
fn py_get_profiling_stats(py: Python<'_>) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    let stats = profiling::inner::get_stats();
    let dict = PyDict::new(py);
    for (name, s) in &stats {
        let inner = PyDict::new(py);
        inner.set_item("calls", s.calls)?;
        inner.set_item("total_us", s.total_ns as f64 / 1000.0)?;
        dict.set_item(*name, inner)?;
    }
    Ok(dict.into())
}

#[cfg(feature = "profiling")]
#[pyfunction]
#[pyo3(name = "reset_profiling_stats")]
fn py_reset_profiling_stats() {
    profiling::inner::reset_stats();
}

// ===========================================================================
// Exported status-message helper
// ===========================================================================

// ===========================================================================
// Python module definition
// ===========================================================================

/// The `_core` native module — Rust implementation of the SLSQP solver.
///
/// This module is not meant to be imported directly.  Use the `rslsqp`
/// Python package instead, which provides a thin wrapper layer and the
/// high-level `SlsqpSolver` class.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- Enum types (single source of truth) ---
    m.add_class::<GradientMode>()?;
    m.add_class::<LinesearchMode>()?;
    m.add_class::<NnlsMode>()?;
    m.add_class::<SlsqpStatus>()?;

    // --- Core solver (reverse-communication step) ---
    m.add_class::<PySlsqpWorkspace>()?;
    m.add_function(wrap_pyfunction!(py_slsqp, m)?)?;

    // --- High-level optimize (entire loop in Rust) ---
    m.add_class::<PyOptResult>()?;
    m.add_function(wrap_pyfunction!(py_optimize, m)?)?;

    // --- Finite-difference gradient helper ---
    m.add_function(wrap_pyfunction!(py_compute_fd_gradients, m)?)?;

    // --- Profiling (only available with --features profiling) ---
    #[cfg(feature = "profiling")]
    {
        m.add_function(wrap_pyfunction!(py_get_profiling_stats, m)?)?;
        m.add_function(wrap_pyfunction!(py_reset_profiling_stats, m)?)?;
    }

    Ok(())
}
