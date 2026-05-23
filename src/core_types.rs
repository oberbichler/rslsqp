//! Core data types shared across the SLSQP solver modules.
//!
//! This module defines the persistent state structures used by the iterative
//! algorithms ([`LinminData`] for Brent's line-search and [`SlsqpbData`] for the
//! main SQP loop) and the column-major matrix type [`ColMat`] used throughout
//! the Rust core.

use pyo3::prelude::*;
use std::ops::{Index, IndexMut};

// ---------------------------------------------------------------------------
// Solver enumerations — single source of truth, exported to Python via PyO3
// ---------------------------------------------------------------------------

/// How gradients are supplied or approximated.
///
/// Exported to Python as `rslsqp._core.GradientMode`.  Supports `int()`
/// conversion and `==` comparison with integers via `eq_int`.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GradientMode {
    /// User-supplied gradient callback.
    #[pyo3(name = "USER")]
    User = 0,
    /// Backward finite differences.
    #[pyo3(name = "BACKWARD")]
    Backward = 1,
    /// Forward finite differences.
    #[pyo3(name = "FORWARD")]
    Forward = 2,
    /// Central finite differences.
    #[pyo3(name = "CENTRAL")]
    Central = 3,
}

/// Line-search strategy.
///
/// Exported to Python as `rslsqp._core.LinesearchMode`.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum LinesearchMode {
    /// Inexact (Armijo-type) line-search.
    #[pyo3(name = "INEXACT")]
    Inexact = 1,
    /// Exact (golden-section / parabolic) line-search.
    #[pyo3(name = "EXACT")]
    Exact = 2,
}

/// Which non-negative least-squares method to use.
///
/// Exported to Python as `rslsqp._core.NnlsMode`.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum NnlsMode {
    /// Original NNLS algorithm.
    #[pyo3(name = "NNLS")]
    Nnls = 1,
    /// Newer BVLS algorithm.
    #[pyo3(name = "BVLS")]
    Bvls = 2,
}

// ---------------------------------------------------------------------------
// SLSQP status codes — typed representation of the solver's `mode` flag
// ---------------------------------------------------------------------------

/// Solver status returned by the SLSQP reverse-communication protocol.
///
/// Exported to Python as `rslsqp._core.SlsqpStatus`.  Supports `int()`
/// conversion and `==` comparison with integers via `eq_int`.
///
/// The inner solver loop uses a raw `i32` mode flag (Fortran convention).
/// This enum provides a typed, `Display`-capable representation at output
/// boundaries.  Convert with `SlsqpStatus::from(mode)`.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SlsqpStatus {
    /// Mode 0 — solver converged within requested accuracy.
    #[pyo3(name = "CONVERGED")]
    Converged = 0,
    /// Mode 1 — solver requests a function evaluation.
    #[pyo3(name = "FUNC_EVAL_REQUIRED")]
    FuncEvalRequired = 1,
    /// Mode -1 — solver requests a gradient evaluation.
    #[pyo3(name = "GRAD_EVAL_REQUIRED")]
    GradEvalRequired = -1,
    /// Mode -2 — user called `abort()`.
    #[pyo3(name = "USER_STOP")]
    UserStop = -2,
    /// Mode 2 — more equality constraints than variables.
    #[pyo3(name = "TOO_MANY_EQUALITY_CONSTRAINTS")]
    TooManyEqualityConstraints = 2,
    /// Mode 3 — LSQ sub-problem exceeded 3·n iterations.
    #[pyo3(name = "LSQ_ITERATIONS_EXCEEDED")]
    LsqIterationsExceeded = 3,
    /// Mode 4 — inequality constraints are incompatible.
    #[pyo3(name = "INCOMPATIBLE_INEQUALITY")]
    IncompatibleInequality = 4,
    /// Mode 5 — singular matrix E in LSQ sub-problem.
    #[pyo3(name = "SINGULAR_MATRIX_E")]
    SingularMatrixE = 5,
    /// Mode 6 — singular matrix C in LSQ sub-problem.
    #[pyo3(name = "SINGULAR_MATRIX_C")]
    SingularMatrixC = 6,
    /// Mode 7 — rank-deficient equality constraint sub-problem (HFTI).
    #[pyo3(name = "RANK_DEFICIENT_HFTI")]
    RankDeficientHfti = 7,
    /// Mode 8 — positive directional derivative for line-search.
    #[pyo3(name = "POSITIVE_DIRECTIONAL_DERIVATIVE")]
    PositiveDirectionalDerivative = 8,
    /// Mode 9 — exceeded `max_iter` iterations.
    #[pyo3(name = "MAX_ITERATIONS_REACHED")]
    MaxIterationsReached = 9,
    /// Mode -100 — `x` has wrong length.
    #[pyo3(name = "INVALID_X_SIZE")]
    InvalidXSize = -100,
    /// Mode -101 — invalid line-search mode value.
    #[pyo3(name = "INVALID_LINESEARCH_MODE")]
    InvalidLinesearchMode = -101,
    /// Mode -102 — objective function not provided.
    #[pyo3(name = "FUNCTION_NOT_ASSOCIATED")]
    FunctionNotAssociated = -102,
    /// Mode -103 — gradient function not provided.
    #[pyo3(name = "GRADIENT_NOT_ASSOCIATED")]
    GradientNotAssociated = -103,
    /// Mode -104 — invalid gradient mode value.
    #[pyo3(name = "INVALID_GRADIENT_MODE")]
    InvalidGradientMode = -104,
    /// Mode -105 — invalid FD perturbation step.
    #[pyo3(name = "INVALID_PERTURBATION_STEP")]
    InvalidPerturbationStep = -105,
    /// Any unrecognised mode value.
    #[pyo3(name = "UNKNOWN")]
    Unknown = i32::MIN,
}

impl From<i32> for SlsqpStatus {
    fn from(mode: i32) -> Self {
        match mode {
            0 => Self::Converged,
            1 => Self::FuncEvalRequired,
            -1 => Self::GradEvalRequired,
            -2 => Self::UserStop,
            2 => Self::TooManyEqualityConstraints,
            3 => Self::LsqIterationsExceeded,
            4 => Self::IncompatibleInequality,
            5 => Self::SingularMatrixE,
            6 => Self::SingularMatrixC,
            7 => Self::RankDeficientHfti,
            8 => Self::PositiveDirectionalDerivative,
            9 => Self::MaxIterationsReached,
            -100 => Self::InvalidXSize,
            -101 => Self::InvalidLinesearchMode,
            -102 => Self::FunctionNotAssociated,
            -103 => Self::GradientNotAssociated,
            -104 => Self::InvalidGradientMode,
            -105 => Self::InvalidPerturbationStep,
            _ => Self::Unknown,
        }
    }
}

impl std::fmt::Display for SlsqpStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Converged => f.write_str("Required accuracy for solution obtained"),
            Self::FuncEvalRequired | Self::GradEvalRequired => f.write_str("In progress"),
            Self::UserStop => f.write_str("User-triggered stop of slsqp"),
            Self::TooManyEqualityConstraints => {
                f.write_str("Number of equality constraints larger than n")
            }
            Self::LsqIterationsExceeded => {
                f.write_str("More than 3*n iterations in LSQ subproblem")
            }
            Self::IncompatibleInequality => f.write_str("Inequality constraints incompatible"),
            Self::SingularMatrixE => f.write_str("Singular matrix E in LSQ subproblem"),
            Self::SingularMatrixC => f.write_str("Singular matrix C in LSQ subproblem"),
            Self::RankDeficientHfti => {
                f.write_str("Rank-deficient equality constraint subproblem HFTI")
            }
            Self::PositiveDirectionalDerivative => {
                f.write_str("Positive directional derivative for linesearch")
            }
            Self::MaxIterationsReached => f.write_str("More than max_iter iterations in SLSQP"),
            Self::InvalidXSize => f.write_str("Invalid size(x) in slsqp_wrapper"),
            Self::InvalidLinesearchMode => f.write_str("Invalid linesearch_mode in slsqp_wrapper"),
            Self::FunctionNotAssociated => f.write_str("Function is not associated"),
            Self::GradientNotAssociated => f.write_str("Gradient function is not associated"),
            Self::InvalidGradientMode => f.write_str("Invalid gradient mode"),
            Self::InvalidPerturbationStep => {
                f.write_str("Invalid perturbation step size for finite difference gradients")
            }
            Self::Unknown => f.write_str("Unknown slsqp error"),
        }
    }
}

#[pymethods]
impl SlsqpStatus {
    /// Return the human-readable status message.
    fn __str__(&self) -> String {
        self.to_string()
    }

    /// Return the human-readable status message.
    fn __repr__(&self) -> String {
        format!("SlsqpStatus.{:?}", self)
    }

    /// Return the integer mode value.
    fn __int__(&self) -> i32 {
        *self as i32
    }
}

// ---------------------------------------------------------------------------
// LinminData — persistent state for Brent's line-search
// ---------------------------------------------------------------------------

/// Persistent data for the [`linmin`](crate::core_slsqp::linmin) golden-section /
/// parabolic line-search (Brent's method).
///
/// All fields are carried between reverse-communication calls so that the
/// line-search can be resumed after each function evaluation.
///
/// # Field semantics (Brent's algorithm)
///
/// | Field | Meaning |
/// |-------|---------|
/// | `a`, `b` | Current bracket endpoints |
/// | `x` | Current best point (lowest `fx`) |
/// | `w` | Second-best point (lowest `fw`) |
/// | `v` | Previous value of `w` |
/// | `u` | Most recently evaluated point |
/// | `fx`, `fw`, `fv`, `fu` | Function values at `x`, `w`, `v`, `u` |
/// | `d`, `e` | Step size and previous step (for parabolic test) |
/// | `p`, `q`, `r` | Parabolic interpolation temporaries |
/// | `m` | Bracket midpoint `0.5 * (a + b)` |
/// | `tol1`, `tol2` | Convergence tolerances derived from `|x|` |
#[derive(Clone, Debug)]
pub struct LinminData {
    /// Left bracket endpoint.
    pub a: f64,
    /// Right bracket endpoint.
    pub b: f64,
    /// Current step size.
    pub d: f64,
    /// Previous step size (used for parabolic fit test).
    pub e: f64,
    /// Parabolic interpolation numerator.
    pub p: f64,
    /// Parabolic interpolation denominator.
    pub q: f64,
    /// Parabolic interpolation temporary.
    pub r: f64,
    /// Most recently evaluated point.
    pub u: f64,
    /// Previous second-best point.
    pub v: f64,
    /// Second-best point (second lowest function value).
    pub w: f64,
    /// Current best point (lowest function value).
    pub x: f64,
    /// Bracket midpoint `0.5 * (a + b)`.
    pub m: f64,
    /// Function value at `u`.
    pub fu: f64,
    /// Function value at `v`.
    pub fv: f64,
    /// Function value at `w`.
    pub fw: f64,
    /// Function value at `x`.
    pub fx: f64,
    /// Primary convergence tolerance `sqrt(eps) * |x| + tol`.
    pub tol1: f64,
    /// Secondary convergence tolerance `2 * tol1`.
    pub tol2: f64,
}

impl Default for LinminData {
    fn default() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            d: 0.0,
            e: 0.0,
            p: 0.0,
            q: 0.0,
            r: 0.0,
            u: 0.0,
            v: 0.0,
            w: 0.0,
            x: 0.0,
            m: 0.0,
            fu: 0.0,
            fv: 0.0,
            fw: 0.0,
            fx: 0.0,
            tol1: 0.0,
            tol2: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// SlsqpbData — persistent state for the main SQP iteration
// ---------------------------------------------------------------------------

/// Persistent data for the [`slsqpb`](crate::core_slsqp::slsqpb) main SQP loop.
///
/// This struct carries the iteration state between reverse-communication calls.
/// The caller invokes `slsqpb` repeatedly, and the algorithm resumes from where
/// it left off using these saved values.
///
/// # Field semantics
///
/// | Field | Meaning |
/// |-------|---------|
/// | `f0` | Objective value at previous iterate `x0` |
/// | `t`, `t0` | Augmented Lagrangian merit function values |
/// | `h1`–`h4` | Temporary scalars for convergence / line-search logic |
/// | `gs` | Directional derivative `g^T s` |
/// | `tol` | Working convergence tolerance (`10 * acc`) |
/// | `alpha` | Current line-search step length |
/// | `line` | Line-search mode / iteration counter |
/// | `iexact` | 0 = inexact (Armijo), 1 = exact (Brent) line-search |
/// | `incons` | Counter for inconsistent linearization retries |
/// | `ireset` | Counter for BFGS resets |
/// | `itermx` | Maximum allowed SQP iterations |
/// | `n1` | `n + 1` (number of variables plus one) |
/// | `n2` | `n1 * n / 2` (packed triangle size) |
/// | `n3` | `n2 + 1` (packed triangle size plus one) |
#[derive(Clone, Debug)]
pub struct SlsqpbData {
    /// Augmented Lagrangian merit function value.
    pub t: f64,
    /// Objective value at the previous iterate `x0`.
    pub f0: f64,
    /// Temporary scalar (directional derivative magnitude / convergence test).
    pub h1: f64,
    /// Temporary scalar (constraint violation sum).
    pub h2: f64,
    /// Temporary scalar (predicted merit reduction / line-search tolerance).
    pub h3: f64,
    /// Temporary scalar (augmented problem scaling factor `1 - s[n]`).
    pub h4: f64,
    /// Merit function value at previous iterate.
    pub t0: f64,
    /// Directional derivative `g^T s`.
    pub gs: f64,
    /// Working convergence tolerance (`10 * acc`).
    pub tol: f64,
    /// Current line-search step length.
    pub alpha: f64,
    /// Line-search mode / iteration counter.
    pub line: i32,
    /// Line-search type: 0 = inexact (Armijo), 1 = exact (Brent).
    pub iexact: i32,
    /// Counter for inconsistent linearization retries.
    pub incons: i32,
    /// Counter for BFGS resets (resets to identity if > 5).
    pub ireset: i32,
    /// Maximum allowed SQP iterations.
    pub itermx: usize,
    /// `n + 1` — number of variables plus one.
    pub n1: usize,
    /// `n1 * n / 2` — packed upper-triangle size.
    pub n2: usize,
    /// `n2 + 1` — packed upper-triangle size plus one.
    pub n3: usize,
}

impl Default for SlsqpbData {
    fn default() -> Self {
        Self {
            t: 0.0,
            f0: 0.0,
            h1: 0.0,
            h2: 0.0,
            h3: 0.0,
            h4: 0.0,
            t0: 0.0,
            gs: 0.0,
            tol: 0.0,
            alpha: 0.0,
            line: 0,
            iexact: 0,
            incons: 0,
            ireset: 0,
            itermx: 0,
            n1: 0,
            n2: 0,
            n3: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Mat — flat row-major 2-D workspace buffer (crate-internal)
// ---------------------------------------------------------------------------

/// A 2-D matrix stored as a flat `Vec<f64>` in **row-major** order.
///
/// **Used only** as a crate-internal workspace buffer inside [`LseiWs`] for
/// the Householder triangularisation of C in `lsei_ws`.  The `h12_construct`
/// and `h12_apply` functions operate on contiguous rows (stride-1 within a
/// row), so column-major storage would require an extra transpose.
///
/// All public APIs use [`ColMat`] / [`ColMatView`] instead.
///
/// Element `(i, j)` lives at `data[i * stride + j]`.  For a freshly created
/// matrix `stride == cols`.
#[derive(Clone, Debug)]
pub(crate) struct Mat {
    /// Flat row-major storage: `data[i * stride + j]` is element `(i, j)`.
    pub(crate) data: Vec<f64>,
    /// Number of rows.
    pub(crate) rows: usize,
    /// Number of columns.
    pub(crate) cols: usize,
    /// Number of `f64`s between the start of consecutive rows in `data`.
    /// Equal to `cols` for dense matrices.
    pub(crate) stride: usize,
}

impl Mat {
    /// Create a zero-filled `rows × cols` matrix with `stride == cols`.
    #[inline]
    pub(crate) fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            data: vec![0.0; rows * cols],
            rows,
            cols,
            stride: cols,
        }
    }

    /// Resize the matrix to `rows × cols`, zeroing all elements.
    ///
    /// If the existing `data` buffer has enough capacity, no heap allocation
    /// occurs — only a `fill(0.0)` over the used portion.  Otherwise the
    /// buffer is grown (but never shrunk).
    #[inline]
    pub(crate) fn resize_zero(&mut self, rows: usize, cols: usize) {
        let needed = rows * cols;
        if self.data.len() < needed {
            self.data.resize(needed, 0.0);
        }
        // Zero only the portion we'll use
        for v in self.data[..needed].iter_mut() {
            *v = 0.0;
        }
        self.rows = rows;
        self.cols = cols;
        self.stride = cols;
    }
}

/// Immutable indexing by `(row, col)` tuple.
impl Index<(usize, usize)> for Mat {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.stride + j]
    }
}

/// Mutable indexing by `(row, col)` tuple.
impl IndexMut<(usize, usize)> for Mat {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.stride + j]
    }
}

// ---------------------------------------------------------------------------
// ColMat — flat column-major 2-D matrix
// ---------------------------------------------------------------------------

/// A 2-D matrix stored as a flat `Vec<f64>` in **column-major** order.
///
/// Element `(i, j)` lives at `data[i + j * rows]`.  Column `j` occupies
/// `data[j*rows .. (j+1)*rows]` — **contiguous in memory** — which makes
/// column-oriented algorithms (Householder QR, Givens rotations on columns)
/// cache-friendly.
///
/// This type is used internally by NNLS/LDP where all heavy computation
/// operates on columns.
///
/// # Constructors
///
/// - [`ColMat::zeros(rows, cols)`](ColMat::zeros) — zero-filled.
#[derive(Clone, Debug)]
pub struct ColMat {
    /// Flat column-major storage: `data[i + j * stride]` is element `(i, j)`.
    pub data: Vec<f64>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Number of elements between successive columns (≥ rows, allows padding).
    pub stride: usize,
}

impl ColMat {
    /// Create a zero-filled `rows × cols` column-major matrix.
    #[inline]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        ColMat {
            data: vec![0.0; rows * cols],
            rows,
            cols,
            stride: rows,
        }
    }

    /// Create a `ColMat` from existing data with an explicit stride.
    ///
    /// `stride` is the number of elements between successive columns (≥ rows).
    #[allow(dead_code)]
    pub fn with_stride(data: Vec<f64>, rows: usize, cols: usize, stride: usize) -> Self {
        debug_assert!(stride >= rows);
        debug_assert!(
            data.len() >= stride * cols.max(1).saturating_sub(1) + rows,
            "data too short for {}×{} with stride {}",
            rows,
            cols,
            stride
        );
        ColMat {
            data,
            rows,
            cols,
            stride,
        }
    }

    /// Wrap existing column-major flat data (no padding) into a `ColMat`.
    #[allow(dead_code)]
    pub fn from_col_major(data: &[f64], rows: usize, cols: usize) -> Self {
        debug_assert!(data.len() >= rows * cols);
        ColMat {
            data: data[..rows * cols].to_vec(),
            rows,
            cols,
            stride: rows,
        }
    }

    /// Resize the matrix to `rows × cols`, zeroing all elements.
    ///
    /// Reuses the existing buffer if large enough; never shrinks.
    #[inline]
    pub fn resize_zero(&mut self, rows: usize, cols: usize) {
        let needed = rows * cols;
        if self.data.len() < needed {
            self.data.resize(needed, 0.0);
        }
        for v in self.data[..needed].iter_mut() {
            *v = 0.0;
        }
        self.rows = rows;
        self.cols = cols;
        self.stride = rows;
    }

    /// Resize without zeroing — caller **must** fully overwrite before reading.
    ///
    /// This saves the O(rows×cols) memset when the caller will immediately
    /// populate all elements (e.g. a transpose copy).
    #[inline]
    pub fn resize_uninit(&mut self, rows: usize, cols: usize) {
        let needed = rows * cols;
        if self.data.len() < needed {
            self.data.resize(needed, 0.0);
        }
        self.rows = rows;
        self.cols = cols;
        self.stride = rows;
    }

    /// Return a mutable slice of column `j` (contiguous in memory).
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        let start = j * self.stride;
        &mut self.data[start..start + self.rows]
    }

    /// Return a shared slice of column `j` (contiguous in memory).
    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        let start = j * self.stride;
        &self.data[start..start + self.rows]
    }

    /// Extract row `i` into a new `Vec` (elements are stride apart in column-major).
    pub fn row(&self, i: usize) -> Vec<f64> {
        debug_assert!(i < self.rows);
        (0..self.cols)
            .map(|j| self.data[i + j * self.stride])
            .collect()
    }

    /// Write values from `src` into row `i`.
    #[allow(dead_code)]
    pub fn set_row(&mut self, i: usize, src: &[f64]) {
        debug_assert!(i < self.rows);
        debug_assert!(src.len() >= self.cols);
        for j in 0..self.cols {
            self.data[i + j * self.stride] = src[j];
        }
    }

    /// Create a new `ColMat` containing the top-left `(rows × cols)` sub-matrix.
    #[allow(dead_code)]
    pub fn sub(&self, rows: usize, cols: usize) -> ColMat {
        debug_assert!(rows <= self.rows);
        debug_assert!(cols <= self.cols);
        let mut result = ColMat::zeros(rows, cols);
        for j in 0..cols {
            for i in 0..rows {
                result.data[i + j * rows] = self.data[i + j * self.stride];
            }
        }
        result
    }

    /// Return a read-only [`ColMatView`] over the top-left `(rows × cols)` sub-matrix.
    ///
    /// The view borrows the underlying data without copying.
    pub fn view(&self, rows: usize, cols: usize) -> ColMatView<'_> {
        debug_assert!(rows <= self.rows);
        debug_assert!(cols <= self.cols);
        ColMatView {
            data: &self.data,
            rows,
            cols,
            stride: self.stride,
        }
    }

    /// Build a `ColMat` from a slice of `Vec<f64>` rows (row-major input,
    /// stored column-major).
    ///
    /// Each inner `Vec` represents one row. All rows must have the same length.
    /// Returns a zero-sized matrix if `vv` is empty.
    #[allow(dead_code)]
    pub fn from_vv(vv: &[Vec<f64>]) -> ColMat {
        let rows = vv.len();
        if rows == 0 {
            return ColMat::zeros(0, 0);
        }
        let cols = vv[0].len();
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i + j * rows] = vv[i][j];
            }
        }
        ColMat {
            data,
            rows,
            cols,
            stride: rows,
        }
    }
}

/// Immutable indexing by `(row, col)` tuple.
impl Index<(usize, usize)> for ColMat {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i + j * self.stride]
    }
}

/// Mutable indexing by `(row, col)` tuple.
impl IndexMut<(usize, usize)> for ColMat {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i + j * self.stride]
    }
}

// ---------------------------------------------------------------------------
// ColMatView — read-only view into a ColMat (no allocation)
// ---------------------------------------------------------------------------

/// A read-only view into a column-major matrix.
///
/// Borrows the underlying data slice; column `j` is contiguous at
/// `data[j*stride .. j*stride + rows]`.
#[derive(Debug)]
pub struct ColMatView<'a> {
    pub data: &'a [f64],
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
}

impl<'a> ColMatView<'a> {
    /// Return a shared slice of column `j` (contiguous in memory).
    #[allow(dead_code)]
    pub fn col(&self, j: usize) -> &'a [f64] {
        debug_assert!(j < self.cols);
        &self.data[j * self.stride..j * self.stride + self.rows]
    }

    /// Extract row `i` into a new `Vec` (elements are stride apart).
    #[allow(dead_code)]
    pub fn row(&self, i: usize) -> Vec<f64> {
        debug_assert!(i < self.rows);
        (0..self.cols)
            .map(|j| self.data[i + j * self.stride])
            .collect()
    }
}

/// Immutable indexing by `(row, col)` tuple.
impl<'a> Index<(usize, usize)> for ColMatView<'a> {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        debug_assert!(i < self.rows && j < self.cols);
        &self.data[i + j * self.stride]
    }
}

// ---------------------------------------------------------------------------
// Helper: ensure a Vec is at least `len` elements, zeroed
// ---------------------------------------------------------------------------

/// Ensure `v` has at least `len` elements, zeroing the used portion.
///
/// Never shrinks the allocation — only grows when necessary.
/// After this call, `v[0..len]` is all zeros.
#[inline]
pub fn ensure_vec_len(v: &mut Vec<f64>, len: usize) {
    if v.len() < len {
        v.resize(len, 0.0);
    }
    for x in v[..len].iter_mut() {
        *x = 0.0;
    }
}

/// Ensure `v` has at least `len` elements (usize), zeroed.
#[inline]
#[allow(dead_code)]
pub fn ensure_vec_len_usize(v: &mut Vec<usize>, len: usize) {
    if v.len() < len {
        v.resize(len, 0);
    }
    for x in v[..len].iter_mut() {
        *x = 0;
    }
}

// ---------------------------------------------------------------------------
// LsWorkspace — pre-allocated temporaries for the LS chain
// ---------------------------------------------------------------------------
//
// The workspace is split into per-level sub-structs so that each function
// in the chain (`lsq` → `lsei` → `lsi` → `ldp` → `nnls`) can borrow its
// own temporaries independently from the sub-workspaces it passes down.
// This satisfies Rust's disjoint-field borrow rules.
//
// Chain topology (each level borrows its own `*Ws` and passes `inner`):
//
//   LsWorkspace { lsq: LsqWs, inner: LseiChainWs }
//     LseiChainWs { lsei: LseiWs, inner: LsiChainWs }
//       LsiChainWs { lsi: LsiWs, inner: LdpChainWs }
//         LdpChainWs { ldp: LdpWs, nnls: NnlsWs }

/// NNLS-level pre-allocated temporaries.
pub struct NnlsWs {
    /// NNLS solution vector (length n).
    pub x: Vec<f64>,
    /// NNLS dual vector (length n).
    pub w: Vec<f64>,
    /// NNLS trial vector (length m).
    pub zz: Vec<f64>,
    /// NNLS active-set index (length n).
    pub index: Vec<usize>,
}

impl NnlsWs {
    pub fn new() -> Self {
        NnlsWs {
            x: Vec::new(),
            w: Vec::new(),
            zz: Vec::new(),
            index: Vec::new(),
        }
    }

    /// Prepare NNLS buffers for an m × n problem.
    #[allow(dead_code)]
    pub fn prepare(&mut self, m: usize, n: usize) {
        ensure_vec_len(&mut self.x, n);
        ensure_vec_len(&mut self.w, n);
        ensure_vec_len(&mut self.zz, m);
        ensure_vec_len_usize(&mut self.index, n);
        for i in 0..n {
            self.index[i] = i;
        }
    }
}

/// LDP-level pre-allocated temporaries.
pub struct LdpWs {
    /// LDP dual NNLS matrix E = [G^T; h^T] ((n+1) × m), column-major.
    ///
    /// Stored as `ColMat` to match NNLS's native column-major storage,
    /// eliminating a row↔column-major conversion at the LDP→NNLS boundary.
    pub e_cm: ColMat,
    /// LDP dual NNLS RHS f = [0..0, 1] (length n+1).
    pub f_vec: Vec<f64>,
    /// LDP output vector (length n).
    pub x_out: Vec<f64>,
    /// LDP output dual vector (length m).
    pub w_out: Vec<f64>,
}

impl LdpWs {
    pub fn new() -> Self {
        LdpWs {
            e_cm: ColMat::zeros(0, 0),
            f_vec: Vec::new(),
            x_out: Vec::new(),
            w_out: Vec::new(),
        }
    }

    /// Prepare LDP buffers for an m-constraint, n-variable problem.
    pub fn prepare(&mut self, m: usize, n: usize) {
        let n1 = n + 1;
        self.e_cm.resize_zero(n1, m);
        ensure_vec_len(&mut self.f_vec, n1);
        ensure_vec_len(&mut self.x_out, n);
        ensure_vec_len(&mut self.w_out, m);
    }
}

/// LDP + NNLS chain workspace (passed to `ldp_ws`).
pub struct LdpChainWs {
    pub ldp: LdpWs,
    pub nnls: NnlsWs,
}

impl LdpChainWs {
    pub fn new() -> Self {
        LdpChainWs {
            ldp: LdpWs::new(),
            nnls: NnlsWs::new(),
        }
    }
}

/// LSI-level pre-allocated temporaries.
pub struct LsiWs {
    /// LSI column-major copy of E (me × n).
    pub e_cm: ColMat,
    /// LSI column-major copy of G (mg × n).
    ///
    /// Used on all platforms — column-major storage enables direct LAPACK
    /// dispatch and avoids the row↔column-major round-trip that previously
    /// occurred via the removed `g_mat` (row-major) intermediate.
    pub g_cm: ColMat,
    /// LSI working copy of f (length me).
    pub f_arr: Vec<f64>,
    /// LSI working copy of h (length mg).
    pub h_arr: Vec<f64>,
    /// LSI output vector (length n).
    pub x_out: Vec<f64>,
    /// LSI diagonal reciprocals (length n).
    pub inv_diag: Vec<f64>,
    /// LAPACK Householder scalars for dgeqrf (length min(me,n)).
    #[cfg(feature = "blas")]
    pub tau: Vec<f64>,
    /// LAPACK workspace for dgeqrf/dormqr (auto-sized).
    #[cfg(feature = "blas")]
    pub lapack_work: Vec<f64>,
    /// When true, `e_cm` was pre-populated by the caller (e.g. `lsq_ws`)
    /// and `lsi_ws` should skip the row→col E copy.
    pub e_cm_ready: bool,
    /// When true, `g_cm` was pre-populated by the caller (e.g. `lsq_ws`)
    /// and `lsi_ws` should skip the row→col G copy.
    pub g_cm_ready: bool,
    /// Cached LAPACK optimal lwork value (avoids workspace queries on repeat calls).
    #[cfg(feature = "blas")]
    pub lapack_lwork: i32,
    /// Cached me dimension for workspace invalidation.
    #[cfg(feature = "blas")]
    pub lapack_cached_me: usize,
    /// Cached n dimension for workspace invalidation.
    #[cfg(feature = "blas")]
    pub lapack_cached_n: usize,
}

impl LsiWs {
    pub fn new() -> Self {
        LsiWs {
            e_cm: ColMat::zeros(0, 0),
            g_cm: ColMat::zeros(0, 0),
            f_arr: Vec::new(),
            h_arr: Vec::new(),
            x_out: Vec::new(),
            inv_diag: Vec::new(),
            #[cfg(feature = "blas")]
            tau: Vec::new(),
            #[cfg(feature = "blas")]
            lapack_work: Vec::new(),
            e_cm_ready: false,
            g_cm_ready: false,
            #[cfg(feature = "blas")]
            lapack_lwork: 0,
            #[cfg(feature = "blas")]
            lapack_cached_me: 0,
            #[cfg(feature = "blas")]
            lapack_cached_n: 0,
        }
    }

    /// Prepare LSI buffers.
    pub fn prepare(&mut self, me: usize, mg: usize, n: usize) {
        if !self.e_cm_ready {
            self.e_cm.resize_zero(me, n);
        }
        ensure_vec_len(&mut self.f_arr, me);
        ensure_vec_len(&mut self.h_arr, mg);
        ensure_vec_len(&mut self.x_out, n);
        ensure_vec_len(&mut self.inv_diag, n);
        if !self.g_cm_ready {
            self.g_cm.resize_zero(mg, n);
        }
        // tau and lapack_work are sized on first use by lsi_qr_factor_lapack
    }
}

/// LSI + LDP + NNLS chain workspace (passed to `lsi_ws`).
pub struct LsiChainWs {
    pub lsi: LsiWs,
    pub inner: LdpChainWs,
}

impl LsiChainWs {
    pub fn new() -> Self {
        LsiChainWs {
            lsi: LsiWs::new(),
            inner: LdpChainWs::new(),
        }
    }
}

/// LSEI-level pre-allocated temporaries.
pub struct LseiWs {
    /// Row-major working copy of C (mc × n) — required by `h12_construct`/`h12_apply`.
    pub(crate) c_mat: Mat,
    /// Row-major working copy of E (me × n) — Householder factors applied row-major.
    pub(crate) e_mat: Mat,
    /// Row-major working copy of G (mg × n) — Householder factors applied row-major.
    pub(crate) g_mat: Mat,
    /// LSEI working copy of d (length mc).
    pub d_arr: Vec<f64>,
    /// LSEI working copy of f (length me).
    pub f_arr: Vec<f64>,
    /// LSEI working copy of h (length mg).
    pub h_arr: Vec<f64>,
    /// LSEI Householder scalars for C (length mc+1).
    pub w_hh: Vec<f64>,
    /// LSEI scratch buffer for breaking aliasing (length n).
    pub u_buf: Vec<f64>,
    /// LSEI output vector (length n).
    pub x_out: Vec<f64>,
    /// LSEI output dual vector (length mc+mg).
    pub w_out: Vec<f64>,
    /// LSEI reduced f (length me).
    pub f_reduced: Vec<f64>,
    /// LSEI sub-matrix of E (me × l), column-major — passed to lsi_ws.
    pub e_sub: ColMat,
    /// LSEI sub-matrix of G (mg × l), column-major — passed to lsi_ws.
    pub g_sub: ColMat,
    /// LSEI HFTI RHS matrix (column-major — passed to hfti).
    pub b_hfti: ColMat,
    /// LSEI temporary ColMat for hfti(a) — avoids allocation on repeat calls.
    pub hfti_a_tmp: ColMat,
}

impl LseiWs {
    pub fn new() -> Self {
        LseiWs {
            c_mat: Mat::zeros(0, 0),
            e_mat: Mat::zeros(0, 0),
            g_mat: Mat::zeros(0, 0),
            d_arr: Vec::new(),
            f_arr: Vec::new(),
            h_arr: Vec::new(),
            w_hh: Vec::new(),
            u_buf: Vec::new(),
            x_out: Vec::new(),
            w_out: Vec::new(),
            f_reduced: Vec::new(),
            e_sub: ColMat::zeros(0, 0),
            g_sub: ColMat::zeros(0, 0),
            b_hfti: ColMat::zeros(0, 0),
            hfti_a_tmp: ColMat::zeros(0, 0),
        }
    }

    /// Prepare LSEI buffers.
    pub fn prepare(&mut self, mc: usize, me: usize, mg: usize, n: usize) {
        let mc1 = mc.max(1);
        let me1 = me.max(1);
        let mg1 = mg.max(1);
        let l = n.saturating_sub(mc);
        self.c_mat.resize_zero(mc1, n);
        self.e_mat.resize_zero(me1, n);
        self.g_mat.resize_zero(mg1, n);
        ensure_vec_len(&mut self.d_arr, mc1);
        ensure_vec_len(&mut self.f_arr, me);
        ensure_vec_len(&mut self.h_arr, mg);
        ensure_vec_len(&mut self.w_hh, mc + 1);
        ensure_vec_len(&mut self.u_buf, n);
        ensure_vec_len(&mut self.x_out, n);
        ensure_vec_len(&mut self.w_out, mc + mg);
        ensure_vec_len(&mut self.f_reduced, me);
        self.e_sub.resize_zero(me, l);
        self.g_sub.resize_zero(mg1, l);
        let rows_hfti = me.max(l);
        self.b_hfti.resize_zero(rows_hfti, 1);
    }
}

/// LSEI + LSI + LDP + NNLS chain workspace (passed to `lsei_ws`).
pub struct LseiChainWs {
    pub lsei: LseiWs,
    pub inner: LsiChainWs,
}

impl LseiChainWs {
    pub fn new() -> Self {
        LseiChainWs {
            lsei: LseiWs::new(),
            inner: LsiChainWs::new(),
        }
    }
}

/// LSQ-level pre-allocated temporaries.
pub struct LsqWs {
    /// LSQ column-major E flat buffer (n × n+1).
    pub e_flat: Vec<f64>,
    /// LSQ reduced f (length n).
    pub f_arr: Vec<f64>,
    /// LSQ E as column-major ColMat (n × n).
    pub e_cm: ColMat,
    /// LSQ equality constraint matrix C (meq × n), column-major.
    pub c_cm: ColMat,
    /// LSQ equality RHS d (length meq).
    pub d_arr: Vec<f64>,
    /// LSQ inequality matrix G (mg × n), column-major.
    pub g_cm: ColMat,
    /// LSQ inequality RHS h (length mg).
    pub h_arr: Vec<f64>,
}

impl LsqWs {
    pub fn new() -> Self {
        LsqWs {
            e_flat: Vec::new(),
            f_arr: Vec::new(),
            e_cm: ColMat::zeros(0, 0),
            c_cm: ColMat::zeros(0, 0),
            d_arr: Vec::new(),
            g_cm: ColMat::zeros(0, 0),
            h_arr: Vec::new(),
        }
    }

    /// Prepare LSQ buffers.
    pub fn prepare(&mut self, n: usize, meq: usize, mg: usize) {
        let n1 = n + 1;
        let meq1 = meq.max(1);
        ensure_vec_len(&mut self.e_flat, n * n1);
        ensure_vec_len(&mut self.f_arr, n);
        self.e_cm.resize_zero(n, n);
        self.c_cm.resize_zero(meq1, n);
        ensure_vec_len(&mut self.d_arr, meq1);
        self.g_cm.resize_zero(mg.max(1), n);
        ensure_vec_len(&mut self.h_arr, mg.max(1));
    }
}

/// Pre-allocated workspace for the entire least-squares constraint chain
/// (`lsq` → `lsei` → `lsi` → `ldp` → `nnls`).
///
/// Split into nested per-level sub-structs so that each function can borrow
/// its own temporaries independently from the sub-workspaces it passes down,
/// satisfying Rust's disjoint-field borrow rules.
///
/// This eliminates ~1MB of allocation traffic per SQP iteration for n≈200.
pub struct LsWorkspace {
    /// LSQ-level temporaries (used by `lsq_ws`).
    pub lsq: LsqWs,
    /// Chain workspace for LSEI → LSI → LDP → NNLS.
    pub inner: LseiChainWs,
}

impl LsWorkspace {
    /// Create a new empty workspace (all buffers start at zero capacity).
    ///
    /// Buffers will grow lazily on first use via the per-level `prepare` methods.
    pub fn new() -> Self {
        LsWorkspace {
            lsq: LsqWs::new(),
            inner: LseiChainWs::new(),
        }
    }
}

impl Default for LsWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Mat (row-major workspace) -------------------------------------------

    #[test]
    fn mat_zeros_and_indexing() {
        let mut m = Mat::zeros(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert_eq!(m.stride, 4);
        assert_eq!(m.data.len(), 12);
        assert!(m.data.iter().all(|&v| v == 0.0));

        m[(1, 2)] = 7.5;
        assert_eq!(m.data[1 * 4 + 2], 7.5);
        assert_eq!(m[(1, 2)], 7.5);
    }

    #[test]
    fn mat_resize_zero() {
        let mut m = Mat::zeros(2, 3);
        m[(0, 0)] = 99.0;
        m.resize_zero(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert_eq!(m.stride, 4);
        assert!(m.data[..12].iter().all(|&v| v == 0.0));
    }

    // -- LinminData / SlsqpbData defaults ------------------------------------

    #[test]
    fn linmin_data_default_all_zero() {
        let d = LinminData::default();
        assert_eq!(d.a, 0.0);
        assert_eq!(d.fx, 0.0);
        assert_eq!(d.tol2, 0.0);
    }

    #[test]
    fn slsqpb_data_default_all_zero() {
        let d = SlsqpbData::default();
        assert_eq!(d.t, 0.0);
        assert_eq!(d.line, 0);
        assert_eq!(d.n3, 0);
    }
}
