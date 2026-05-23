# Default recipe: list available recipes
default:
    @just --list


# Build the package (develop mode, installs into venv)
[group('build')]
build:
    uv run maturin develop

# Build the package in release mode
[group('build')]
release:
    uv run maturin develop --release

# Build in release mode with BLAS/Accelerate support
[group('build')]
release-blas:
    uv run maturin develop --release --features blas

# Build a wheel
[group('build')]
wheel:
    uv run maturin build --release --features blas

# Clean build artifacts
[group('build')]
clean:
    cargo clean
    rm -rf target/ dist/ *.egg-info


# Run Python tests
[group('test')]
test-py: build
    uv run pytest

# Run Rust tests
[group('test')]
test-rs:
    cargo test

# Run all tests (Rust + Python)
[group('test')]
test: test-rs test-py


# Check Rust formatting
[group('lint')]
check-rs:
    cargo fmt -- --check

# Check Python formatting
[group('lint')]
check-py:
    uv run ruff format --check src/rslsqp

# Check formatting for all code (Rust + Python)
[group('lint')]
check: check-rs check-py

# Type-check with ty
[group('lint')]
typecheck:
    uv run ty check src/rslsqp

# Lint Rust code
[group('lint')]
lint-rs:
    cargo clippy -- -D warnings

# Lint Python code
[group('lint')]
lint-py:
    uv run ruff check src/rslsqp

# Lint and format check
[group('lint')]
lint: lint-rs check-rs lint-py check-py


# Format Rust code
[group('format')]
fmt-rs:
    cargo fmt

# Format Python code
[group('format')]
fmt-py:
    uv run ruff format src/rslsqp

# Format all code (Rust + Python)
[group('format')]
fmt: fmt-rs fmt-py


# Run benchmarks (requires scipy; uses release build)
[group('bench')]
benchmark: release
    uv run python benchmarks/benchmark.py

# Run benchmarks with BLAS/Accelerate (macOS: Apple AMX/NEON, Linux: OpenBLAS)
[group('bench')]
benchmark-blas: release-blas
    uv run python benchmarks/benchmark.py
