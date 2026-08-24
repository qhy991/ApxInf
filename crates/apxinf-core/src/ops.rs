//! CPU math operations with optional BLAS acceleration.
//!
//! When a BLAS feature (`accelerate` or `openblas`) is enabled at compile time,
//! matrix multiplication is dispatched to the vendor BLAS library. Otherwise a
//! naive Rust fallback is used.

/// Row-major general matrix multiply:  C = alpha * A * B + beta * C
///
/// - A: [M, K] row-major
/// - B: [K, N] row-major
/// - C: [M, N] row-major (output, must be pre-allocated)
///
/// This is the single-batch variant. Callers handle batching externally.
pub fn sgemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    #[cfg(feature = "accelerate")]
    {
        sgemm_accelerate(m, k, n, a, b, c);
    }
    #[cfg(all(feature = "openblas", not(feature = "accelerate")))]
    {
        sgemm_openblas(m, k, n, a, b, c);
    }
    #[cfg(not(any(feature = "accelerate", feature = "openblas")))]
    {
        sgemm_naive(m, k, n, a, b, c);
    }
}

/// Row-major matrix multiply with an implicitly transposed right-hand side:
/// `C[M, N] = A[M, K] @ B[N, K]^T`.
///
/// Keeping B in `[N, K]` form is important for tied token embeddings: the same
/// storage can serve embedding lookup and output projection without allocating
/// a full transposed vocabulary matrix.
pub fn sgemm_rhs_transposed(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    #[cfg(feature = "accelerate")]
    {
        sgemm_rhs_transposed_accelerate(m, k, n, a, b, c);
    }
    #[cfg(all(feature = "openblas", not(feature = "accelerate")))]
    {
        sgemm_rhs_transposed_openblas(m, k, n, a, b, c);
    }
    #[cfg(not(any(feature = "accelerate", feature = "openblas")))]
    {
        sgemm_rhs_transposed_naive(m, k, n, a, b, c);
    }
}

// ── Naive fallback ──────────────────────────────────────────────────

#[cfg(not(any(feature = "accelerate", feature = "openblas")))]
fn sgemm_naive(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

#[cfg(not(any(feature = "accelerate", feature = "openblas")))]
fn sgemm_rhs_transposed_naive(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
}

// ── Apple Accelerate (CBLAS) ────────────────────────────────────────

#[cfg(feature = "accelerate")]
fn sgemm_accelerate(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use cblas::{sgemm as cblas_sgemm, Layout, Transpose};

    // Our data is row-major; CBLAS uses column-major by default.
    // Row-major C = A @ B  is equivalent to  column-major C^T = B^T @ A^T
    // so we swap A↔B and transpose both.
    unsafe {
        cblas_sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m as i32, // M of output
            n as i32, // N of output
            k as i32, // K (shared dim)
            1.0,      // alpha
            a,        // A: [M, K] row-major
            k as i32, // lda (stride of row in A)
            b,        // B: [K, N] row-major
            n as i32, // ldb (stride of row in B)
            0.0,      // beta
            c,        // C: [M, N] row-major
            n as i32, // ldc (stride of row in C)
        );
    }
}

#[cfg(feature = "accelerate")]
fn sgemm_rhs_transposed_accelerate(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    use cblas::{sgemm as cblas_sgemm, Layout, Transpose};

    unsafe {
        cblas_sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            k as i32,
            0.0,
            c,
            n as i32,
        );
    }
}

// ── OpenBLAS ────────────────────────────────────────────────────────

#[cfg(all(feature = "openblas", not(feature = "accelerate")))]
fn sgemm_openblas(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use cblas::{sgemm as cblas_sgemm, Layout, Transpose};

    // Same row-major CBLAS call as Accelerate — the cblas crate provides a
    // uniform API; the only difference is the linked library underneath.
    unsafe {
        cblas_sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            n as i32,
            0.0,
            c,
            n as i32,
        );
    }
}

#[cfg(all(feature = "openblas", not(feature = "accelerate")))]
fn sgemm_rhs_transposed_openblas(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    use cblas::{sgemm as cblas_sgemm, Layout, Transpose};

    unsafe {
        cblas_sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::Ordinary,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            k as i32,
            0.0,
            c,
            n as i32,
        );
    }
}
