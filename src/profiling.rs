//! Optional compile-time profiling instrumentation.
//!
//! When the `profiling` Cargo feature is enabled, this module provides
//! thread-local timing accumulators that track cumulative time and call
//! counts for each instrumented code section.
//!
//! When `profiling` is disabled, all macros expand to no-ops with zero
//! runtime cost.

#[cfg(feature = "profiling")]
pub mod inner {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::time::Instant;

    /// Per-section timing accumulator.
    #[derive(Default, Clone)]
    pub struct SectionStats {
        pub calls: u64,
        pub total_ns: u64,
    }

    thread_local! {
        pub static STATS: RefCell<HashMap<&'static str, SectionStats>> =
            RefCell::new(HashMap::new());
    }

    /// Record elapsed time for a named section.
    #[inline]
    pub fn record(name: &'static str, elapsed_ns: u64) {
        STATS.with(|s| {
            let mut map = s.borrow_mut();
            let entry = map.entry(name).or_default();
            entry.calls += 1;
            entry.total_ns += elapsed_ns;
        });
    }

    /// Get a snapshot of all accumulated stats.
    pub fn get_stats() -> HashMap<&'static str, SectionStats> {
        STATS.with(|s| s.borrow().clone())
    }

    /// Reset all accumulated stats.
    pub fn reset_stats() {
        STATS.with(|s| s.borrow_mut().clear());
    }

    /// RAII guard that records elapsed time on drop.
    pub struct TimingGuard {
        name: &'static str,
        start: Instant,
    }

    impl TimingGuard {
        #[inline]
        pub fn new(name: &'static str) -> Self {
            TimingGuard {
                name,
                start: Instant::now(),
            }
        }
    }

    impl Drop for TimingGuard {
        #[inline]
        fn drop(&mut self) {
            let elapsed = self.start.elapsed().as_nanos() as u64;
            record(self.name, elapsed);
        }
    }
}

/// Start timing a named section. Returns a guard that records on drop.
/// Expands to a no-op when the `profiling` feature is disabled.
#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! profile_section {
    ($name:expr) => {
        let _guard = $crate::profiling::inner::TimingGuard::new($name);
    };
}

#[cfg(not(feature = "profiling"))]
#[macro_export]
macro_rules! profile_section {
    ($name:expr) => {};
}
