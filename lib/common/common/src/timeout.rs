use std::time::{Duration, Instant as StdInstant};

use tokio::time::Instant as TokioInstant;

/// Trait to abstract the behavior of `elapsed` for different `Instant` types
pub trait InstantTrait {
    fn elapsed(&self) -> Duration;
}

/// Implement `InstantTrait` for `std::time::Instant`
impl InstantTrait for StdInstant {
    fn elapsed(&self) -> Duration {
        StdInstant::elapsed(self)
    }
}

/// Implement `InstantTrait` for `tokio::time::Instant`
impl InstantTrait for TokioInstant {
    fn elapsed(&self) -> Duration {
        TokioInstant::elapsed(self)
    }
}

/// Calculate the remaining timeout, clamped to a minimum value.
pub fn calculate_timeout<I: InstantTrait>(
    timeout_secs: Option<f64>,
    start: I,
    min_timeout_secs: f64,
) -> Option<Duration> {
    timeout_secs.map(|t| {
        let elapsed = start.elapsed();
        Duration::from_secs_f64(t)
            .saturating_sub(elapsed)
            .max(Duration::from_secs_f64(min_timeout_secs))
    })
}
