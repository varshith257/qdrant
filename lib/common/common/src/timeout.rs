use std::time::Duration;

use tokio::time::Instant;

/// Calculate the remaining timeout, clamped to a minimum value.
pub fn calculate_timeout(
    timeout_secs: Option<f64>,
    start: Instant,
    min_timeout_secs: f64,
) -> Option<Duration> {
    timeout_secs.map(|t| {
        let elapsed = start.elapsed();
        Duration::from_secs_f64(t)
            .saturating_sub(elapsed)
            .max(Duration::from_secs_f64(min_timeout_secs))
    })
}
