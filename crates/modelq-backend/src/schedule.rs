//! Dynamic work distribution shared by the parallel CPU paths.
//!
//! Splitting a chunk into one equal range per worker leaves the whole chunk
//! waiting on the slowest core, which on a hybrid performance/efficiency CPU
//! (or a core shared with other work) can be several times slower than the
//! fastest.  Instead each chunk is cut into several small ranges per worker
//! and the workers take the next range from a shared queue, so a slow core
//! simply processes fewer ranges.  Results are returned in range order, so
//! the output never depends on which thread ran which range.

use std::{
    sync::{Mutex, PoisonError},
    thread,
};

/// Ranges handed out per worker, so faster cores can take more of them.
pub(crate) const RANGES_PER_WORKER: usize = 4;

/// A worker thread panicked while running a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerPanicked;

/// Runs `work` on every item with up to `threads` scoped threads and returns
/// the results in item order.  With one item or one thread the work runs on
/// the calling thread and no thread is spawned.
pub(crate) fn run_all<T, R>(
    items: Vec<T>,
    threads: usize,
    work: impl Fn(usize, T) -> R + Sync,
) -> Result<Vec<R>, WorkerPanicked>
where
    T: Send,
    R: Send,
{
    let threads = threads.min(items.len());
    if threads <= 1 {
        return Ok(items
            .into_iter()
            .enumerate()
            .map(|(index, item)| work(index, item))
            .collect());
    }
    let queue = Mutex::new(items.into_iter().enumerate());
    let (queue, work) = (&queue, &work);
    let finished: Vec<Vec<(usize, R)>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        // The lock is only held to take the next item, never
                        // while `work` runs, so a panic cannot poison it.
                        let next = queue.lock().unwrap_or_else(PoisonError::into_inner).next();
                        let Some((index, item)) = next else { break };
                        done.push((index, work(index, item)));
                    }
                    done
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().map_err(|_| WorkerPanicked))
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut results: Vec<(usize, R)> = finished.into_iter().flatten().collect();
    results.sort_by_key(|(index, _)| *index);
    Ok(results.into_iter().map(|(_, result)| result).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{WorkerPanicked, run_all};

    #[test]
    fn results_come_back_in_item_order_for_any_thread_count() {
        for threads in [0, 1, 2, 7, 64] {
            let items: Vec<usize> = (0..1000).collect();
            let results = run_all(items, threads, |index, item| {
                assert_eq!(index, item);
                item * 2
            })
            .expect("no panic");
            assert_eq!(results, (0..1000).map(|item| item * 2).collect::<Vec<_>>());
        }
    }

    #[test]
    fn every_item_runs_exactly_once() {
        let counter = AtomicUsize::new(0);
        let results = run_all(vec![(); 5000], 8, |_, ()| {
            counter.fetch_add(1, Ordering::Relaxed);
        })
        .expect("no panic");
        assert_eq!(results.len(), 5000);
        assert_eq!(counter.load(Ordering::Relaxed), 5000);
    }

    #[test]
    fn empty_and_single_item_inputs_do_not_spawn() {
        let empty: Vec<u8> = run_all(Vec::<u8>::new(), 8, |_, item| item).unwrap();
        assert!(empty.is_empty());
        let main = std::thread::current().id();
        let ran_on = run_all(vec![()], 8, |_, ()| std::thread::current().id()).unwrap();
        assert_eq!(ran_on, [main]);
    }

    #[test]
    fn a_panicking_worker_is_reported() {
        let result = run_all(vec![0, 1, 2, 3], 4, |_, item| {
            assert!(item != 2, "boom");
            item
        });
        assert_eq!(result, Err(WorkerPanicked));
    }
}
