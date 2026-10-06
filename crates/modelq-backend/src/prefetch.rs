//! Chunk feeding for the parallel backends, optionally overlapped with
//! compute.
//!
//! The parallel paths read a replayable source in chunks: the source iterator
//! (a memory-mapped tensor converting `F16`/`BF16` to `f32`, say) produces one
//! chunk at a time, and workers then process it.  Reading is serial, so on a
//! many-core machine it becomes about half of the run time.  A
//! [`Feed`] hides where the chunks come from:
//!
//! - an *inline* feed fills each chunk on the calling thread, exactly as the
//!   original loops did; and
//! - a *prefetching* feed runs the source on one reader thread that fills the
//!   next chunk while workers process the current one.
//!
//! Both deliver the same chunks in the same order, so results never depend on
//! which one ran.  Memory stays bounded: a fixed pool of chunk buffers is
//! recycled between the reader and the consumer (two in flight plus the one
//! being processed).
//!
//! The value factory runs on the reader thread, so it must be `Send`; the
//! iterator it returns is created and consumed there and need not be.

use std::{
    sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel},
    thread,
};

use crate::schedule::WorkerPanicked;

/// Chunk buffers circulating between the reader and the consumer, in
/// addition to the one the consumer is processing.
const POOL_BUFFERS: usize = 2;

/// Delivers a replayable source as a sequence of passes of chunks.
pub(crate) trait Feed {
    /// Starts the next pass over the source.
    fn start_pass(&mut self);

    /// Replaces `buffer`'s contents with the next chunk of the current pass.
    /// An empty buffer means the pass has ended.
    fn fill(&mut self, buffer: &mut Vec<f32>) -> Result<(), WorkerPanicked>;
}

struct InlineFeed<F, I: IntoIterator> {
    values: F,
    source: Option<I::IntoIter>,
    chunk_values: usize,
}

impl<F, I> Feed for InlineFeed<F, I>
where
    F: FnMut() -> I,
    I: IntoIterator<Item = f32>,
{
    fn start_pass(&mut self) {
        self.source = Some((self.values)().into_iter());
    }

    fn fill(&mut self, buffer: &mut Vec<f32>) -> Result<(), WorkerPanicked> {
        buffer.clear();
        if let Some(source) = self.source.as_mut() {
            buffer.extend(source.by_ref().take(self.chunk_values));
        }
        Ok(())
    }
}

struct PrefetchFeed {
    chunks: Receiver<Vec<f32>>,
    recycle: Sender<Vec<f32>>,
}

impl Feed for PrefetchFeed {
    fn start_pass(&mut self) {
        // The reader runs through every pass on its own; nothing to do.
    }

    fn fill(&mut self, buffer: &mut Vec<f32>) -> Result<(), WorkerPanicked> {
        let mut next = self.chunks.recv().map_err(|_| WorkerPanicked)?;
        // Hand the consumer the prefetched chunk and return its old buffer to
        // the reader for refilling.
        std::mem::swap(buffer, &mut next);
        // The reader is gone only after the last pass; a failed send then is
        // harmless.
        let _ = self.recycle.send(next);
        Ok(())
    }
}

/// Runs `consume` with a [`Feed`] over `passes` replays of `values`.
///
/// With `overlap` the source is read on a separate reader thread; otherwise
/// on the calling thread.  `chunk_values` is the number of values per chunk,
/// and `capacity` the buffer capacity to preallocate (at most one chunk).
pub(crate) fn with_feed<F, I, R>(
    values: F,
    chunk_values: usize,
    capacity: usize,
    passes: usize,
    overlap: bool,
    consume: impl FnOnce(&mut dyn Feed) -> R,
) -> Result<R, WorkerPanicked>
where
    F: FnMut() -> I + Send,
    I: IntoIterator<Item = f32>,
{
    if !overlap {
        let mut feed = InlineFeed {
            values,
            source: None,
            chunk_values,
        };
        return Ok(consume(&mut feed));
    }

    thread::scope(|scope| {
        let (chunks_tx, chunks_rx): (SyncSender<Vec<f32>>, _) = sync_channel(POOL_BUFFERS);
        let (recycle_tx, recycle_rx) = channel::<Vec<f32>>();
        for _ in 0..POOL_BUFFERS {
            let _ = recycle_tx.send(Vec::with_capacity(capacity));
        }

        let reader = scope.spawn(move || {
            let mut values = values;
            for _ in 0..passes {
                let mut source = values().into_iter();
                loop {
                    let Ok(mut buffer) = recycle_rx.recv() else {
                        return;
                    };
                    buffer.clear();
                    buffer.extend(source.by_ref().take(chunk_values));
                    let finished = buffer.is_empty();
                    if chunks_tx.send(buffer).is_err() {
                        return;
                    }
                    if finished {
                        break;
                    }
                }
            }
        });

        let mut feed = PrefetchFeed {
            chunks: chunks_rx,
            recycle: recycle_tx,
        };
        let result = consume(&mut feed);
        // Closing both channels lets a still-running reader exit.
        drop(feed);
        match reader.join() {
            Ok(()) => Ok(result),
            Err(_) => Err(WorkerPanicked),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{Feed, with_feed};
    use crate::schedule::WorkerPanicked;

    /// Reads every pass completely and returns the chunk lengths per pass and
    /// all values.
    fn drain(feed: &mut dyn Feed, passes: usize) -> Vec<(Vec<usize>, Vec<f32>)> {
        let mut buffer = Vec::new();
        (0..passes)
            .map(|_| {
                feed.start_pass();
                let (mut lengths, mut all) = (Vec::new(), Vec::new());
                loop {
                    feed.fill(&mut buffer).expect("feed works");
                    if buffer.is_empty() {
                        break;
                    }
                    lengths.push(buffer.len());
                    all.extend_from_slice(&buffer);
                }
                (lengths, all)
            })
            .collect()
    }

    #[test]
    fn inline_and_prefetching_feeds_deliver_identical_chunks() {
        for count in [0_usize, 1, 15, 16, 17, 100, 1000, 4097] {
            let values: Vec<f32> = (0..count).map(|index| index as f32 * 0.5).collect();
            for chunk in [1_usize, 7, 16, 100, 5000] {
                let run = |overlap| {
                    with_feed(
                        || values.iter().copied(),
                        chunk,
                        chunk.min(count),
                        3,
                        overlap,
                        |feed| drain(feed, 3),
                    )
                    .expect("no panic")
                };
                let (inline, overlapped) = (run(false), run(true));
                assert_eq!(inline, overlapped, "count={count} chunk={chunk}");
                for (lengths, all) in &inline {
                    assert_eq!(all, &values);
                    assert!(lengths.iter().all(|&length| length <= chunk));
                }
            }
        }
    }

    #[test]
    fn a_consumer_that_stops_early_does_not_hang_or_leak_the_reader() {
        let values: Vec<f32> = (0..100_000).map(|index| index as f32).collect();
        let first = with_feed(
            || values.iter().copied(),
            64,
            64,
            2,
            true,
            |feed| {
                let mut buffer = Vec::new();
                feed.start_pass();
                feed.fill(&mut buffer).unwrap();
                buffer.len()
            },
        )
        .expect("no panic");
        assert_eq!(first, 64);
    }

    #[test]
    fn a_panicking_source_is_reported_not_propagated() {
        let result = with_feed(
            || {
                (0..10).map(|index| {
                    assert!(index != 5, "source failed");
                    index as f32
                })
            },
            4,
            4,
            1,
            true,
            |feed| {
                let mut buffer = Vec::new();
                feed.start_pass();
                loop {
                    if feed.fill(&mut buffer).is_err() {
                        return Err(WorkerPanicked);
                    }
                    if buffer.is_empty() {
                        return Ok(());
                    }
                }
            },
        );
        assert_eq!(result, Err(WorkerPanicked));
    }
}
