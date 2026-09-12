//! Tick message queues. The bevy build used `Message` events
//! (`FloorStarted`, `ReactiveAudioRequest`, `UiBridgeAction`,
//! `GamepadRumbleRequest`); here each channel is a drained `Vec`
//! resource — same single-tick delivery, no engine events needed.
//!
//! Concrete payloads land with their owner modules (areas, audio, ui);
//! this file owns the queue mechanics both sides share.

use bevy_ecs::prelude::*;
use std::collections::VecDeque;

/// One message channel. Writers push during systems; exactly one
/// consumer drains per tick (FIFO, bevy `MessageReader` parity).
#[derive(Resource, Debug)]
pub struct Queue<T: Send + Sync + 'static> {
    pending: VecDeque<T>,
}

impl<T: Send + Sync + 'static> Default for Queue<T> {
    fn default() -> Self {
        Self {
            pending: VecDeque::new(),
        }
    }
}

impl<T: Send + Sync + 'static> Queue<T> {
    pub fn push(&mut self, msg: T) {
        self.pending.push_back(msg);
    }

    /// Drain everything queued (call once per tick per channel).
    pub fn drain(&mut self) -> Vec<T> {
        self.pending.drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_is_fifo_and_drains_once() {
        let mut q: Queue<(u32, &'static str)> = Queue::default();
        q.push((1, "a"));
        q.push((2, "b"));
        assert_eq!(q.len(), 2);
        let got = q.drain();
        assert_eq!(got, vec![(1, "a"), (2, "b")]);
        assert!(q.is_empty());
        assert!(q.drain().is_empty());
    }
}
