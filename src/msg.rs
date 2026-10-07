//! Tick message queues - port-only: GML has no event bus, callers write
//! straight into the owner instance. Each channel is a drained `VecDeque`
//! resource, FIFO, one drain per tick.
//! Concrete payloads land with their owner modules (areas, audio, ui); this file
//! owns the queue mechanics every channel shares.

use bevy_ecs::prelude::*;
use std::collections::VecDeque;

/// One message channel. Writers push during systems; exactly one
/// consumer drains per tick (FIFO).
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
