//! Core types shared across the Stratum workspace.
//!
//! See `plan/29-error-taxonomy-and-logging.md` for the error-code policy and
//! `plan/16-multi-llm-providers.md` for the Provider/Registry roles these
//! enums feed into.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod block;
mod capability;
/// Stable error taxonomy and the `StratumError` type. The `codes` submodule
/// is the catalog every other crate should reach for.
pub mod error;
mod ids;
mod mem;

pub use block::{AudioData, Block, ImageData};
pub use capability::{Capability, ConcurrencyModel, Family};
pub use error::{ErrorCode, StratumError, StratumResult};
pub use ids::{ModelId, RoleId};
pub use mem::MemEstimate;
