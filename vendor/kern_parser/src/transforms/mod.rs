//! Bar splitting and padding for synthetic data.
pub mod layout;
pub mod pad_final_measure;
pub mod split;
pub use pad_final_measure::pad_final_measure_with_rests;
pub use split::{
    musical_bar_count, slice_measure_range, split_trailing_bars, SliceMeasuresError, SplitBars,
    TruncateBarsError,
};
