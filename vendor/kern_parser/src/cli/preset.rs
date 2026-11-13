use crate::{ParseDialect, SerializePreset};
use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(in crate::cli) enum InputDialectArg {
    /// Standard Humdrum **kern input.
    HumdrumKern,
    /// Transcoda v2 OMR target dialect input.
    TranscodaV2,
}

impl From<InputDialectArg> for ParseDialect {
    fn from(value: InputDialectArg) -> Self {
        match value {
            InputDialectArg::HumdrumKern => ParseDialect::HumdrumKern,
            InputDialectArg::TranscodaV2 => ParseDialect::TranscodaV2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(in crate::cli) enum SerializePresetArg {
    /// Renderable standard Humdrum **kern for notation engines.
    HumdrumKern,
    /// Renderable Humdrum **kern containing only **kern spines.
    KernOnly,

    /// Current Transcoda transcription target with compact barline lines and empty data nulls.
    TranscodaV2,
    /// Transcoda v2 training labels with editorial accidental signifiers stripped.
    TranscodaV2Training,
}

impl From<SerializePresetArg> for SerializePreset {
    fn from(value: SerializePresetArg) -> Self {
        match value {
            SerializePresetArg::HumdrumKern => SerializePreset::HumdrumKern,
            SerializePresetArg::KernOnly => SerializePreset::KernOnly,

            SerializePresetArg::TranscodaV2 => SerializePreset::TranscodaV2,
            SerializePresetArg::TranscodaV2Training => SerializePreset::TranscodaV2Training,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub(in crate::cli) enum FramingArg {
    #[default]
    Complete,
    Excerpt,
}
impl From<FramingArg> for crate::document::Framing {
    fn from(value: FramingArg) -> Self {
        match value {
            FramingArg::Complete => Self::Complete,
            FramingArg::Excerpt => Self::Excerpt,
        }
    }
}
