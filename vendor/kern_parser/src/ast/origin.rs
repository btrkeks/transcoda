//! Field-level source provenance for [`KernDocument`](super::KernDocument)
//! lines.
//!
//! A parsed line comes from one physical source record. Passes that merge,
//! project, relocate or synthesize lines must say where each resulting field
//! came from, so source locations can identify the original bytes of
//! every field it keeps and report every change against the record it came
//! from. A pass that only edits values in place keeps [`LineOrigin::Record`].
//!
//! Example: merging `*clefG2\t*` (record 4) with `*\t*clefF4` (record 5) gives
//! a line owned by record 4 whose second field comes from record 5:
//! ```text
//! Fields { owner: 4, fields: [Source { record: 4, field: 0 }, Source { record: 5, field: 1 }] }
//! ```
//!
//! [`KernDocument`](super::KernDocument)'s equality ignores origins, so a pass
//! that only relabels provenance cannot change serialized output or the
//! fixpoint of a normalization pipeline.

use std::num::NonZeroUsize;

/// 1-based physical record of the parsed source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordId(NonZeroUsize);

impl RecordId {
    /// `None` for record 0, which no parser produces.
    pub fn new(record_1based: usize) -> Option<Self> {
        NonZeroUsize::new(record_1based).map(Self)
    }

    pub fn get(self) -> usize {
        self.0.get()
    }
}

/// Where one output field's bytes may come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldOrigin {
    /// Field `field` (0-based) of record `record`.
    Source { record: RecordId, field: usize },
    /// No source field: a null a pass wrote, or content it synthesized.
    None,
}

/// Provenance of one line.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LineOrigin {
    /// Field `i` comes from field `i` of the record. Parsed lines start here,
    /// and in-place value edits keep it.
    Record(RecordId),
    /// Per-field provenance, for merged, column-projected, relocated or
    /// widened lines. `owner` is the record whose text the line replaces.
    Fields {
        owner: RecordId,
        fields: Vec<FieldOrigin>,
    },
    /// No source line: content the parser or a pass synthesized, such as the
    /// default `*clefG2` the parser inserts. Reports source line 0.
    Generated,
    /// No source line was recorded: documents assembled without provenance.
    /// Reports the line's 1-based position as its source line number.
    Untracked,
}

impl LineOrigin {
    /// Whole-record provenance from an optional 1-based record number, the
    /// shape every pre-provenance caller has. Record 0 is the parser's marker
    /// for a generated line.
    pub fn from_record(record: Option<usize>) -> Self {
        match record {
            None => Self::Untracked,
            Some(record) => RecordId::new(record).map_or(Self::Generated, Self::Record),
        }
    }

    /// Whether the line has no source record to copy from.
    pub fn is_sourceless(&self) -> bool {
        self.owner().is_none()
    }

    /// The record whose source text this line replaces.
    pub fn owner(&self) -> Option<RecordId> {
        match self {
            Self::Record(record) | Self::Fields { owner: record, .. } => Some(*record),
            Self::Generated | Self::Untracked => None,
        }
    }

    /// Origin of field `field`.
    pub fn field(&self, field: usize) -> FieldOrigin {
        match self {
            Self::Record(record) => FieldOrigin::Source {
                record: *record,
                field,
            },
            Self::Fields { fields, .. } => fields.get(field).copied().unwrap_or(FieldOrigin::None),
            Self::Generated | Self::Untracked => FieldOrigin::None,
        }
    }

    /// The line keeps only the source columns `retained` (0-based, in output
    /// order).
    pub fn projected(&self, retained: &[usize]) -> Self {
        match self.owner() {
            None => self.clone(),
            Some(owner) => Self::Fields {
                owner,
                fields: retained.iter().map(|&column| self.field(column)).collect(),
            },
        }
    }

    /// Two same-width lines fused into one. `self` keeps ownership (or
    /// `other` when `self` is generated); field `k` comes from `other` where
    /// `take_from_other(k)`.
    pub fn merged(
        &self,
        other: &Self,
        take_from_other: impl Fn(usize) -> bool,
        width: usize,
    ) -> Self {
        let Some(owner) = self.owner().or_else(|| other.owner()) else {
            return self.clone();
        };
        let fields: Vec<_> = (0..width)
            .map(|column| {
                if take_from_other(column) {
                    other.field(column)
                } else {
                    self.field(column)
                }
            })
            .collect();
        Self::Fields { owner, fields }.simplified()
    }

    /// Field `from` moved to column `to` of the same line; column `from`
    /// keeps no source.
    pub fn relocated(&self, from: usize, to: usize, width: usize) -> Self {
        let Some(owner) = self.owner() else {
            return self.clone();
        };
        let fields = (0..width)
            .map(|column| {
                if column == to {
                    self.field(from)
                } else if column == from {
                    FieldOrigin::None
                } else {
                    self.field(column)
                }
            })
            .collect();
        Self::Fields { owner, fields }
    }

    /// Sourceless columns inserted at `inserted_at` (ascending positions in
    /// the new width).
    pub fn widened(&self, inserted_at: &[usize], new_width: usize) -> Self {
        let Some(owner) = self.owner() else {
            return self.clone();
        };
        let mut source_column = 0;
        let fields = (0..new_width)
            .map(|column| {
                if inserted_at.contains(&column) {
                    FieldOrigin::None
                } else {
                    source_column += 1;
                    self.field(source_column - 1)
                }
            })
            .collect();
        Self::Fields { owner, fields }
    }

    /// `Fields` whose every field is the same column of its owner collapses
    /// back to `Record`.
    fn simplified(self) -> Self {
        match self {
            Self::Fields { owner, fields }
                if fields.iter().enumerate().all(|(column, field)| {
                    *field
                        == FieldOrigin::Source {
                            record: owner,
                            field: column,
                        }
                }) =>
            {
                Self::Record(owner)
            }
            other => other,
        }
    }
}

impl From<Option<usize>> for LineOrigin {
    fn from(record: Option<usize>) -> Self {
        Self::from_record(record)
    }
}

impl From<RecordId> for LineOrigin {
    fn from(record: RecordId) -> Self {
        Self::Record(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(n: usize) -> RecordId {
        RecordId::new(n).unwrap()
    }

    #[test]
    fn merged_lines_name_each_field_source() {
        let left = LineOrigin::Record(record(4));
        let right = LineOrigin::Record(record(5));
        let merged = left.merged(&right, |column| column == 1, 2);
        assert_eq!(merged.owner(), Some(record(4)));
        assert_eq!(
            merged.field(1),
            FieldOrigin::Source {
                record: record(5),
                field: 1
            }
        );
    }

    #[test]
    fn merging_without_contribution_stays_whole_record() {
        let left = LineOrigin::Record(record(4));
        let right = LineOrigin::Record(record(5));
        assert_eq!(left.merged(&right, |_| false, 3), left);
    }

    #[test]
    fn projection_and_widening_compose() {
        let origin = LineOrigin::Record(record(7));
        let projected = origin.projected(&[0, 2]);
        assert_eq!(
            projected.field(1),
            FieldOrigin::Source {
                record: record(7),
                field: 2
            }
        );
        let widened = projected.widened(&[1], 3);
        assert_eq!(widened.field(1), FieldOrigin::None);
        assert_eq!(
            widened.field(2),
            FieldOrigin::Source {
                record: record(7),
                field: 2
            }
        );
    }

    #[test]
    fn relocation_moves_one_field() {
        let relocated = LineOrigin::Record(record(3)).relocated(2, 0, 3);
        assert_eq!(
            relocated.field(0),
            FieldOrigin::Source {
                record: record(3),
                field: 2
            }
        );
        assert_eq!(relocated.field(2), FieldOrigin::None);
    }

    #[test]
    fn record_zero_is_generated_and_none_untracked() {
        assert_eq!(LineOrigin::from(Some(0)), LineOrigin::Generated);
        assert_eq!(LineOrigin::from(None), LineOrigin::Untracked);
    }
}
