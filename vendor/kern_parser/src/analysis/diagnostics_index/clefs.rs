use super::{pitch, StaffIdentity, StaffMap};
use crate::{Clef, ClefSign, Pitch, SpineType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClefLetter {
    G,
    F,
    C,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClefSpec {
    sign: ClefLetter,
    line: u8,
    octave_shift: i8,
}
impl ClefSpec {
    /// Ignores `invisible`; TAB, percussion and lines outside 1-5 are unsupported.
    pub fn of(clef: &Clef) -> Option<Self> {
        let sign = match clef.sign {
            ClefSign::G => ClefLetter::G,
            ClefSign::F => ClefLetter::F,
            ClefSign::C => ClefLetter::C,
            ClefSign::Tab | ClefSign::Percussion => return None,
        };
        (1..=5).contains(&clef.line).then_some(Self {
            sign,
            line: clef.line,
            octave_shift: clef.octave_shift,
        })
    }

    pub fn staff_position(self, pitch: Pitch) -> Option<i16> {
        let clef = Clef {
            sign: match self.sign {
                ClefLetter::G => ClefSign::G,
                ClefLetter::F => ClefSign::F,
                ClefLetter::C => ClefSign::C,
            },
            line: self.line,
            octave_shift: self.octave_shift,
            invisible: false,
        };
        pitch::staff_position(pitch, &clef)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaffClef {
    Known(ClefSpec),
    Unsupported,
    /// The staff's track has no clef in force.
    Undeclared,
    /// The staff has no kern column on this record.
    Absent,
}
impl StaffClef {
    pub fn known(self) -> Option<ClefSpec> {
        match self {
            Self::Known(spec) => Some(spec),
            _ => None,
        }
    }
}
impl StaffMap {
    /// The clef of the staff's track: one value for all of its subspines,
    /// whichever subspine wrote it (D5).
    pub fn staff_clef(&self, line_index: usize, staff: u32) -> StaffClef {
        let Some(column) = self
            .line(line_index)
            .iter()
            .find(|c| c.spine_type == SpineType::Kern && c.staff == StaffIdentity::Known(staff))
        else {
            return StaffClef::Absent;
        };
        match &column.clef {
            Some(clef) => ClefSpec::of(clef).map_or(StaffClef::Unsupported, StaffClef::Known),
            None => StaffClef::Undeclared,
        }
    }
}
