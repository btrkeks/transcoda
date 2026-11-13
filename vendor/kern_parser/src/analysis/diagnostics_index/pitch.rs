use crate::types::{Accidental, DiatonicStep};
use crate::{Clef, ClefSign, Pitch};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScientificPitch {
    pub pitch: Pitch,
    pub accidental: Option<Accidental>,
}

pub(crate) const STEPS: [DiatonicStep; 7] = [
    DiatonicStep::C,
    DiatonicStep::D,
    DiatonicStep::E,
    DiatonicStep::F,
    DiatonicStep::G,
    DiatonicStep::A,
    DiatonicStep::B,
];
pub(crate) fn step_index(step: DiatonicStep) -> i16 {
    STEPS.iter().position(|s| *s == step).unwrap() as i16
}
pub(crate) fn letter(step: DiatonicStep) -> char {
    ['C', 'D', 'E', 'F', 'G', 'A', 'B'][step_index(step) as usize]
}
pub(crate) fn alteration(accidental: Option<&Accidental>) -> i8 {
    match accidental {
        Some(Accidental::DoubleFlat) => -2,
        Some(Accidental::Flat) => -1,
        Some(Accidental::Sharp) => 1,
        Some(Accidental::DoubleSharp) => 2,
        _ => 0,
    }
}

impl ScientificPitch {
    pub fn spn(&self) -> String {
        format!(
            "{}{}{}",
            letter(self.pitch.step),
            match alteration(self.accidental.as_ref()) {
                -2 => "bb",
                -1 => "b",
                1 => "#",
                2 => "##",
                _ => "",
            },
            self.pitch.octave
        )
    }

    pub(crate) fn shifted(&self, steps: i16) -> Option<Self> {
        let index = i16::from(self.pitch.octave) * 7 + step_index(self.pitch.step) + steps;
        let pitch = Pitch {
            step: STEPS[index.rem_euclid(7) as usize],
            octave: i8::try_from(index.div_euclid(7)).ok()?,
        };
        pitch.is_supported_kern_pitch().then(|| Self {
            pitch,
            accidental: self.accidental.clone(),
        })
    }
}
pub(crate) fn staff_position(pitch: Pitch, clef: &Clef) -> Option<i16> {
    let reference = match clef.sign {
        ClefSign::G => 4 * 7 + 4,
        ClefSign::F => 3 * 7 + 3,
        ClefSign::C => 4 * 7,
        _ => return None,
    } + i16::from(clef.octave_shift) * 7;
    let middle = reference + (3 - i16::from(clef.line)) * 2;
    Some(i16::from(pitch.octave) * 7 + step_index(pitch.step) - middle)
}
