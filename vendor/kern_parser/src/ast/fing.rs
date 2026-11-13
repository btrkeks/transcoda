use std::fmt;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FingContent {
    /// `.` null data token.
    Null,
    /// One space-separated slot per note of the concurrent `**kern` event, in
    /// source order, such as `1 3 5` or `. 3 .`. `None` is an unfingered note,
    /// written `.`. The parser rejects a token whose slots are all `.`.
    Fingerings(Vec<Option<Fingering>>),
}

impl fmt::Display for FingContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => formatter.write_str("."),
            Self::Fingerings(slots) => {
                for (index, slot) in slots.iter().enumerate() {
                    if index > 0 {
                        formatter.write_str(" ")?;
                    }
                    match slot {
                        Some(fingering) => write!(formatter, "{fingering}")?,
                        None => formatter.write_str(".")?,
                    }
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FingerNumber(u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("invalid finger number {value}; expected 0 through 5")]
pub struct InvalidFingerNumber {
    pub value: u8,
}

impl FingerNumber {
    pub const fn value(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for FingerNumber {
    type Error = InvalidFingerNumber;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if value <= 5 {
            Ok(Self(value))
        } else {
            Err(InvalidFingerNumber { value })
        }
    }
}

impl From<FingerNumber> for u8 {
    fn from(value: FingerNumber) -> Self {
        value.value()
    }
}

impl fmt::Display for FingerNumber {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.value())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fingering {
    Single(FingerNumber),
    Substitution {
        from: FingerNumber,
        to: FingerNumber,
    },
    /// Two or more digits over one note, written `4/3/2`, such as the fingers
    /// of an ornament from its upper to its lower note.
    Stacked {
        first: FingerNumber,
        second: FingerNumber,
        rest: Vec<FingerNumber>,
    },
}

impl Fingering {
    pub const fn single(number: FingerNumber) -> Self {
        Self::Single(number)
    }

    pub const fn substitution(from: FingerNumber, to: FingerNumber) -> Self {
        Self::Substitution { from, to }
    }

    /// One digit is a single fingering and more digits stack; no digits is none.
    pub fn from_digits(digits: &[FingerNumber]) -> Option<Self> {
        match digits {
            [] => None,
            [number] => Some(Self::Single(*number)),
            [first, second, rest @ ..] => Some(Self::Stacked {
                first: *first,
                second: *second,
                rest: rest.to_vec(),
            }),
        }
    }

    pub const fn primary(&self) -> FingerNumber {
        match self {
            Self::Single(number)
            | Self::Substitution { from: number, .. }
            | Self::Stacked { first: number, .. } => *number,
        }
    }

    pub const fn substitution_target(&self) -> Option<FingerNumber> {
        match self {
            Self::Substitution { to, .. } => Some(*to),
            Self::Single(_) | Self::Stacked { .. } => None,
        }
    }
}

impl fmt::Display for Fingering {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Single(number) => write!(formatter, "{number}"),
            Self::Substitution { from, to } => write!(formatter, "{from}-{to}"),
            Self::Stacked {
                first,
                second,
                rest,
            } => {
                write!(formatter, "{first}/{second}")?;
                for number in rest {
                    write!(formatter, "/{number}")?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finger_numbers_are_checked_and_fingerings_have_canonical_spelling() {
        let two = FingerNumber::try_from(2).unwrap();
        let one = FingerNumber::try_from(1).unwrap();

        assert_eq!(two.value(), 2);
        assert_eq!(Fingering::single(two).to_string(), "2");
        assert_eq!(Fingering::substitution(two, one).to_string(), "2-1");
        let stacked = Fingering::from_digits(&[two, one, FingerNumber::try_from(4).unwrap()]);
        assert_eq!(
            stacked.as_ref().map(ToString::to_string).as_deref(),
            Some("2/1/4")
        );
        assert_eq!(stacked.as_ref().map(Fingering::primary), Some(two));
        assert_eq!(Fingering::from_digits(&[two]), Some(Fingering::single(two)));
        assert_eq!(Fingering::from_digits(&[]), None);
        assert_eq!(
            FingContent::Fingerings(vec![None, Some(Fingering::single(one)), None]).to_string(),
            ". 1 ."
        );
        assert_eq!(
            FingerNumber::try_from(6),
            Err(InvalidFingerNumber { value: 6 })
        );
    }
}
