//! Line classification and token parsing for **kern files.

use crate::duration::RhythmDuration;
use crate::error::ParseError;

/// Information about repeat symbols in a barline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RepeatInfo {
    /// Contains end repeat marker (e.g., `:|`, `:|!`, `:||`)
    pub is_end_repeat: bool,
    /// Contains start repeat marker (e.g., `|:`, `!|:`, `||:`)
    pub is_start_repeat: bool,
}

impl RepeatInfo {
    /// Returns true if this is a segue (both end and start repeat).
    pub fn is_segue(&self) -> bool {
        self.is_end_repeat && self.is_start_repeat
    }
}

/// Parse repeat information from a barline token.
///
/// Detects:
/// - End repeat: `:|`, `:|!`, `:!|`, `:||`
/// - Start repeat: `|:`, `!|:`, `||:`
/// - Segue (both): `:|!|:`, `:!|:`, `:|:`
fn parse_repeat_info(token: &str) -> RepeatInfo {
    // End repeat patterns: colon followed by bar-like characters
    // We look for `:` followed by `|`, `!`, or at end
    let is_end_repeat = token.contains(":|") || token.contains(":!");

    // Start repeat patterns: bar-like characters followed by colon
    // We look for `|` or `!` followed by `:`
    let is_start_repeat = token.contains("|:") || token.contains("!:");

    RepeatInfo {
        is_end_repeat,
        is_start_repeat,
    }
}

/// Classification of a line in a **kern file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineType {
    /// Empty line or comment (starts with !)
    Empty,
    /// Exclusive interpretation (e.g., **kern, **dynam)
    ExclusiveInterpretation,
    /// Tandem interpretation (e.g., *M4/4, *clefG2, *^, *v)
    TandemInterpretation,
    /// Barline (starts with =)
    Barline {
        is_double: bool,        // `==` semantic double barline
        is_visual_double: bool, // `=||` visual double barline
        measure_num: Option<u32>,
        repeat: RepeatInfo,
    },
    /// Data line containing notes, rests, or null tokens
    Data,
    /// Spine terminator (*-)
    SpineTerminator,
}

impl LineType {
    /// Classify a line from a **kern file.
    pub fn classify(line: &str) -> Self {
        let line = line.trim();

        if line.is_empty() || line.starts_with('!') {
            return LineType::Empty;
        }

        // Get first token
        let first_token = line.split('\t').next().unwrap_or("");

        if first_token.starts_with("**") {
            LineType::ExclusiveInterpretation
        } else if first_token == "*-" {
            LineType::SpineTerminator
        } else if first_token.starts_with('*') {
            LineType::TandemInterpretation
        } else if first_token.starts_with('=') {
            let is_double = first_token.contains("==");
            let is_visual_double = first_token.contains("||");
            let measure_num = parse_measure_number(first_token);
            let repeat = parse_repeat_info(first_token);
            LineType::Barline {
                is_double,
                is_visual_double,
                measure_num,
                repeat,
            }
        } else {
            LineType::Data
        }
    }
}

/// Parse a measure number from a barline token (e.g., "=5" -> Some(5)).
fn parse_measure_number(token: &str) -> Option<u32> {
    // Strip leading = signs
    let rest = token.trim_start_matches('=');
    // Try to parse the number (may have trailing characters like :|)
    rest.chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

/// Information about a parsed token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenInfo {
    /// A note or rest with a duration.
    Duration(RhythmDuration),
    /// A grace note (zero duration).
    GraceNote,
    /// A null token (period).
    Null,
    /// A chord (multiple notes) - all notes have the same duration.
    Chord(RhythmDuration),
    /// Unparseable token.
    Unknown(String),
}

/// Parse a single **kern token and extract its rhythmic information.
///
/// # Token Types
/// - Notes: `4c`, `8d#`, `16.e-`
/// - Rests: `4r`, `8r`
/// - Grace notes: `8qc`, `16qqd` (zero duration)
/// - Chords: `4c 4e 4g` (space-separated, count once)
/// - Null: `.`
pub fn parse_token(token: &str) -> TokenInfo {
    let token = token.trim();

    // Null token
    if token == "." {
        return TokenInfo::Null;
    }

    // Check for chord (space-separated notes)
    if token.contains(' ') {
        let parts: Vec<&str> = token.split(' ').collect();
        // Parse the first note to get the duration (all should be same)
        if let Some(first) = parts.first() {
            match parse_single_token(first) {
                TokenInfo::Duration(d) => return TokenInfo::Chord(d),
                TokenInfo::GraceNote => return TokenInfo::GraceNote,
                other => return other,
            }
        }
    }

    parse_single_token(token)
}

/// Parse a single (non-chord) **kern token.
fn parse_single_token(token: &str) -> TokenInfo {
    let token = token.trim();

    if token == "." {
        return TokenInfo::Null;
    }

    // Check for grace notes (contain 'q' or 'Q')
    if token.contains('q') || token.contains('Q') {
        return TokenInfo::GraceNote;
    }

    // Try to extract duration
    match extract_duration(token) {
        Ok(duration) => TokenInfo::Duration(duration),
        Err(_) => TokenInfo::Unknown(token.to_string()),
    }
}

/// Extract the rhythmic duration from a **kern token.
///
/// The token format is: [modifiers]reciprocal[dots][pitch][modifiers]
/// Or for rational durations: [modifiers]numerator%denominator[dots][pitch][modifiers]
/// Examples: 4c, 8.d, 16..e-, [4f, 4g], (8a, 3%5c, 3%5.ryy
pub fn extract_duration(token: &str) -> Result<RhythmDuration, ParseError> {
    // Strip leading modifiers: ties [, ], slurs (, ), etc.
    let token = token.trim_start_matches(|c| matches!(c, '[' | ']' | '(' | ')' | '{' | '}' | '_'));

    // Find the numerator/reciprocal number
    let mut numerator_str = String::new();
    let mut dots = 0u32;
    let mut chars = token.chars().peekable();

    // Parse the first number (numerator for rational, reciprocal otherwise)
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            numerator_str.push(c);
            chars.next();
        } else {
            break;
        }
    }

    if numerator_str.is_empty() {
        return Err(ParseError::InvalidDuration(format!(
            "no duration found: {}",
            token
        )));
    }

    // Check for rational duration (% separator)
    if chars.peek() == Some(&'%') {
        chars.next(); // consume '%'

        // Parse denominator
        let mut denominator_str = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                denominator_str.push(c);
                chars.next();
            } else {
                break;
            }
        }

        if denominator_str.is_empty() {
            return Err(ParseError::InvalidDuration(format!(
                "missing denominator in rational duration: {}",
                token
            )));
        }

        // Count dots after the denominator
        for c in chars {
            if c == '.' {
                dots += 1;
            }
            if c.is_ascii_alphabetic() && !matches!(c, 'q' | 'Q') {
                break;
            }
        }

        let numerator: u32 = numerator_str.parse().map_err(|_| {
            ParseError::InvalidDuration(format!("invalid numerator: {}", numerator_str))
        })?;

        let denominator: u32 = denominator_str.parse().map_err(|_| {
            ParseError::InvalidDuration(format!("invalid denominator: {}", denominator_str))
        })?;

        if denominator == 0 {
            return Err(ParseError::InvalidDuration(format!(
                "zero denominator in rational duration: {}",
                token
            )));
        }

        return Ok(RhythmDuration::from_rational(numerator, denominator, dots));
    }

    // Standard reciprocal notation
    // Count dots (can appear anywhere after the reciprocal, but typically right after)
    for c in chars {
        if c == '.' {
            dots += 1;
        }
        // Stop at first pitch letter (we don't need to parse further)
        if c.is_ascii_alphabetic() && !matches!(c, 'q' | 'Q') {
            break;
        }
    }

    let reciprocal: u32 = numerator_str.parse().map_err(|_| {
        ParseError::InvalidDuration(format!("invalid reciprocal: {}", numerator_str))
    })?;

    // Handle breve (0 = double whole note in **kern)
    if reciprocal == 0 {
        return Ok(RhythmDuration::breve(dots));
    }

    Ok(RhythmDuration::from_reciprocal(reciprocal, dots))
}

/// Returns true if this token represents a rest.
pub fn is_rest(token: &str) -> bool {
    token.contains('r')
}

/// Split a line into tokens (tab-separated).
pub fn split_tokens(line: &str) -> Vec<&str> {
    line.split('\t').collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_type_classification() {
        assert_eq!(LineType::classify(""), LineType::Empty);
        assert_eq!(LineType::classify("! comment"), LineType::Empty);
        assert_eq!(
            LineType::classify("**kern\t**dynam"),
            LineType::ExclusiveInterpretation
        );
        assert_eq!(
            LineType::classify("*M4/4\t*"),
            LineType::TandemInterpretation
        );
        assert_eq!(LineType::classify("*^\t*"), LineType::TandemInterpretation);
        assert_eq!(
            LineType::classify("=1"),
            LineType::Barline {
                is_double: false,
                is_visual_double: false,
                measure_num: Some(1),
                repeat: RepeatInfo::default(),
            }
        );
        assert_eq!(
            LineType::classify("=="),
            LineType::Barline {
                is_double: true,
                is_visual_double: false,
                measure_num: None,
                repeat: RepeatInfo::default(),
            }
        );
        assert_eq!(
            LineType::classify("=5:|"),
            LineType::Barline {
                is_double: false,
                is_visual_double: false,
                measure_num: Some(5),
                repeat: RepeatInfo {
                    is_end_repeat: true,
                    is_start_repeat: false
                },
            }
        );
        assert_eq!(LineType::classify("4c\t8d"), LineType::Data);
        assert_eq!(LineType::classify("*-\t*-"), LineType::SpineTerminator);
    }

    #[test]
    fn test_parse_basic_tokens() {
        // Quarter note C
        assert_eq!(
            parse_token("4c"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(4, 0))
        );
        // Eighth note D
        assert_eq!(
            parse_token("8d"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(8, 0))
        );
        // Dotted quarter
        assert_eq!(
            parse_token("4.e"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(4, 1))
        );
        // Double-dotted half
        assert_eq!(
            parse_token("2..f"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(2, 2))
        );
    }

    #[test]
    fn test_parse_rests() {
        assert_eq!(
            parse_token("4r"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(4, 0))
        );
        assert_eq!(
            parse_token("8r"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(8, 0))
        );
    }

    #[test]
    fn test_parse_grace_notes() {
        assert_eq!(parse_token("8qc"), TokenInfo::GraceNote);
        assert_eq!(parse_token("16qqd"), TokenInfo::GraceNote);
        assert_eq!(parse_token("8Qe"), TokenInfo::GraceNote);
    }

    #[test]
    fn test_parse_null_token() {
        assert_eq!(parse_token("."), TokenInfo::Null);
    }

    #[test]
    fn test_parse_chord() {
        // Chord: all notes have same duration, count once
        assert_eq!(
            parse_token("4c 4e 4g"),
            TokenInfo::Chord(RhythmDuration::from_reciprocal(4, 0))
        );
    }

    #[test]
    fn test_parse_with_modifiers() {
        // Tied note
        assert_eq!(
            parse_token("[4c"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(4, 0))
        );
        assert_eq!(
            parse_token("4c]"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(4, 0))
        );
        // Slurred note
        assert_eq!(
            parse_token("(8d"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(8, 0))
        );
        assert_eq!(
            parse_token("8d)"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(8, 0))
        );
    }

    #[test]
    fn test_parse_with_accidentals() {
        assert_eq!(
            parse_token("4c#"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(4, 0))
        );
        assert_eq!(
            parse_token("4e-"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(4, 0))
        );
        assert_eq!(
            parse_token("4fn"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(4, 0))
        );
    }

    #[test]
    fn test_parse_triplets() {
        // Triplet eighth
        assert_eq!(
            parse_token("12c"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(12, 0))
        );
        // Triplet quarter
        assert_eq!(
            parse_token("6d"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(6, 0))
        );
    }

    #[test]
    fn test_parse_breve() {
        // Breve (double whole note)
        assert_eq!(
            parse_token("0c"),
            TokenInfo::Duration(RhythmDuration::breve(0))
        );
    }

    #[test]
    fn test_is_rest() {
        assert!(is_rest("4r"));
        assert!(is_rest("8rr")); // double rest in some notation
        assert!(!is_rest("4c"));
        assert!(!is_rest("."));
    }

    #[test]
    fn test_split_tokens() {
        let tokens = split_tokens("4c\t8d\t.");
        assert_eq!(tokens, vec!["4c", "8d", "."]);
    }

    #[test]
    fn test_parse_rational_duration() {
        // Basic rational duration: 3/5 of a whole note
        assert_eq!(
            parse_token("3%5c"),
            TokenInfo::Duration(RhythmDuration::new(3, 5))
        );
        // Dotted rational: 3/5 * 3/2 = 9/10
        assert_eq!(
            parse_token("3%5.c"),
            TokenInfo::Duration(RhythmDuration::new(9, 10))
        );
        // Rest with rational duration
        assert_eq!(
            parse_token("3%5.ryy"),
            TokenInfo::Duration(RhythmDuration::new(9, 10))
        );
    }

    #[test]
    fn test_parse_rational_with_modifiers() {
        // Tied rational note
        assert_eq!(
            parse_token("[3%5c"),
            TokenInfo::Duration(RhythmDuration::new(3, 5))
        );
        // Slurred rational note
        assert_eq!(
            parse_token("(3%5d"),
            TokenInfo::Duration(RhythmDuration::new(3, 5))
        );
    }

    #[test]
    fn test_parse_rational_chord() {
        // Chord with rational duration
        assert_eq!(
            parse_token("3%5c 3%5e"),
            TokenInfo::Chord(RhythmDuration::new(3, 5))
        );
    }

    #[test]
    fn test_parse_rational_errors() {
        // Missing denominator
        assert!(matches!(parse_token("3%c"), TokenInfo::Unknown(_)));
        // Missing numerator (starts with %)
        assert!(matches!(parse_token("%5c"), TokenInfo::Unknown(_)));
        // Zero denominator
        assert!(matches!(parse_token("3%0c"), TokenInfo::Unknown(_)));
    }

    #[test]
    fn test_rational_equivalence() {
        // 1/4 via rational should equal quarter note
        assert_eq!(
            parse_token("1%4c"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(4, 0))
        );
        // 1/8 via rational should equal eighth note
        assert_eq!(
            parse_token("1%8d"),
            TokenInfo::Duration(RhythmDuration::from_reciprocal(8, 0))
        );
    }

    #[test]
    fn test_measure_number_parsing() {
        assert_eq!(
            LineType::classify("=1"),
            LineType::Barline {
                is_double: false,
                is_visual_double: false,
                measure_num: Some(1),
                repeat: RepeatInfo::default(),
            }
        );
        assert_eq!(
            LineType::classify("=23"),
            LineType::Barline {
                is_double: false,
                is_visual_double: false,
                measure_num: Some(23),
                repeat: RepeatInfo::default(),
            }
        );
        assert_eq!(
            LineType::classify("=:|"),
            LineType::Barline {
                is_double: false,
                is_visual_double: false,
                measure_num: None,
                repeat: RepeatInfo {
                    is_end_repeat: true,
                    is_start_repeat: false
                },
            }
        );
    }

    #[test]
    fn test_parse_end_repeat() {
        // =5:| detects end repeat
        let LineType::Barline { repeat, .. } = LineType::classify("=5:|") else {
            panic!()
        };
        assert!(repeat.is_end_repeat);
        assert!(!repeat.is_start_repeat);

        // :|| also end repeat
        let LineType::Barline { repeat, .. } = LineType::classify("=:||") else {
            panic!()
        };
        assert!(repeat.is_end_repeat);

        // :|! also end repeat
        let LineType::Barline { repeat, .. } = LineType::classify("=:|!") else {
            panic!()
        };
        assert!(repeat.is_end_repeat);
    }

    #[test]
    fn test_parse_start_repeat() {
        // =|: detects start repeat
        let LineType::Barline { repeat, .. } = LineType::classify("=|:") else {
            panic!()
        };
        assert!(repeat.is_start_repeat);
        assert!(!repeat.is_end_repeat);

        // ||: also start repeat
        let LineType::Barline { repeat, .. } = LineType::classify("=||:") else {
            panic!()
        };
        assert!(repeat.is_start_repeat);

        // !|: also start repeat
        let LineType::Barline { repeat, .. } = LineType::classify("=!|:") else {
            panic!()
        };
        assert!(repeat.is_start_repeat);
    }

    #[test]
    fn test_parse_segue() {
        // =:|!|: detects both (segue)
        let LineType::Barline { repeat, .. } = LineType::classify("=:|!|:") else {
            panic!()
        };
        assert!(repeat.is_end_repeat);
        assert!(repeat.is_start_repeat);
        assert!(repeat.is_segue());

        // :|: also segue
        let LineType::Barline { repeat, .. } = LineType::classify("=:|:") else {
            panic!()
        };
        assert!(repeat.is_segue());
    }

    #[test]
    fn test_double_barline_not_repeat() {
        // == is not a repeat
        let LineType::Barline {
            repeat, is_double, ..
        } = LineType::classify("==")
        else {
            panic!()
        };
        assert!(is_double);
        assert!(!repeat.is_end_repeat);
        assert!(!repeat.is_start_repeat);
    }
}
