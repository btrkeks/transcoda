use super::{document::SystemBreak, note::FermataPlacement};

/// A barline record with either a shared appearance or authored per-spine values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Barline {
    appearances: BarlineAppearances,
    /// Global layout break that must serialize immediately before this boundary.
    pub system_break_before: Option<SystemBreak>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BarlineAppearances {
    Uniform(SpineBarline),
    PerSpine(Vec<SpineBarline>),
}

impl Barline {
    pub fn new(styles: Vec<BarlineStyle>, repeat: RepeatStyle) -> Self {
        Self::from_spine_barlines(vec![SpineBarline::new(styles, repeat)])
    }

    pub(crate) fn from_spine_barlines(spine_barlines: Vec<SpineBarline>) -> Self {
        assert!(
            !spine_barlines.is_empty(),
            "a barline record must contain at least one token"
        );
        let first = &spine_barlines[0];
        let appearances = if spine_barlines.iter().all(|barline| barline == first) {
            BarlineAppearances::Uniform(first.clone())
        } else {
            BarlineAppearances::PerSpine(spine_barlines)
        };
        Self {
            appearances,
            system_break_before: None,
        }
    }

    /// Strongest visible style across the record, or invisible if every field is invisible.
    pub fn styles(&self) -> &[BarlineStyle] {
        match &self.appearances {
            BarlineAppearances::Uniform(barline) => &barline.styles,
            BarlineAppearances::PerSpine(barlines) => aggregate_styles(barlines),
        }
    }

    /// Combined repeat faces across the record, for playback and structural analysis.
    pub fn repeat(&self) -> RepeatStyle {
        match &self.appearances {
            BarlineAppearances::Uniform(barline) => barline.repeat,
            BarlineAppearances::PerSpine(barlines) => aggregate_repeat(barlines),
        }
    }

    pub fn is_spine_specific(&self) -> bool {
        matches!(self.appearances, BarlineAppearances::PerSpine(_))
    }

    /// Uniform records take their width from the active spine topology.
    pub fn spine_count(&self) -> Option<usize> {
        self.spine_barlines().map(<[SpineBarline]>::len)
    }

    pub fn spine_barlines(&self) -> Option<&[SpineBarline]> {
        match &self.appearances {
            BarlineAppearances::Uniform(_) => None,
            BarlineAppearances::PerSpine(barlines) => Some(barlines),
        }
    }

    pub fn appearance_for_spine(&self, spine_index: usize) -> &SpineBarline {
        match &self.appearances {
            BarlineAppearances::Uniform(barline) => barline,
            BarlineAppearances::PerSpine(barlines) => &barlines[spine_index],
        }
    }

    pub fn styles_for_spine(&self, spine_index: usize) -> &[BarlineStyle] {
        &self.appearance_for_spine(spine_index).styles
    }

    pub fn repeat_for_spine(&self, spine_index: usize) -> RepeatStyle {
        self.appearance_for_spine(spine_index).repeat
    }

    pub fn fermata_for_spine(&self, spine_index: usize) -> Option<BarlineFermata> {
        self.appearance_for_spine(spine_index).fermata
    }

    pub fn has_fermata(&self) -> bool {
        self.appearances()
            .iter()
            .any(|barline| barline.fermata.is_some())
    }

    /// Whether this boundary can cross split/merge operations without changing
    /// the placement or number of per-spine symbols.
    pub fn is_topology_invariant(&self) -> bool {
        !self.is_spine_specific() && !self.has_fermata()
    }

    pub fn is_invisible_for_spine(&self, spine_index: usize) -> bool {
        self.styles_for_spine(spine_index)
            .contains(&BarlineStyle::Invisible)
    }

    /// Select and reorder authored spine appearances.
    pub(crate) fn projected_to_spines(&self, source_indices: &[usize]) -> Option<Self> {
        let Some(barlines) = self.spine_barlines() else {
            return Some(self.clone());
        };
        let projected = source_indices
            .iter()
            .map(|index| barlines.get(*index).cloned())
            .collect::<Option<Vec<_>>>()?;
        if projected.is_empty() {
            return None;
        }
        let mut barline = Self::from_spine_barlines(projected);
        barline.system_break_before = self.system_break_before.clone();
        Some(barline)
    }

    /// At a page cut the left face belongs to the previous excerpt.
    pub(crate) fn right_face(&self) -> Self {
        let mut barline = self.clone();
        barline.system_break_before = None;
        for appearance in barline.appearances_mut() {
            appearance.repeat = match appearance.repeat {
                RepeatStyle::Start | RepeatStyle::StartAndEnd => RepeatStyle::Start,
                _ => RepeatStyle::None,
            };
            appearance.styles = vec![BarlineStyle::Basic];
            appearance.fermata = None;
        }
        barline
    }

    pub(crate) fn doubleize_basic(&mut self) {
        if self.repeat() != RepeatStyle::None || self.styles() != [BarlineStyle::Basic] {
            return;
        }
        for barline in self.appearances_mut() {
            if barline.repeat == RepeatStyle::None && barline.styles == [BarlineStyle::Basic] {
                barline.styles = vec![BarlineStyle::Double];
            }
        }
    }

    fn appearances(&self) -> &[SpineBarline] {
        match &self.appearances {
            BarlineAppearances::Uniform(barline) => std::slice::from_ref(barline),
            BarlineAppearances::PerSpine(barlines) => barlines,
        }
    }

    fn appearances_mut(&mut self) -> &mut [SpineBarline] {
        match &mut self.appearances {
            BarlineAppearances::Uniform(barline) => std::slice::from_mut(barline),
            BarlineAppearances::PerSpine(barlines) => barlines,
        }
    }
}

/// Appearance contributed by one field of a Humdrum barline record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpineBarline {
    pub styles: Vec<BarlineStyle>,
    pub repeat: RepeatStyle,
    pub fermata: Option<BarlineFermata>,
}

impl SpineBarline {
    pub fn new(styles: Vec<BarlineStyle>, repeat: RepeatStyle) -> Self {
        Self {
            styles,
            repeat,
            fermata: None,
        }
    }
}

impl From<&Barline> for SpineBarline {
    fn from(barline: &Barline) -> Self {
        barline.appearance_for_spine(0).clone()
    }
}

/// A pause at a measure boundary, rather than on a note or rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarlineFermata {
    /// `;`, a single fermata in the automatic position.
    Single,
    /// `;>` or `;<`, a single fermata forced above or below the staff.
    PlacedSingle(FermataPlacement),
    /// `;;`, a pair above and below the staff.
    Pair,
}

fn aggregate_styles(spine_barlines: &[SpineBarline]) -> &'static [BarlineStyle] {
    if spine_barlines
        .iter()
        .all(|barline| barline.styles.contains(&BarlineStyle::Invisible))
    {
        return &[BarlineStyle::Invisible];
    }
    if spine_barlines
        .iter()
        .any(|barline| barline.styles.contains(&BarlineStyle::Double))
    {
        return &[BarlineStyle::Double];
    }
    if spine_barlines
        .iter()
        .any(|barline| barline.styles.contains(&BarlineStyle::Thin))
    {
        return &[BarlineStyle::Thin];
    }
    if spine_barlines
        .iter()
        .any(|barline| barline.styles.contains(&BarlineStyle::Basic))
    {
        return &[BarlineStyle::Basic];
    }
    if spine_barlines
        .iter()
        .any(|barline| barline.styles.contains(&BarlineStyle::Dashed))
    {
        return &[BarlineStyle::Dashed];
    }
    &[BarlineStyle::Basic]
}

fn aggregate_repeat(spine_barlines: &[SpineBarline]) -> RepeatStyle {
    let starts = spine_barlines.iter().any(|barline| {
        matches!(
            barline.repeat,
            RepeatStyle::Start | RepeatStyle::StartAndEnd
        )
    });
    let ends = spine_barlines
        .iter()
        .any(|barline| matches!(barline.repeat, RepeatStyle::End | RepeatStyle::StartAndEnd));
    match (starts, ends) {
        (true, true) => RepeatStyle::StartAndEnd,
        (true, false) => RepeatStyle::Start,
        (false, true) => RepeatStyle::End,
        (false, false) => RepeatStyle::None,
    }
}

/// Visual or semantic barline style marker.
///
/// Examples: `==` is double, `!` is heavy, `-` is invisible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarlineStyle {
    /// `=`
    Basic,
    /// `==`, `==||`, `==:|!`, `=:|!`
    Double,
    /// =||
    Thin,
    /// `=-`
    Invisible,
    /// `=:`, a bare colon without a repeat bar
    Dashed,
}

/// Repeat direction encoded on a barline.
///
/// Examples: `=|:` starts a repeat, `=:|` ends one, `=:|!|:` does both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatStyle {
    None,
    Start,
    End,
    StartAndEnd,
}
