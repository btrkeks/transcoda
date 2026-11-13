//! One spelling for the spine splits and merges Verovio draws alike.
//!
//! Verovio gives each token the layer of its column position within its staff
//! on that line, and a staff's layer count is the most columns it has on any
//! data line of the measure. Where a `*^` or `*v` sits inside a measure is
//! invisible as long as every token keeps its position and the layer count
//! stays; moving a split while keeping spine identity renumbers layers. So
//! each measure is written on a positional grid: a staff keeps its layer count
//! of columns for the whole measure, splits follow the opening barline,
//! merges precede the closing one, and a column a written line lacks holds a
//! null while its last token sounds and an invisible rest otherwise.
//!
//! That rest must be what iohumdrum already reads into the gap. iohumdrum
//! fills a layer's gaps before, between and after its tokens with spaces of
//! its own spelling (`getTimingInformation`, `setDuration`), and several of
//! its analyses read the layer's own token list or the line rather than the
//! measure. A measure is left as written when the grid would change any of
//! them:
//!
//! - a gap iohumdrum spells as more than one space, or that compaction would
//!   join with a written invisible rest: spacing follows the spaces;
//! - a filled column holding only rests: a lone rest becomes a measure rest
//!   (`hasFullMeasureRest`), and rests alone are silent layers;
//! - a filled column a beam enters or leaves, or a staff with a beam open
//!   across either barline: iohumdrum writes a straddle space, and beams follow
//!   spine paths across barlines;
//! - a clef after a gap in its staff: iohumdrum copies the clef into the
//!   absent layer at the time that layer has reached;
//! - a cross-staff note in a measure with a widened line: humlib assigns
//!   cross-staff stems from each line's sub-spine count
//!   (`prepareStaffAboveNoteStems`);
//! - written invisible rests the removed manipulators kept apart, which
//!   compaction would join;
//! - a token landing on a column whose last token still sounds, which happens
//!   when a split left of a sounding voice moves it;
//! - a clef or other emitted tandem inside a split region, or an ottava end on
//!   a widened line.
//!
//! An ottava start goes into every column of its staff, which draws alike
//! because iohumdrum opens a staff's ottava once, unless the ottava also
//! closes in the measure. An emitted tandem at the edge of a measure, where
//! every staff has one column, stays outside the grid and holds the barline
//! at its written width.

use std::{collections::BTreeMap, ops::Range};

use crate::{
    analysis::{data_line_durations, measure_spans},
    measure::MusicalDuration,
    normalization::{tandem_omitted_by, NormalizationPass},
    serialize::TandemPolicy,
    types::{ChordBeams, DurationValue},
    Duration, KernContent, KernDocument, KernLine, LocalComment, Ottava, OttavaKind, Rest,
    SpineContent, SpineManipulator, SpineManipulatorLine, SpineType, TandemInterpretation,
    TandemInterpretationLine,
};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Write every measure's split and merge layout on its positional grid.
pub struct CanonicalizeMeasureGrid {
    /// The output whose omitted tandems ride along instead of freezing a
    /// measure.
    omitted: TandemPolicy,
}

impl CanonicalizeMeasureGrid {
    pub(crate) fn omitting(omitted: TandemPolicy) -> Self {
        Self { omitted }
    }
}

impl NormalizationPass for CanonicalizeMeasureGrid {
    fn normalize(&self, document: &mut KernDocument) {
        let Ok(durations) = data_line_durations(document) else {
            return;
        };
        let durations = durations
            .into_iter()
            .map(|duration| (duration.line_index, duration.duration))
            .collect::<BTreeMap<_, _>>();
        let Some(page) = Page::plan(document, self.omitted, &durations) else {
            return;
        };

        let rewrites = (0..page.measures.len())
            .filter_map(|index| page.rewrite(document, index))
            .collect::<Vec<_>>();
        for (range, lines) in rewrites.into_iter().rev() {
            document.splice_lines(range, lines);
        }
    }
}

/// Columns per staff, one entry per spine of the exclusive interpretation
/// line, left to right.
type Widths = Vec<usize>;

/// The columns every staff has on every line of the page.
struct Topology {
    staff_types: Vec<SpineType>,
    /// `at[i]` is what line `i` is read at; `at[len]` follows the last line.
    at: Vec<Widths>,
    /// Beams each staff has open before line `i`. Beams follow spine paths
    /// across a barline, so a split or merge the grid moves there would
    /// hand an open beam to another layer.
    open_beams: Vec<Vec<i32>>,
}

/// The lines between two measures, usually one barline.
struct Gap {
    lines: Range<usize>,
    /// Only barlines that take any width, so the grid may choose it.
    free: bool,
}

enum MeasurePlan {
    Grid(Grid),
    Frozen,
}

struct Grid {
    layers: Widths,
    /// Edge lines with an emitted tandem, kept at the barline's written width.
    leading: Range<usize>,
    body: Range<usize>,
    /// The body's lines on the grid, without its manipulator lines.
    gridded: Vec<(KernLine, Option<usize>)>,
    trailing: Range<usize>,
}

struct Measure {
    content: Range<usize>,
    plan: MeasurePlan,
}

/// Measures and the gaps around them: `gaps[k]` precedes `measures[k]`, and
/// the last gap follows the last measure.
struct Page {
    topology: Topology,
    measures: Vec<Measure>,
    gaps: Vec<Gap>,
}

type Rewrite = (Range<usize>, Vec<(KernLine, Option<usize>)>);

impl Page {
    fn plan(
        document: &KernDocument,
        omitted: TandemPolicy,
        durations: &BTreeMap<usize, MusicalDuration>,
    ) -> Option<Self> {
        let topology = Topology::trace(document)?;
        let mut measures = Vec::new();
        let mut gaps = Vec::new();
        let mut previous_end = 0;
        for span in measure_spans(document) {
            gaps.push(Gap::new(document, previous_end..span.content.start));
            measures.push(Measure {
                plan: plan_measure(
                    document,
                    &topology,
                    span.content.clone(),
                    omitted,
                    durations,
                ),
                content: span.content.clone(),
            });
            previous_end = span.content.end;
        }
        gaps.push(Gap::new(document, previous_end..document.lines().len()));
        Some(Self {
            topology,
            measures,
            gaps,
        })
    }

    /// The widths at gap `index`: what the measure before it closes to and
    /// what the measure after it opens from. Two gridded measures meet at the
    /// narrower of their grids; anything else meets at the written widths.
    fn boundary(&self, index: usize) -> (Widths, Widths) {
        let gap = &self.gaps[index];
        let left = index.checked_sub(1).map(|left| &self.measures[left].plan);
        let right = self.measures.get(index).map(|right| &right.plan);
        if let (Some(MeasurePlan::Grid(left)), Some(MeasurePlan::Grid(right))) = (left, right) {
            if gap.free && left.trailing.is_empty() && right.leading.is_empty() {
                let widths = left
                    .layers
                    .iter()
                    .zip(&right.layers)
                    .map(|(left, right)| *left.min(right))
                    .collect::<Widths>();
                return (widths.clone(), widths);
            }
        }
        (
            self.topology.at[gap.lines.start].clone(),
            self.topology.at[gap.lines.end].clone(),
        )
    }

    fn rewrite(&self, document: &KernDocument, index: usize) -> Option<Rewrite> {
        let measure = &self.measures[index];
        let MeasurePlan::Grid(grid) = &measure.plan else {
            return None;
        };
        let (_, entry) = self.boundary(index);
        let (exit, _) = self.boundary(index + 1);
        let lines = grid.render(document, &entry, &exit);
        let unchanged = lines
            .iter()
            .map(|(line, _)| line)
            .eq(&document.lines()[measure.content.clone()]);
        (!unchanged).then(|| (measure.content.clone(), lines))
    }
}

impl Gap {
    fn new(document: &KernDocument, lines: Range<usize>) -> Self {
        let free = !lines.is_empty()
            && document.lines()[lines.clone()].iter().all(
                |line| matches!(line, KernLine::Barline(barline) if barline.is_topology_invariant()),
            );
        Self { lines, free }
    }
}

impl Topology {
    /// `None` when a line's width disagrees with the topology, a merge joins
    /// two staffs, or a spine ends or begins inside the page.
    fn trace(document: &KernDocument) -> Option<Self> {
        let lines = document.lines();
        let (staff_types, first) = match lines.first()? {
            KernLine::SpineTypeLine(types) => (types.clone(), 1),
            line => (vec![SpineType::Kern; line_width(line)?], 0),
        };
        let mut widths = vec![1; staff_types.len()];
        let mut at = vec![widths.clone(); first];
        let mut beams = vec![0; staff_types.len()];
        let mut open_beams = vec![beams.clone(); first];
        let mut terminated = false;
        for line in &lines[first..] {
            if terminated {
                return None;
            }
            at.push(widths.clone());
            open_beams.push(beams.clone());
            let total = widths.iter().sum::<usize>();
            match line {
                KernLine::SpineTypeLine(_) => return None,
                KernLine::SpineManipulatorLine(manipulators) => {
                    if manipulators.len() != total {
                        return None;
                    }
                    let tokens = manipulators.tokens();
                    if tokens
                        .iter()
                        .all(|token| matches!(token, SpineManipulator::Terminate))
                    {
                        terminated = true;
                    } else {
                        widths = apply_manipulators(&widths, tokens)?;
                    }
                }
                KernLine::Barline(barline) => {
                    if barline.spine_count().is_some_and(|count| count != total) {
                        return None;
                    }
                }
                line => {
                    if line_width(line)? != total {
                        return None;
                    }
                    if let KernLine::Data(contents) = line {
                        let staff_of = widths
                            .iter()
                            .enumerate()
                            .flat_map(|(staff, width)| std::iter::repeat_n(staff, *width));
                        for (content, staff) in contents.iter().zip(staff_of) {
                            if let SpineContent::Kern(content) = content {
                                if let Some(marks) = content_beams(content) {
                                    beams[staff] += i32::from(marks.beam_start_count())
                                        - i32::from(marks.beam_end_count());
                                }
                            }
                        }
                    }
                }
            }
        }
        at.push(widths);
        open_beams.push(beams);
        Some(Self {
            staff_types,
            at,
            open_beams,
        })
    }
}

fn line_width(line: &KernLine) -> Option<usize> {
    match line {
        KernLine::SpineTypeLine(types) => Some(types.len()),
        KernLine::TandemInterpretationLine(tandems) => Some(tandems.len()),
        KernLine::SpineManipulatorLine(manipulators) => Some(manipulators.len()),
        KernLine::LocalCommentLine(comments) => Some(comments.len()),
        KernLine::Barline(barline) => barline.spine_count(),
        KernLine::Data(contents) => Some(contents.len()),
    }
}

fn apply_manipulators(widths: &[usize], tokens: &[SpineManipulator]) -> Option<Widths> {
    let staff_of = widths
        .iter()
        .enumerate()
        .flat_map(|(staff, width)| std::iter::repeat_n(staff, *width))
        .collect::<Vec<_>>();
    let mut next = widths.to_vec();
    let mut column = 0;
    while column < tokens.len() {
        match tokens[column] {
            SpineManipulator::Null => column += 1,
            SpineManipulator::Split => {
                next[staff_of[column]] += 1;
                column += 1;
            }
            SpineManipulator::Merge => {
                let start = column;
                while column < tokens.len() && matches!(tokens[column], SpineManipulator::Merge) {
                    column += 1;
                }
                let staff = staff_of[start];
                if staff_of[column - 1] != staff {
                    return None;
                }
                next[staff] -= column - start - 1;
            }
            SpineManipulator::Terminate => return None,
        }
    }
    Some(next)
}

fn plan_measure(
    document: &KernDocument,
    topology: &Topology,
    content: Range<usize>,
    omitted: TandemPolicy,
    durations: &BTreeMap<usize, MusicalDuration>,
) -> MeasurePlan {
    let lines = document.lines();
    let at = |index: usize| &topology.at[index];
    let staffs = topology.staff_types.len();

    if content.clone().any(|index| {
        topology
            .staff_types
            .iter()
            .zip(at(index))
            .any(|(spine_type, width)| *spine_type != SpineType::Kern && *width != 1)
    }) {
        return MeasurePlan::Frozen;
    }

    let mut layers = vec![1; staffs];
    for index in content.clone() {
        if matches!(lines[index], KernLine::Data(_)) {
            for (layer, width) in layers.iter_mut().zip(at(index)) {
                *layer = (*layer).max(*width);
            }
        }
    }

    let crossing_beam = (0..staffs).any(|staff| {
        layers[staff] > 1
            && (topology.open_beams[content.start][staff] != 0
                || topology.open_beams[content.end][staff] != 0)
    });
    if crossing_beam {
        return MeasurePlan::Frozen;
    }

    let is_edge = |index: &usize| {
        matches!(
            lines[*index],
            KernLine::TandemInterpretationLine(_) | KernLine::LocalCommentLine(_)
        )
    };
    let emits = |index: usize| match &lines[index] {
        KernLine::TandemInterpretationLine(tandems) => tandems
            .tokens()
            .iter()
            .any(|tandem| emitted_outside_grid(tandem, omitted)),
        _ => false,
    };
    let leading_end = content.start + content.clone().take_while(is_edge).count();
    let trailing_start = content.end - content.clone().rev().take_while(is_edge).count();
    let leading = if (content.start..leading_end).any(emits) {
        content.start..leading_end
    } else {
        content.start..content.start
    };
    let trailing = if (trailing_start..content.end).any(emits) {
        trailing_start..content.end
    } else {
        content.end..content.end
    };
    let body = leading.end..trailing.start;

    let split = |widths: &[usize]| widths.iter().any(|width| *width > 1);
    for index in content.clone() {
        let in_body = body.contains(&index);
        if emits(index) && (split(at(index)) || (in_body && split(&layers))) {
            return MeasurePlan::Frozen;
        }
        let projects = matches!(
            lines[index],
            KernLine::TandemInterpretationLine(_) | KernLine::LocalCommentLine(_)
        );
        if in_body && projects && at(index).iter().zip(&layers).any(|(w, l)| w > l) {
            return MeasurePlan::Frozen;
        }
    }

    match grid_body(document, topology, &content, &body, &layers, durations) {
        Some(gridded) => MeasurePlan::Grid(Grid {
            layers,
            leading,
            body,
            gridded,
            trailing,
        }),
        None => MeasurePlan::Frozen,
    }
}

/// The body lines on the grid, or `None` when Verovio would draw them
/// differently from the written lines (see the module doc).
fn grid_body(
    document: &KernDocument,
    topology: &Topology,
    content: &Range<usize>,
    body: &Range<usize>,
    layers: &[usize],
    durations: &BTreeMap<usize, MusicalDuration>,
) -> Option<Vec<(KernLine, Option<usize>)>> {
    let mut columns = vec![Column::default(); layers.iter().sum()];
    let ottava_ends = staff_tandems(
        document,
        topology,
        content,
        |tandem| matches!(tandem, TandemInterpretation::Ottava(ottava) if !ottava.active),
    );
    let mut widened = false;
    let mut cross_staff = false;
    let mut gridded = Vec::new();
    for index in body.clone() {
        let widths = &topology.at[index];
        let line = match &document.lines()[index] {
            KernLine::Data(contents) => {
                let duration = durations
                    .get(&index)
                    .copied()
                    .unwrap_or_else(MusicalDuration::zero);
                widened |= widths
                    .iter()
                    .zip(layers)
                    .any(|(width, layers)| width < layers);
                cross_staff |= contents.iter().any(is_cross_staff);
                let row = Row {
                    index,
                    gridded: gridded.len(),
                    duration,
                };
                KernLine::Data(project_data(contents, widths, layers, row, &mut columns)?)
            }
            KernLine::TandemInterpretationLine(tandems) => {
                let closing = |staff: usize| {
                    ottava_ends
                        .iter()
                        .any(|&(line, end_staff)| end_staff == staff && line > index)
                };
                KernLine::TandemInterpretationLine(project_tandems(
                    tandems.tokens(),
                    widths,
                    layers,
                    closing,
                )?)
            }
            KernLine::LocalCommentLine(comments) => {
                KernLine::LocalCommentLine(project(comments, widths, layers, |_| {
                    LocalComment::Null
                }))
            }
            KernLine::SpineManipulatorLine(_) => {
                columns.iter_mut().for_each(|column| column.crossed = true);
                continue;
            }
            KernLine::SpineTypeLine(_) | KernLine::Barline(_) => return None,
        };
        gridded.push((line, document.source(index)));
    }
    if cross_staff && widened {
        return None;
    }

    let staff_of = layers
        .iter()
        .enumerate()
        .flat_map(|(staff, layers)| std::iter::repeat_n(staff, *layers))
        .collect::<Vec<_>>();
    let clefs = staff_tandems(document, topology, content, |tandem| {
        matches!(tandem, TandemInterpretation::Clef(_))
    });
    let mut rests = Vec::new();
    for (column, facts) in columns.iter_mut().enumerate() {
        facts.close_gap();
        let clef_after_fill = facts.first_fill.is_some_and(|fill| {
            clefs
                .iter()
                .any(|&(line, staff)| staff == staff_of[column] && line > fill)
        });
        if !facts.reads_as_written(clef_after_fill) {
            return None;
        }
        rests.extend(facts.runs.drain(..).map(|run| (column, run)));
    }
    for (column, run) in rests {
        for (n, &row) in run.rows.iter().enumerate() {
            let KernLine::Data(cells) = &mut gridded[row].0 else {
                continue;
            };
            cells[column] = SpineContent::Kern(if n == 0 {
                KernContent::Rest(Rest::invisible(run.spelled.clone()))
            } else {
                KernContent::Null
            });
        }
    }
    Some(gridded)
}

/// A data line of the body: its document index, its row in the gridded
/// lines, and the time it takes.
#[derive(Debug, Clone, Copy)]
struct Row {
    index: usize,
    gridded: usize,
    duration: MusicalDuration,
}

/// What iohumdrum and Verovio read of one grid column over a measure. A
/// column the written lines lack for a while is, to iohumdrum, a gap it
/// fills with spaces of its own spelling; the dormant rest must spell the
/// same space.
#[derive(Debug, Clone)]
struct Column {
    /// What is left of the last token's duration.
    sounding: MusicalDuration,
    filled: bool,
    /// The first line the grid fills here.
    first_fill: Option<usize>,
    has_note: bool,
    /// A gap whose dormant rest would not be the one space iohumdrum writes.
    unmatched: bool,
    /// Two written invisible rests the body's removed manipulators kept
    /// apart, which compaction would join.
    joined: bool,
    /// A manipulator line of the body came after the last token.
    crossed: bool,
    /// Open beams; below zero a beam began in another column.
    beams: i32,
    /// A beam crosses into or out of this column, so iohumdrum wraps the
    /// column's gap into an invisible beam where it wrote a straddle space.
    straddled: bool,
    /// The last written token here was an invisible rest.
    last_invisible: bool,
    gap: Option<Silence>,
    runs: Vec<Fill>,
}

/// Consecutive silent lines of a column.
#[derive(Debug, Clone)]
struct Silence {
    total: MusicalDuration,
    rows: Vec<usize>,
    /// The gap ends a written invisible rest, which compaction would join.
    after_rest: bool,
}

/// One dormant rest spanning `rows`, null after the first.
#[derive(Debug, Clone)]
struct Fill {
    rows: Vec<usize>,
    spelled: Duration,
}

impl Default for Column {
    fn default() -> Self {
        Self {
            sounding: MusicalDuration::zero(),
            filled: false,
            first_fill: None,
            has_note: false,
            unmatched: false,
            joined: false,
            crossed: false,
            beams: 0,
            straddled: false,
            last_invisible: false,
            gap: None,
            runs: Vec::new(),
        }
    }
}

impl Column {
    fn onset(&mut self, content: &KernContent, duration: MusicalDuration) {
        let invisible = matches!(content, KernContent::Rest(rest) if rest.invisible)
            || matches!(content, KernContent::InvisibleRestSlice(_));
        self.unmatched |= invisible && self.gap.is_some();
        self.joined |= invisible && self.last_invisible && self.crossed;
        self.crossed = false;
        self.close_gap();
        self.has_note |= !matches!(
            content,
            KernContent::Rest(_) | KernContent::InvisibleRestSlice(_)
        );
        if let Some(beams) = content_beams(content) {
            self.beams += i32::from(beams.beam_start_count()) - i32::from(beams.beam_end_count());
            self.straddled |= self.beams < 0;
        }
        self.last_invisible = invisible;
        self.sounding = duration;
    }

    fn fill(&mut self, row: Row) {
        self.filled = true;
        self.first_fill.get_or_insert(row.index);
        let after_rest = self.last_invisible;
        let gap = self.gap.get_or_insert(Silence {
            total: MusicalDuration::zero(),
            rows: Vec::new(),
            after_rest,
        });
        gap.total += row.duration;
        gap.rows.push(row.gridded);
        self.sounding = row.duration;
    }

    /// Whether iohumdrum reads the gridded column as the written layer.
    fn reads_as_written(&self, clef_after_fill: bool) -> bool {
        let straddled = self.straddled || self.beams != 0;
        !self.joined
            && (!self.filled
                || (self.has_note && !self.unmatched && !straddled && !clef_after_fill))
    }

    fn close_gap(&mut self) {
        let Some(gap) = self.gap.take() else {
            return;
        };
        match iohumdrum_space(gap.total) {
            Some(spelled) if !gap.after_rest => self.runs.push(Fill {
                rows: gap.rows,
                spelled,
            }),
            _ => self.unmatched = true,
        }
    }
}

/// Lines in `content` with a tandem `kind` holds, by staff. iohumdrum
/// copies a clef into the staff's layers that are absent on its line, at the
/// time that layer has reached, and closes an ottava from the first layer
/// that ends it.
fn staff_tandems(
    document: &KernDocument,
    topology: &Topology,
    content: &Range<usize>,
    kind: impl Fn(&TandemInterpretation) -> bool,
) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    for index in content.clone() {
        let KernLine::TandemInterpretationLine(tandems) = &document.lines()[index] else {
            continue;
        };
        let staff_of = topology.at[index]
            .iter()
            .enumerate()
            .flat_map(|(staff, width)| std::iter::repeat_n(staff, *width));
        for (tandem, staff) in tandems.tokens().iter().zip(staff_of) {
            if kind(tandem) {
                found.push((index, staff));
            }
        }
    }
    found
}

fn content_beams(content: &KernContent) -> Option<&ChordBeams> {
    match content {
        KernContent::Chord(chord) => Some(&chord.modifiers.beams),
        KernContent::Grace(grace) => Some(&grace.modifiers.beams),
        _ => None,
    }
}

fn is_cross_staff(content: &SpineContent) -> bool {
    match content {
        SpineContent::Kern(KernContent::Note(note)) => note.modifiers.cross_staff.is_some(),
        SpineContent::Kern(KernContent::Chord(chord)) => chord
            .notes
            .iter()
            .any(|note| note.modifiers.cross_staff.is_some()),
        SpineContent::Kern(KernContent::Rest(rest)) => rest.cross_staff.is_some(),
        _ => false,
    }
}

/// `duration` as a power-of-two value with at most three dots.
fn dotted(duration: MusicalDuration) -> Option<Duration> {
    let spelled = duration.into_duration_prefer_dots()?;
    match spelled.value {
        DurationValue::Reciprocal(value) if value.is_power_of_two() => Some(spelled),
        DurationValue::Breve | DurationValue::Long | DurationValue::Maxima => Some(spelled),
        DurationValue::Reciprocal(_) | DurationValue::Rational { .. } => None,
    }
}

/// The one space iohumdrum writes for a gap of `duration`
/// (`HumdrumInput::setDuration`), or `None` when it writes several: it
/// spells a plain or once-dotted note value whole and splits anything else
/// greedily into long, breve, whole, half and quarter.
fn iohumdrum_space(duration: MusicalDuration) -> Option<Duration> {
    let spelled = dotted(duration)?;
    let single = match (&spelled.value, spelled.augmentation_dots) {
        (DurationValue::Long | DurationValue::Breve, 0) => true,
        (DurationValue::Reciprocal(1 | 2 | 4), 0) => true,
        (DurationValue::Reciprocal(value), 0) => *value >= 8 && *value <= 256,
        (DurationValue::Reciprocal(2 | 4 | 8), 1) => true,
        _ => false,
    };
    single.then_some(spelled)
}

/// Whether Verovio draws this tandem differently inside a split region. An
/// ottava keeps its column and draws alike.
fn emitted_outside_grid(tandem: &TandemInterpretation, omitted: TandemPolicy) -> bool {
    !matches!(tandem, TandemInterpretation::Null)
        && !tandem_omitted_by(omitted, tandem)
        && !is_octave_ottava(tandem)
}

fn is_octave_ottava(tandem: &TandemInterpretation) -> bool {
    matches!(
        tandem,
        TandemInterpretation::Ottava(Ottava {
            kind: OttavaKind::OctaveUp | OttavaKind::OctaveDown,
            ..
        })
    )
}

impl Grid {
    fn render(
        &self,
        document: &KernDocument,
        entry: &[usize],
        exit: &[usize],
    ) -> Vec<(KernLine, Option<usize>)> {
        let edge = |range: &Range<usize>| {
            range
                .clone()
                .map(|index| (document.lines()[index].clone(), document.source(index)))
                .collect::<Vec<_>>()
        };
        let opening = document
            .source(self.body.start.saturating_sub(1))
            .or_else(|| document.source(self.body.start));
        let closing = document.source(self.body.end - 1);

        let mut out = edge(&self.leading);
        out.extend(manipulators(transition(entry, &self.layers), opening));
        out.extend(self.gridded.iter().cloned());
        out.extend(manipulators(transition(&self.layers, exit), closing));
        out.extend(edge(&self.trailing));
        out
    }
}

fn manipulators(
    lines: Vec<SpineManipulatorLine>,
    source: Option<usize>,
) -> impl Iterator<Item = (KernLine, Option<usize>)> {
    lines
        .into_iter()
        .map(move |line| (KernLine::SpineManipulatorLine(line), source))
}

/// Each staff's cells at their column positions, widened to `layers` with
/// `fill(position)`.
fn project<T: Clone>(
    cells: &[T],
    widths: &[usize],
    layers: &[usize],
    mut fill: impl FnMut(usize) -> T,
) -> Vec<T> {
    let mut out = Vec::with_capacity(layers.iter().sum());
    let mut offset = 0;
    for (width, layers) in widths.iter().zip(layers) {
        let staff = &cells[offset..offset + width];
        out.extend((0..*layers).map(|position| match staff.get(position) {
            Some(cell) => cell.clone(),
            None => fill(position),
        }));
        offset += width;
    }
    out
}

/// A missing cell is null while the last token at its position sounds and a
/// gap otherwise, which [`grid_body`] fills. `None` when a token lands on a
/// column whose last token still sounds, or a null on a silent one: a split
/// left of a sounding voice moves it to another column.
fn project_data(
    contents: &[SpineContent],
    widths: &[usize],
    layers: &[usize],
    row: Row,
    columns: &mut [Column],
) -> Option<Vec<SpineContent>> {
    let duration = row.duration;
    let projected = project(
        &contents.iter().cloned().map(Some).collect::<Vec<_>>(),
        widths,
        layers,
        |_| None,
    );
    let mut line = Vec::with_capacity(projected.len());
    for (cell, column) in projected.into_iter().zip(columns.iter_mut()) {
        let silent = column.sounding.is_zero();
        match &cell {
            Some(SpineContent::Kern(KernContent::Null)) if silent && !duration.is_zero() => {
                return None
            }
            Some(SpineContent::Kern(content)) => {
                let token = match content {
                    KernContent::InvisibleRestSlice(_) => duration,
                    content => content.duration().ok()?,
                };
                if !token.is_zero() {
                    if !silent {
                        return None;
                    }
                    column.onset(content, token);
                }
            }
            Some(_) => {}
            None if silent && !duration.is_zero() => column.fill(row),
            None => {}
        }
        column.sounding = column
            .sounding
            .checked_sub(duration)
            .unwrap_or_else(MusicalDuration::zero);
        line.push(cell.unwrap_or(SpineContent::Kern(KernContent::Null)));
    }
    Some(line)
}

/// Widen a tandem line. An ottava start goes into every column of its staff:
/// iohumdrum opens a staff's ottava once, from its first layer, so the copies
/// draw nothing, unless the ottava also closes in this measure, when each
/// copy opens an ottava of its own after the first layer closed it. `None`
/// when an ottava end would move to a column its ottava never opened in.
fn project_tandems(
    tokens: &[TandemInterpretation],
    widths: &[usize],
    layers: &[usize],
    closing: impl Fn(usize) -> bool,
) -> Option<TandemInterpretationLine> {
    let mut projected = project(tokens, widths, layers, |_| TandemInterpretation::Null);
    let mut start = 0;
    for (staff, (layers, width)) in layers.iter().zip(widths).enumerate() {
        let cells = &mut projected[start..start + layers];
        let ottava = cells
            .iter()
            .find(|tandem| is_octave_ottava(tandem))
            .cloned();
        match ottava {
            Some(TandemInterpretation::Ottava(Ottava { active: false, .. })) if width < layers => {
                return None
            }
            Some(ottava @ TandemInterpretation::Ottava(Ottava { active: true, .. }))
                if !closing(staff) =>
            {
                cells
                    .iter_mut()
                    .filter(|tandem| matches!(tandem, TandemInterpretation::Null))
                    .for_each(|tandem| *tandem = ottava.clone());
            }
            _ => {}
        }
        start += layers;
    }
    TandemInterpretationLine::new(projected).ok()
}

/// Manipulator lines from `from` to `to`: each shrinking staff merges its
/// last columns, then each growing staff splits its last column until it is
/// wide enough. Merge groups of neighbouring staffs that would touch go on
/// separate lines.
fn transition(from: &[usize], to: &[usize]) -> Vec<SpineManipulatorLine> {
    let mut now = from.to_vec();
    let mut lines = Vec::new();
    loop {
        let mut tokens = Vec::new();
        let mut merged_previous = false;
        for (width, target) in now.iter_mut().zip(to) {
            let merge = *width > *target && !(merged_previous && *target == 1);
            if merge {
                tokens.extend(std::iter::repeat_n(SpineManipulator::Null, target - 1));
                tokens.extend(std::iter::repeat_n(
                    SpineManipulator::Merge,
                    *width - target + 1,
                ));
                *width = *target;
            } else {
                tokens.extend(std::iter::repeat_n(SpineManipulator::Null, *width));
            }
            merged_previous = merge;
        }
        if !tokens.contains(&SpineManipulator::Merge) {
            break;
        }
        lines.push(SpineManipulatorLine::new(tokens).expect("merge groups span two columns"));
    }
    loop {
        let mut tokens = Vec::new();
        for (width, target) in now.iter_mut().zip(to) {
            if *width < *target {
                tokens.extend(std::iter::repeat_n(SpineManipulator::Null, *width - 1));
                tokens.push(SpineManipulator::Split);
                *width += 1;
            } else {
                tokens.extend(std::iter::repeat_n(SpineManipulator::Null, *width));
            }
        }
        if !tokens.contains(&SpineManipulator::Split) {
            break;
        }
        lines.push(SpineManipulatorLine::new(tokens).expect("split lines are well formed"));
    }
    lines
}

#[cfg(test)]
mod tests {
    use crate::{serialize::SerializePreset, KernDocument};

    const HEADER: [&str; 4] = [
        "**kern\t**kern",
        "*clefF4\t*clefG2",
        "*k[]\t*k[]",
        "*M2/4\t*M2/4",
    ];

    fn label(text: &str) -> String {
        SerializePreset::TranscodaV2
            .try_serialize_document(&KernDocument::parse(text).unwrap())
            .unwrap()
    }

    fn rows(rows: &[&str]) -> String {
        rows.iter()
            .map(|row| row.replace(' ', "\t") + "\n")
            .collect()
    }

    const LABEL_HEADER: [&str; 3] = ["*clefF4 *clefG2", "*k[] *k[]", "*M2/4 *M2/4"];

    /// Labels `body` under a two-staff header, compares the label after its
    /// header with `expected` (`.` for an empty null), and relabels the label.
    fn assert_grid(body: &[&str], expected: &[&str]) {
        let input = rows(&HEADER) + &rows(body) + "*-\t*-\n";
        let expected = rows(&LABEL_HEADER)
            + &expected
                .iter()
                .map(|row| {
                    let cells = row
                        .split(' ')
                        .map(|cell| if cell == "." { "" } else { cell });
                    cells.collect::<Vec<_>>().join("\t") + "\n"
                })
                .collect::<String>();
        let output = label(&input);
        assert_eq!(output, expected);
        assert_eq!(label(&output), output, "the label relabels to itself");
    }

    #[test]
    fn a_merge_inside_the_next_measure_moves_before_its_closing_barline() {
        assert_grid(
            &[
                "=1",
                "* *^",
                "4C 4cc 4e",
                "=2",
                "4C 4cc 4e",
                "* *v *v",
                "4C 4cc",
                "=3",
            ],
            &[
                "* *^",
                "4C 4cc 4e",
                "=",
                "4C 4cc 4e",
                "4C 4cc 4ryy",
                "* *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn a_split_inside_a_measure_moves_after_its_opening_barline() {
        assert_grid(
            &[
                "=1",
                "4C 4cc",
                "* *^",
                "4C 4cc 4e",
                "=2",
                "4C 4cc 4e",
                "4C 4cc 4e",
                "* *v *v",
                "=3",
            ],
            &[
                "* *^",
                "4C 4cc 4ryy",
                "4C 4cc 4e",
                "=",
                "4C 4cc 4e",
                "4C 4cc 4e",
                "* *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn a_voice_shifted_by_a_nested_split_keeps_every_token_at_its_column() {
        assert_grid(
            &[
                "=1",
                "* *^",
                "4C 2cc 4e",
                "* * *^",
                "4C . 4g 4b",
                "* * *v *v",
                "* *v *v",
                "=2",
            ],
            &[
                "* *^",
                "* * *^",
                "4C 2cc 4e 4ryy",
                "4C . 4g 4b",
                "* *v *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn a_merge_of_sub_spines_from_two_splits_becomes_a_grid() {
        assert_grid(
            &[
                "=1",
                "* *^",
                "4C 4cc 4e",
                "=2",
                "* *^ *^",
                "4C 4cc 4a 4e 4g",
                "* * *v *v *v",
                "4D 4dd 4f",
                "* *v *v",
                "=3",
                "4C 4cc",
                "=4",
            ],
            &[
                "* *^",
                "4C 4cc 4e",
                "=",
                "* * *^",
                "* * * *^",
                "4C 4cc 4a 4e 4g",
                "4D 4dd 4f 4ryy 4ryy",
                "* *v *v *v *v",
                "=",
                "4C 4cc",
                "=",
            ],
        );
    }

    #[test]
    fn an_ottava_start_inside_the_grid_goes_into_every_column_of_its_staff() {
        let grid = [
            "* *^",
            "4C 4cc 4ryy",
            "* *8va *8va",
            "4C 4ccc 4eee",
            "* *v *v",
            "=",
        ];
        assert_grid(
            &[
                "=1",
                "4C 4cc",
                "* *8va",
                "* *^",
                "4C 4ccc 4eee",
                "* *v *v",
                "=2",
            ],
            &grid,
        );
        assert_grid(
            &[
                "=1",
                "* *^",
                "4C 4cc 4ryy",
                "* *8va *8va",
                "4C 4ccc 4eee",
                "* *v *v",
                "=2",
            ],
            &grid,
        );
    }

    #[test]
    fn a_clef_inside_a_split_region_leaves_its_measure_as_written() {
        assert_grid(
            &[
                "=1",
                "4C 4cc",
                "* *^",
                "4C 4cc 4e",
                "=2",
                "4C 4cc 4e",
                "*clefG2 * *",
                "4c 4dd 4f",
                "=3",
                "4c 4cc 4e",
                "* *v *v",
                "4c 4cc",
                "=4",
            ],
            &[
                "* *^",
                "4C 4cc 4ryy",
                "4C 4cc 4e",
                "=",
                "4C 4cc 4e",
                "*clefG2 * *",
                "4c 4dd 4f",
                "=",
                "4c 4cc 4e",
                "4c 4cc 4ryy",
                "* *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn a_clef_at_the_measure_start_stays_before_the_split() {
        assert_grid(
            &[
                "=1",
                "* *^",
                "2C 2cc 2e",
                "=2",
                "*clefG2 * *",
                "4c 4cc 4e",
                "* *v *v",
                "4c 4cc",
                "=3",
                "*clefF4 *",
                "4C 4cc",
                "* *^",
                "4C 4cc 4e",
                "* *v *v",
                "=4",
            ],
            &[
                "* *^",
                "2C 2cc 2e",
                "=",
                "*clefG2 * *",
                "4c 4cc 4e",
                "* *v *v",
                "4c 4cc",
                "=",
                "*clefF4 *",
                "* *^",
                "4C 4cc 4ryy",
                "4C 4cc 4e",
                "* *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn a_column_stays_null_while_its_last_token_sounds() {
        assert_grid(
            &[
                "=1",
                "* *^",
                "16C 8cc 8e",
                "* *v *v",
                "16D .",
                "8E 8dd",
                "4F 4ee",
                "=2",
            ],
            &[
                "* *^",
                "16C 8cc 8e",
                "16D . .",
                "8E 8dd 4.ryy",
                "4F 4ee .",
                "* *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn a_split_that_moves_a_sounding_voice_leaves_its_measure_as_written() {
        assert_grid(
            &[
                "=1",
                "* *^",
                "4C 4a 2b",
                "* *^ *",
                "4C 4c 4d .",
                "* *v *v *v",
                "=2",
            ],
            &[
                "* *^",
                "4C 4a 2b",
                "* *^ *",
                "4C 4c 4d .",
                "* *v *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn a_gap_iohumdrum_spells_as_two_spaces_leaves_the_measure_as_written() {
        assert_grid(
            &[
                "=1",
                "* *^",
                "16C 16cc 16e",
                "* *v *v",
                "16D 16dd",
                "4.E 4.ee",
                "=2",
            ],
            &[
                "* *^",
                "16C 16cc 16e",
                "* *v *v",
                "16D 16dd",
                "4.E 4.ee",
                "=",
            ],
        );
    }

    #[test]
    fn a_column_of_rests_only_leaves_the_measure_as_written() {
        assert_grid(
            &["=1", "* *^", "4C 4cc 4r", "* *v *v", "4D 4dd", "=2"],
            &["* *^", "4C 4cc 4r", "* *v *v", "4D 4dd", "="],
        );
    }

    #[test]
    fn a_clef_after_a_gap_in_its_staff_leaves_the_measure_as_written() {
        assert_grid(
            &[
                "=1",
                "* *^",
                "4C 4cc 4e",
                "* *v *v",
                "4D 4dd",
                "* *clefG2",
                "=2",
            ],
            &["* *^", "4C 4cc 4e", "* *v *v", "4D 4dd", "* *clefG2", "="],
        );
    }

    #[test]
    fn a_beam_that_crosses_into_a_gapped_column_leaves_the_measure_as_written() {
        assert_grid(
            &[
                "=1",
                "8C 8ccL",
                "8D 8dd",
                "* *^",
                "8E 8ee 8g",
                "8F 8ff 8aJ",
                "* *v *v",
                "=2",
            ],
            &[
                "8C 8ccL",
                "8D 8dd",
                "* *^",
                "8E 8ee 8g",
                "8F 8ff 8aJ",
                "* *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn a_cross_staff_note_on_a_widened_line_leaves_the_measure_as_written() {
        assert_grid(
            &["=1", "4C 4G<", "* *^", "4D 4dd 4ff", "* *v *v", "=2"],
            &["4C 4G<", "* *^", "4D 4dd 4ff", "* *v *v", "="],
        );
    }

    #[test]
    fn invisible_rests_kept_apart_by_a_merge_leave_the_measure_as_written() {
        assert_grid(
            &[
                "=1",
                "* *^",
                "4C 4cc 4ryy",
                "* *v *v",
                "* *^",
                "4D 4dd 4ryy",
                "* *v *v",
                "=2",
            ],
            &[
                "* *^",
                "4C 4cc 4ryy",
                "* *v *v",
                "* *^",
                "4D 4dd 4ryy",
                "* *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn a_beam_open_across_a_barline_leaves_its_measures_as_written() {
        assert_grid(
            &[
                "=1",
                "4C 4cc",
                "* *^",
                "8D 8ddL 4f",
                "8E 8ee .",
                "* *v *v",
                "=2",
                "8F 8ffJ",
                "8G 8gg",
                "=3",
            ],
            &[
                "4C 4cc",
                "* *^",
                "8D 8ddL 4f",
                "8E 8ee .",
                "* *v *v",
                "=",
                "8F 8ffJ",
                "8G 8gg",
                "=",
            ],
        );
    }

    #[test]
    fn an_ottava_that_closes_in_its_measure_is_not_copied() {
        assert_grid(
            &[
                "=1",
                "4C 4cc",
                "* *8va",
                "* *^",
                "8D 8ddd 8fff",
                "8E 8eee 8ggg",
                "* *X8va *",
                "* *v *v",
                "=2",
            ],
            &[
                "* *^",
                "4C 4cc 4ryy",
                "* *8va *",
                "8D 8ddd 8fff",
                "8E 8eee 8ggg",
                "* *X8va *",
                "* *v *v",
                "=",
            ],
        );
    }

    #[test]
    fn an_ottava_end_on_a_widened_line_leaves_the_measure_as_written() {
        assert_grid(
            &[
                "=1",
                "* *8va",
                "* *^",
                "4C 4ccc 4eee",
                "* *v *v",
                "* *X8va",
                "4D 4dd",
                "=2",
            ],
            &[
                "* *8va",
                "* *^",
                "4C 4ccc 4eee",
                "* *v *v",
                "* *X8va",
                "4D 4dd",
                "=",
            ],
        );
    }
}
