use crate::*;

#[test]
fn kern_document_rejects_local_comment_overlay_on_kern_data() {
    assert!(matches!(
        KernDocument::parse("**kern\t**kern\n4c\t!LO:TX:t=cue\n").unwrap_err(),
        ParseError::MixedKernLineTypes {
            line: 2,
            column: 2,
            ..
        }
    ));
}

#[test]
fn kern_document_parses_typed_layout_text_local_comment_lines() {
    let document = KernDocument::parse("**kern\t**kern\n!LO:TX:a:t=Allegro\t!\n4c\t4e\n").unwrap();

    let local_comment_line = document
        .lines()
        .iter()
        .find(|line| matches!(line, KernLine::LocalCommentLine(_)))
        .expect("expected local comment line");
    assert!(matches!(
        local_comment_line,
        KernLine::LocalCommentLine(comments)
            if matches!(&comments[0], LocalComment::LayoutText(text) if text.text == "Allegro")
                && matches!(comments[1], LocalComment::Null)
    ));
}

#[test]
fn kern_document_parses_unsupported_layout_text_local_comments_as_opaque() {
    let document = KernDocument::parse(
        "**kern\t**kern\n!LO:TX:a:t=[quarter]=70\t!LO:TX:a:t=8va  _  _  _\n4c\t4e\n",
    )
    .unwrap();

    let local_comment_line = document
        .lines()
        .iter()
        .find(|line| matches!(line, KernLine::LocalCommentLine(_)))
        .expect("expected local comment line");
    assert!(matches!(
        local_comment_line,
        KernLine::LocalCommentLine(comments)
            if matches!(&comments[0], LocalComment::Unsupported(raw) if raw == "!LO:TX:a:t=[quarter]=70")
                && matches!(&comments[1], LocalComment::Unsupported(raw) if raw == "!LO:TX:a:t=8va  _  _  _")
    ));
}

#[test]
fn kern_document_retains_note_layout_comments_for_rendering() {
    let document =
        KernDocument::parse("**kern\t**kern\n!LO:N:head=invis\t!LO:N:head=x\n4c\t4e\n*-\t*-\n")
            .unwrap();

    let local_comment_line = document
        .lines()
        .iter()
        .find(|line| matches!(line, KernLine::LocalCommentLine(_)))
        .expect("expected note layout comment line");
    assert!(matches!(
        local_comment_line,
        KernLine::LocalCommentLine(comments)
            if matches!(&comments[0], LocalComment::Unsupported(raw) if raw == "!LO:N:head=invis")
                && matches!(&comments[1], LocalComment::Unsupported(raw) if raw == "!LO:N:head=x")
    ));
}
