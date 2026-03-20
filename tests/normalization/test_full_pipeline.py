import re
from pathlib import Path

from scripts.dataset_generation.normalization.presets import preprocess_and_normalize_kern

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _ekern_to_kern(text: str) -> str:
    """Pre-convert ekern to kern (simulates the convert_ekern_to_kern.sh step)."""
    text = re.sub(r"\*\*ekern[^\t\n]*", "**kern", text)
    return text.replace("@", "").replace("\u00b7", "")


def test_full_pipeline_on_original():
    transcription = """**ekern	**ekern
*	*^
*clefF4	*clefG2	*clefG2
*k[b-]	*k[b-]	*k[b-]
*M2/4	*M2/4	*M2/4
.	.	.
4@F	8@.@.@a·L	4@c
.	32@f·J·k	.
4@AA 4@F 4@A	8@.@.@cc·L	4@c 4@f
.	32@a·J·k	.
=	=	=
4@BB@- 4@F 4@B@-	8@.@.@dd·L	4@d 4@f
.	32@b@-·J·k	.
4@D·[ 4@F·[ 4@B@-·[ 4@d·[	4@r	4@f·[ 4@b@-·[ 4@ff·[
=	=	=
4@D·] 4@F·] 4@B@-·] 4@d·]	16@ff·L	4@f·] 4@b@-·]
.	16@ee	.
.	16@dd	.
.	16@cc·J	.
8@E·L 8@c·L	8@g·L 8@b@-·L	4@r
8@F·J 8@c·J	8@f·J 8@a·J	.
=	=	=
*^	*	*
8@.@A·L	4@C	8@.@cc·L	8@c·L
.	.	.	8@c
16@F·J·k	.	16@a·J·k	.
8@E	8@C	16@g·L	8@c·J
.	.	f@##·L·q	.
.	.	g·J·q	.
.	.	32@b@-·L	.
.	.	32@a·J	.
8@r	8@r	32@cc·L	8@r
.	.	32@b@-	.
.	.	48@a	.
.	.	48@b@-	.
.	.	48@g·J	.
=	=	=	=
16@FF·L	4@FF	4@f	4@A
16@C	.	.	.
16@C	.	.	.
16@C·J	.	.	.
16@EE·L	4@EE	8@.@a·L	4@B@-
16@C	.	.	.
16@C	.	.	.
16@C·J	.	16@g·J·k	.
*v	*v	*	*
*	*v	*v
=	=
*-	*-
"""

    transcription = _ekern_to_kern(transcription)
    normalized = preprocess_and_normalize_kern(transcription)

    expected_normalized = """*clefF4	*clefG2
*k[b-]	*k[b-]
*M2/4	*M2/4
*	*^
4F	8..aL	4c
.	32fJk	.
4AA 4F 4A	8..ccL	4c 4f
.	32aJk	.
=	=	=
4BB- 4F 4B-	8..ddL	4d 4f
.	32b-Jk	.
4D[ 4F[ 4B-[ 4d[	4r	4f[ 4b-[ 4ff[
=	=	=
4D] 4F] 4B-] 4d]	16ffL	4f] 4b-]
.	16ee	.
.	16dd	.
.	16ccJ	.
8E 8cL	8g 8b-L	4r
8F 8cJ	8f 8aJ	.
=	=	=
*^	*	*
8.AL	4C	8.ccL	8cL
.	.	.	8c
16FJk	.	16aJk	.
8E	8C	16gL	8cJ
.	.	f##qL	.
.	.	gqJ	.
.	.	32b-L	.
.	.	32aJ	.
8r	8r	32ccL	8r
.	.	32b-	.
.	.	48a	.
.	.	48b-	.
.	.	48gJ	.
=	=	=	=
16FFL	4FF	4f	4A
16C	.	.	.
16C	.	.	.
16CJ	.	.	.
16EEL	4EE	8.aL	4B-
16C	.	.	.
16C	.	.	.
16CJ	.	16gJk	.
=	=	=	="""

    assert normalized == expected_normalized


def test_full_pipeline_on_polish_scores_66():
    """Test pipeline on polish-scores data with malformed interpretation spacing.

    This test verifies that the RepairInterpretationSpacing pass correctly fixes
    lines like '*\\t*Xtuplet    *' (spaces instead of tabs) in external data.
    """
    transcription = (FIXTURES_DIR / "polish_scores_index_66.krn").read_text()
    transcription = _ekern_to_kern(transcription)

    # The pipeline should complete without error after the repair pass fixes
    # the malformed interpretation lines
    normalized = preprocess_and_normalize_kern(transcription)

    expected_normalized = """*clefF4	*clefG2
*k[b-e-a-d-]	*k[b-e-a-d-]
*M3/4	*M3/4
.	8gq
*^	*
4r	2EEn	8cc)L
.	.	16r
.	.	16cc(Jk
4C 4G	.	4g
.	.	16qqgL
.	.	16qqb-J
4BBn	4FF	4a-)
*v	*v	*
=	=
4EEn	2g([
4C 4G	.
4EE-	8g]L
.	8g-J
=	=
.	16qqfL
.	16qqgnJ
4DDn	16f)KL
.	16r
.	8en(
4Dn 4A- 4Bn	8f
.	8g
4DD	8a-
.	8b-)J
=	=
4EE-	20bnL
.	20r
.	20cc
.	20ccc
.	20ccJ
2E- 2A- 2e-	4.eee- 4.cccc
.	8ccc 8aaa-
=	=
*	*^
6EE-	24eee- 24gggKL	4r
.	24r	.
.	12ddd- 12fff	.
12E-	12ff#J	.
12B-L	6ddd-	12ggL
12d-	.	12ff-
12gJ	12ccc	12ee-J
8r	6bb-	12ddnL
.	.	12ee-
8E-	.	.
.	12bbn	12dd-J
*	*v	*v
=	=
4AA- 4A-	4cc[ 4ccc[
4E- 4A- 4c	12cc] 12ccc]L
.	12bn 12bbn
.	12cc[ 12ccc[J
8FF 8CL	16cc] 16ccc]L
.	16ee- 16eee-
8En 8B-J	16ddn 16dddn
.	16dd- 16ddd-J
=	=
.	8ccq( 8cccq(
*^	*
4FF 4F 4A-	4r	8ff) 8fff)L
.	.	16r
.	.	16ff 16fffJk
4AA- 4F	2FF[	4cc 4ccc
.	.	8eee-q(
4BB- 4G	.	4dd-) 4ddd-)
=	=	=
4C 4A-	4FF]	4cc[ 4ccc[
4FFF 4FF	4r	12cc] 12ccc]L
.	.	12cc( 12ccc(
.	.	12bn 12bbnJ
4FF 4C 4A-	4r	12b 12bbL
.	.	12an 12aan
.	.	12a-) 12aa-)J
*v	*v	*
=	=
.	8gq( 8ggq(
4EEn 4C 4G	8cc) 8ccc)L
.	16r
.	16cc 16cccJk
4EE	4g 4gg
.	8bb-q(
4EE 4BBn 4F	4a-) 4aa-)
=	=
4EEn 4C 4G	4.g 4.gg
4EEEn 4EE	.
.	8g 8ggL
4EEE- 4EE-	8g 8gg
.	8g- 8gg-J
=	=
.	16qqffL
.	16qqggJ
8DDDn 8DDn	8f) 8ff)L
8r	8en 8een
4Dn 4A- 4Bn	8f 8ff
.	8gn 8ggn
4DDD-( 4DD-(	8a- 8aa-
.	8b- 8bb-J
=	=
8CCC) 8CC)	32bn 32bbnKL
.	32r
.	16cc
8r	16aa- 16ccc
.	16cccJ
4F 4A- 4d-	8aaa- 8cccc
.	8r
8r	4cc[ 4ccc[
8CC 8C	.
=	=
4CC- 4C-	32cc] 32ccc]KL
.	32r
.	16dd-(
.	16aa- 16ddd-
.	16ddd-)J
4D- 4F 4d-	8aaa- 8dddd-
.	8r
8r	4d-[ 4dd-[
8CC- 8C-	.
=	=
4BBB- 4BB-	32d-] 32dd-]KL
.	32r
.	16d-(
.	16g- 16dd-
.	16dd-J
4D- 4G- 4d-	8ggg-) 8dddd-)
.	8r
4BBB- 4BB-	16r
.	16en(L
.	16gn 16een
.	16eeJ
=	=
4BB- 4Gn 4d-	8ggn) 8eeen)
.	8r
4AAA- 4AA-	16r
.	16f(L
.	16cc 16ff
.	16ffJ
8AA- 8C 8F 8c	8ccc) 8fff)
16r	16r
16C 16c	16r
=	=
8D- 8d-L	4r
16r	.
16D-( 16d-(Jk	.
2EEn 2En	4r
.	4B-( 4d-( 4g(
=	=
8FF) 8F)L	8A-) 8c) 8a-)
16r	8r
16BBn( 16Bn(Jk	.
2DD- 2D-	4r
.	4f( 4bn( 4ff(
=	=
8CC) 8C)L	8a-) 8cc)
16r	8r
16C( 16c(Jk	.
4..CCC) 4..CC)	4r
.	8r
.	16r
16CCC( 16CC(	16g 16b- 16een
=	=
8FFF)	8f 8a- 8ff
8r	8r
8FF: 8C: 8A-: 8c: 8a-: 8ff:	8cc: 8ff: 8ccc: 8fff:
8r	8r
==	=="""

    assert normalized == expected_normalized


def test_full_pipeline_on_grandstaff_607():
    """Test pipeline on polish-scores data with malformed interpretation spacing.

    This test verifies that the RepairInterpretationSpacing pass correctly fixes
    lines like '*\\t*Xtuplet    *' (spaces instead of tabs) in external data.
    """
    transcription = (FIXTURES_DIR / "grandstaff_index_607.krn").read_text()
    transcription = _ekern_to_kern(transcription)

    # The pipeline should complete without error after the repair pass fixes
    # the malformed interpretation lines
    normalized = preprocess_and_normalize_kern(transcription)

    expected_normalized = """*clefF4	*clefG2
*k[b-e-a-]	*k[b-e-a-]
*M2/4	*M2/4
8r	8bb-[L
8E- 8GL	24bb-]L
.	24gg
.	24ee-J
8F 8A-	24ddL
.	24ff
.	24b-J
8G 8B-J	24ee-L
.	24gg
.	24b-J
=	=
8r	8ff[L
8BB- 8DL	24ff]L
.	24dd
.	24b-J
8C 8E-	24aL
.	24b-
.	24fJ
8D 8FJ	24a-L
.	24b-
.	24fJ
=	=
8r	8bb-[L
8E- 8GL	24bb-]L
.	24gg
.	24ee-J
8F 8A-	24ddL
.	24ff
.	24b-J
8G 8B-J	24ee-L
.	24gg
.	24b-J
=	=
8r	8ff[L
8BB- 8DL	24ff]L
.	24dd
.	24b-J
8C 8E-	24aL
.	24b-
.	24fJ
8D 8FJ	24a-L
.	24b-
.	24fJ
=	=
4E-	24gL
.	24e-
.	24gJ
.	24b-L
.	24g
.	24b-J
4r	24ee-L
.	24b-
.	24ee-J
.	24ggL
.	24ee-
.	24ggJ
=	=
*^	*
*	*^	*
4.B-	2E-	2e-[	24bb-L
.	.	.	24gg
.	.	.	24bb-J
.	.	.	24ddd-L
.	.	.	24bb-
.	.	.	24ggJ
.	.	.	24ee-L
.	.	.	24dd-
.	.	.	24b-J
8d-	.	.	24ee-L
.	.	.	24dd-
.	.	.	24bJ
=	=	=	="""

    assert normalized == expected_normalized


def test_full_pipeline_merge_then_terminate_no_barline():
    """Test pipeline handles merge followed by termination (no barline).

    Terminator records are stripped by normalization, so the merged-data
    output should end at the last data line.
    """
    transcription = """\
**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
*k[]\t*k[]\t*k[]
*M4/4\t*M4/4\t*M4/4
4C\t4e\t4g
4D\t4f\t4a
*\t*v\t*v
*-\t*-
"""

    normalized = preprocess_and_normalize_kern(transcription)

    expected_normalized = """*clefF4\t*clefG2\t*clefG2
*k[]\t*k[]\t*k[]
*M4/4\t*M4/4\t*M4/4
4C\t4e\t4g
4D\t4f\t4a"""

    assert normalized == expected_normalized


def test_full_pipeline_reorders_header_correctly():
    """Test that the pipeline reorders clef, key, and meter to canonical order.

    Input header order: meter (*M3/4), key interpretation (*d:), key sig (*k[b-]), clef (*clefG2)
    Expected order: clef, key sig, meter (key interpretation *d: should be removed)
    """
    transcription = """\
!!!OTL: All Things Are Quite Silent
**kern	**mxhm
*M3/4	*
*d:	*
*k[b-]	*
*clefG2	*
=1	=1
4d	.
=	=
4f	D minor
4e	.
4d	.
=	=
4dd	G major
4cc	.
4dd	.
=	=
4.a	D minor
8g	.
8aL	.
8gJ	.
=	=
!LO:TX:a:t=(G)	!
4f	.
4e	.
4d	.
=	=
4f	D minor
4f	.
4f	.
=	=
4g	C major
4f	.
4g	.
=	=
2a	F major
4b-	.
=	=
2a	A major
4a	.
=	=
4cc	C major
4dd	.
4cc	.
=	=
4cc	F major
4a	.
4f	.
=	=
4dd	G major
4cc	.
4dd	.
=	=
4a	D minor
4g	.
4f	.
=	=
4g	G major
4dd	.
4dd	.
=	=
4cc	A minor
4d	.
4e	.
=	=
4.f	F major
8gL	.
8a	.
8b-J	.
=	=
2a	A major
==	==
*k[]	*
2.r	B major
=	=
8aL	.
8gJ	.
=	=
4f	D minor
4e	.
4d	.
=	=
4g	G major
4a	.
4b	.
=	=
4g	G major
8aL	.
8gJ	.
4e	C major
=	=
2d	D minor
8dL	.
8eJ	.
=	=
4f	.
4e	.
4d	.
=	=
4g	G major
8aL	.
8gJ	.
4b	.
=	=
4g	G major
4a	.
4e	A minor
=	=
2a	.
8aL	.
8ccJ	.
=	=
8bL	G major
8aJ	.
8gL	.
8aJ	.
4b	.
=	=
8ccL	A minor
8bJ	.
4a	.
4a	.
=	=
4dd	D minor
4cc	.
4dd	.
=	=
2a	A major
8aL	.
8gJ	.
=	=
4f	D minor
4e	.
4d	.
=	=
4g	G major
4a	.
4b	.
=	=
4g	G major
8aL	.
8gJ	.
4e	C major
=	=
2d	D minor
==	==
*-	*-
"""

    normalized = preprocess_and_normalize_kern(transcription)

    # Verify the header is in canonical order: clef, key, meter
    lines = normalized.split("\n")

    # Find the header lines (should be first 3 non-empty lines)
    header_lines = []
    for line in lines:
        if line.startswith("*clef") or line.startswith("*k[") or line.startswith("*M"):
            header_lines.append(line)
        elif line.startswith("="):
            break

    assert len(header_lines) == 3, f"Expected 3 header lines, got {len(header_lines)}: {header_lines}"

    # Check order: clef first, then key, then meter
    assert header_lines[0].startswith("*clef"), f"First header should be clef, got: {header_lines[0]}"
    assert header_lines[1].startswith("*k["), f"Second header should be key sig, got: {header_lines[1]}"
    assert header_lines[2].startswith("*M"), f"Third header should be meter, got: {header_lines[2]}"


def test_full_pipeline_preserves_percussion_clefs():
    """Test that *clefX (percussion clef) is preserved in both spines.

    Percussion instruments like bass drum and side drum use *clefX.
    The normalization pipeline should not remove or deduplicate these.
    """
    transcription = """\
**kern	**kern
*part2	*part1
*staff2	*staff1
*I"Bass Drum	*I"Side Drum
*I'B.D.	*I'S.D.
*stria1	*stria1
*clefX	*clefX
*k[]	*k[]
*M2/4	*M2/4
*MM116	*MM116
=1	=1
!	!LO:TX:a:t=[quarter]=116
4Re^\\	8r
.	8Rf\\
4Re\\	8r
.	8Rd\\
=	=
4Re^\\	8r
.	8Rf\\
4Re\\	8r
.	8Rd\\
=	=
4Re^\\	8r
.	8Rf\\
4Re\\	8r
.	8Rd\\
=	=
4Re^\\	8r
.	16Rf\\LL
.	16Rf\\JJ
4Re\\	8Rd\\L
.	8Rd\\J
==	==
*-	*-
"""

    normalized = preprocess_and_normalize_kern(transcription)

    # Both *clefX entries should be preserved
    lines = normalized.split("\n")
    clef_lines = [line for line in lines if "*clefX" in line]

    assert len(clef_lines) == 1, f"Expected 1 clef line with both *clefX, got {len(clef_lines)}: {clef_lines}"

    # The clef line should have *clefX in both columns (tab-separated)
    clef_line = clef_lines[0]
    clef_columns = clef_line.split("\t")
    assert len(clef_columns) == 2, f"Expected 2 columns in clef line, got {len(clef_columns)}: {clef_columns}"
    assert clef_columns[0] == "*clefX", f"First spine should have *clefX, got: {clef_columns[0]}"
    assert clef_columns[1] == "*clefX", f"Second spine should have *clefX, got: {clef_columns[1]}"


def test_full_pipeline_removes_null_ties():
    """Test that null ties like [4G] are normalized to 4G.

    A null tie is a tie that opens and closes on the same note token.
    This is invalid kern notation and should be removed by the pipeline.
    """
    transcription = """\
**kern	**kern
*clefF4	*clefG2
*k[]	*k[]
*M4/4	*M4/4
[4G]	2g
4G]	.
4BB	4d
4C	4e
*-	*-
"""

    normalized = preprocess_and_normalize_kern(transcription)

    # The [4G] should become 4G (null tie removed)
    # The 4G] should remain 4G] (valid closing tie)
    lines = normalized.split("\n")

    # Find the line with the note that was [4G]
    # After normalization it should be "4G" not "[4G]"
    note_lines = [line for line in lines if "4G" in line and not line.startswith("*")]

    assert len(note_lines) == 2, f"Expected 2 lines with 4G, got {len(note_lines)}: {note_lines}"

    # First note line should have null tie removed: 4G (not [4G])
    first_note = note_lines[0].split("\t")[0]
    assert first_note == "4G", f"Expected '4G' (null tie removed), got '{first_note}'"

    # Second note line should preserve closing tie: 4G]
    second_note = note_lines[1].split("\t")[0]
    assert second_note == "4G]", f"Expected '4G]' (closing tie preserved), got '{second_note}'"
