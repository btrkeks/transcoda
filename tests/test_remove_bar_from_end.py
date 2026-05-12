"""Tests for remove_bar_from_end method."""

from scripts.dataset_generation.dataset_generation.image_generation.kern_ops import remove_bar_from_end


class TestRemoveBarFromEnd:
    """Test suite for the _remove_bar_from_end static method.

    This method removes the last measure (bar) from a transcription by:
    1. Finding the last single bar line (=\t=)
    2. Removing everything from that bar line up to (but not including) the next bar line or terminator
    """

    def test_removes_last_measure_before_double_bar(self):
        """Should remove the last measure before a final double bar line."""
        transcription = """**kern\t**kern
*clefG2\t*clefF4
4c\t4C
4d\t4D
=\t=
4e\t4E
4f\t4F
=\t=
4g\t4G
4a\t4A
==\t==
*-\t*-"""

        expected = """**kern\t**kern
*clefG2\t*clefF4
4c\t4C
4d\t4D
=\t=
4e\t4E
4f\t4F
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_removes_measure_before_terminator_without_double_bar(self):
        """Should remove last measure when there's no double bar, only terminator."""
        transcription = """4c\t4C
4d\t4D
=\t=
4e\t4E
4f\t4F
=\t=
4g\t4G
*-\t*-"""

        expected = """4c\t4C
4d\t4D
=\t=
4e\t4E
4f\t4F
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_single_measure_before_double_bar(self):
        """Should handle transcription with only one measure."""
        transcription = """*clefG2\t*clefF4
4c\t4C
4d\t4D
=\t=
4e\t4E
==\t==
*-\t*-"""

        expected = """*clefG2\t*clefF4
4c\t4C
4d\t4D
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_no_single_bars_returns_unchanged(self):
        """Should return transcription unchanged if no single bar lines present."""
        transcription = """*clefG2\t*clefF4
4c\t4C
4d\t4D
==\t==
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == transcription

    def test_empty_string(self):
        """Should handle empty string gracefully."""
        result = remove_bar_from_end("")
        assert result == ""

    def test_preserves_double_bar_line(self):
        """Should preserve double bar lines (==) and not treat them as single bars."""
        transcription = """4c\t4C
=\t=
4e\t4E
==\t==
4g\t4G
=\t=
4a\t4A
==\t==
*-\t*-"""

        expected = """4c\t4C
=\t=
4e\t4E
==\t==
4g\t4G
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_multiple_measures_removes_only_last(self):
        """Should remove only the last measure, not affect earlier ones."""
        transcription = """*clefG2\t*clefF4
4c\t4C
=\t=
4d\t4D
=\t=
4e\t4E
=\t=
4f\t4F
==\t==
*-\t*-"""

        expected = """*clefG2\t*clefF4
4c\t4C
=\t=
4d\t4D
=\t=
4e\t4E
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_complex_measure_content(self):
        """Should handle complex measure content with various notation elements."""
        transcription = """*clefG2\t*clefF4
*k[f#c#]\t*k[f#c#]
*M3/4\t*M3/4
4c#\t4C#
4d\t4D
=\t=
4e\t4E
[4f\t[4F
=\t=
4f]\t4F]
4g\t4G
8a\t8A
8b\t8B
==\t==
*-\t*-"""

        expected = """*clefG2\t*clefF4
*k[f#c#]\t*k[f#c#]
*M3/4\t*M3/4
4c#\t4C#
4d\t4D
=\t=
4e\t4E
[4f\t[4F
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_measure_with_interpretation_changes(self):
        """Should remove measure containing interpretation changes like clef changes."""
        transcription = """*clefF4\t*clefG2
4C\t4c
=\t=
4D\t4d
*clefG2\t*
4e\t4e
=\t=
4f\t4f
==\t==
*-\t*-"""

        expected = """*clefF4\t*clefG2
4C\t4c
=\t=
4D\t4d
*clefG2\t*
4e\t4e
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_only_header_and_one_bar(self):
        """Should handle transcription with only header and one bar line."""
        transcription = """**kern\t**kern
*clefG2\t*clefF4
=\t=
*-\t*-"""

        expected = """**kern\t**kern
*clefG2\t*clefF4
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_real_world_example(self):
        """Should handle the real-world example from user's request."""
        # Simplified version of the user's example
        transcription = """**kern\t**kern
*clefF4\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
16C\t32r
16r\t32eeL
=\t=
32CL\t8c
32E\t.
32GJ\t.
=\t=
16CC\t16E 16G 16c
16r\t16r
32GGL\t8F 8G 8B 8d
32GGJ\t.
==\t==
*-\t*-"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
*k[]\t*k[]
*M4/4\t*M4/4
16C\t32r
16r\t32eeL
=\t=
32CL\t8c
32E\t.
32GJ\t.
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_three_staff_consistent(self):
        """Should remove last bar from piece with 3 staves throughout."""
        transcription = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
*k[]\t*k[]\t*k[]
4C\t4c\t4cc
4D\t4d\t4dd
=\t=\t=
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
4G\t4g\t4gg
4A\t4a\t4aa
==\t==\t==
*-\t*-\t*-"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
*k[]\t*k[]\t*k[]
4C\t4c\t4cc
4D\t4d\t4dd
=\t=\t=
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
*-\t*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_staff_split_2_to_3(self):
        """Should handle staff split transitioning from 2 to 3 staves."""
        transcription = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
4D\t4d
=\t=
*\t*^
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
4G\t4g\t4gg
4A\t4a\t4aa
==\t==\t==
*-\t*-\t*-"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
4D\t4d
=\t=
*\t*^
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
*-\t*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_staff_merge_3_to_2(self):
        """Should handle staff merge transitioning from 3 to 2 staves."""
        transcription = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4C\t4c\t4cc
4D\t4d\t4dd
=\t=\t=
*\t*v\t*v
4E\t4e
4F\t4f
=\t=
4G\t4g
4A\t4a
==\t==
*-\t*-"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4C\t4c\t4cc
4D\t4d\t4dd
=\t=\t=
*\t*v\t*v
4E\t4e
4F\t4f
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_multiple_transitions(self):
        """Should handle complex scenario with multiple staff splits and merges."""
        transcription = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
=\t=
*\t*^
4D\t4d\t4dd
=\t=\t=
*\t*v\t*v
4E\t4e
=\t=
*\t*^
4F\t4f\t4ff
=\t=\t=
4G\t4g\t4gg
==\t==\t==
*-\t*-\t*-"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
=\t=
*\t*^
4D\t4d\t4dd
=\t=\t=
*\t*v\t*v
4E\t4e
=\t=
*\t*^
4F\t4f\t4ff
=\t=\t=
*-\t*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_bar_containing_split_marker(self):
        """Should remove bar that contains a staff split marker."""
        transcription = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
4D\t4d
=\t=
4E\t4e
4F\t4f
=\t=
*\t*^
4G\t4g\t4gg
==\t==\t==
*-\t*-\t*-"""

        # After removing the split and its bar, we're back to 2 spines
        # So the bar and terminator should have 2 fields, not 3
        expected = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
4D\t4d
=\t=
4E\t4e
4F\t4f
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_bar_containing_merge_marker(self):
        """Should remove bar that contains staff merge markers."""
        transcription = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4C\t4c\t4cc
4D\t4d\t4dd
=\t=\t=
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
*\t*v\t*v
4G\t4g
==\t==
*-\t*-"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
4C\t4c\t4cc
4D\t4d\t4dd
=\t=\t=
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
*-\t*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_remove_bar_after_staff_split(self):
        """Should properly remove last bar that occurs after a staff split."""
        transcription = """**kern\t**kern
*clefF4\t*clefG2
=\t=
*\t*^
4C\t4c\t4cc
=\t=\t=
4D\t4d\t4dd
==\t==\t==
*-\t*-\t*-"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
=\t=
*\t*^
4C\t4c\t4cc
=\t=\t=
*-\t*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_remove_bar_after_staff_merge(self):
        """Should properly remove last bar that occurs after a staff merge."""
        transcription = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
=\t=\t=
*\t*v\t*v
4C\t4c
=\t=
4D\t4d
==\t==
*-\t*-"""

        expected = """**kern\t**kern\t**kern
*clefF4\t*clefG2\t*clefG2
=\t=\t=
*\t*v\t*v
4C\t4c
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_four_staff_scenario(self):
        """Should handle 4 staves (e.g., piano with both hands split)."""
        transcription = """**kern\t**kern\t**kern\t**kern
*clefF4\t*clefF4\t*clefG2\t*clefG2
4CC\t4C\t4c\t4cc
4DD\t4D\t4d\t4dd
=\t=\t=\t=
4EE\t4E\t4e\t4ee
4FF\t4F\t4f\t4ff
=\t=\t=\t=
4GG\t4G\t4g\t4gg
4AA\t4A\t4a\t4aa
==\t==\t==\t==
*-\t*-\t*-\t*-"""

        expected = """**kern\t**kern\t**kern\t**kern
*clefF4\t*clefF4\t*clefG2\t*clefG2
4CC\t4C\t4c\t4cc
4DD\t4D\t4d\t4dd
=\t=\t=\t=
4EE\t4E\t4e\t4ee
4FF\t4F\t4f\t4ff
=\t=\t=\t=
*-\t*-\t*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_split_after_repeat_bar_terminator_count(self):
        """Should maintain correct terminator count when split occurs after repeat bar.

        This test reproduces a bug where:
        1. A repeat bar (==:|!) is followed by a staff split (*^)
        2. The split creates 3 spines from 2
        3. When removing the last bar, the terminator must have 3 *- markers, not 2
        """
        transcription = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
4D\t4d
==:|!\t==:|!
*^\t*
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
4G\t4g\t4gg
4A\t4a\t4aa
==\t==\t==
*-\t*-\t*-"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
4D\t4d
==:|!\t==:|!
*^\t*
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
*-\t*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_split_without_double_bar_before_terminator(self):
        """Should maintain correct terminator count when split occurs before single bar and terminator.

        This reproduces the exact bug from out.txt where:
        1. A staff split (*^) creates 3 spines from 2
        2. There's a single bar (=) followed directly by a terminator (*-)
        3. The terminator count must match the 3 spines, not the original 2
        """
        transcription = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
4D\t4d
=\t=
*^\t*
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
4G\t4g\t4gg
*-\t*-\t*-"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
4D\t4d
=\t=
*^\t*
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
*-\t*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_split_followed_by_bars_with_wrong_terminator_count(self):
        """Should correct terminator count when it doesn't match spine count after removal.

        This reproduces the exact bug from out.txt where:
        1. Start with 2 spines and terminator has 2 *- markers
        2. A merge and split occur, creating 3 spines
        3. When removing bars, we end up with 3-spine data but 2-spine terminator
        4. The function should update the terminator to have 3 *- markers

        This is the core bug: the terminator count doesn't get updated to match
        the actual spine count after bar removal.
        """
        transcription = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
=\t=
4D\t4d
==:|!\t==:|!
*^\t*
.\t.\t.
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
4G\t4g\t4gg
*-\t*-"""

        expected = """**kern\t**kern
*clefF4\t*clefG2
4C\t4c
=\t=
4D\t4d
==:|!\t==:|!
*^\t*
.\t.\t.
4E\t4e\t4ee
4F\t4f\t4ff
=\t=\t=
*-\t*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_barlines_field_count_after_merge(self):
        """Should ensure barlines have correct field count after spine merges.

        This test ensures that after removing a bar, any subsequent barlines
        have the correct field count matching the current spine count,
        even when there are spine operations (splits/merges) that change
        the spine count throughout the transcription.
        """
        transcription = """**kern\t**kern
*^\t*^
*clefF4\t*clefF4\t*clefG2\t*clefG2
4C\t4CC\t4c\t4cc
4D\t4DD\t4d\t4dd
=\t=\t=\t=
4E\t4EE\t4e\t4ee
*v\t*v\t*\t*
*\t*v\t*v
=\t=
4F\t4f
4G\t4g
*clefG2\t*
=\t=
4a\t4aa
4b\t4bb
=\t=
4cc\t4ccc
==\t==
*-\t*-"""

        # After removing the last bar, the barlines after the merge should have 2 fields
        expected = """**kern\t**kern
*^\t*^
*clefF4\t*clefF4\t*clefG2\t*clefG2
4C\t4CC\t4c\t4cc
4D\t4DD\t4d\t4dd
=\t=\t=\t=
4E\t4EE\t4e\t4ee
*v\t*v\t*\t*
*\t*v\t*v
=\t=
4F\t4f
4G\t4g
*clefG2\t*
=\t=
4a\t4aa
4b\t4bb
=\t=
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_right_facing_repeat_barline(self):
        transcription = """16DJk 16dJk\t.\t48b-
.\t.\t48ff
.\t.\t48bb-J
*\t*v\t*v
=:|!\t=:|!
8EEE-L 8EE-L\t8gL 8b-L 8ee-L
16AAA-Jk 16AA-Jk\t16e-Jk 16a-Jk 16ccJk
16AAA-L 16AA-L\t8.e- 8.a- 8.cc
16E-\t.
16D-J\t.
16CL\t8a-L
16AA-\t.
16E-J[\t16cJk
=!|:\t=!|:
16E-L]\t8fL
16AA\t.
16E-J[\t16cJk
16E-L]\t8e-L
16BB-\t.
16E-J[\t16d-Jk
16E-L]\t8b-L
16FF#\t.
16E-J[\t16B-Jk
*-\t*-"""

        # After removing the last bar, the remaining barline and terminator should
        # have 2 fields to match the merged spine count.
        expected = """16DJk 16dJk\t.\t48b-
.\t.\t48ff
.\t.\t48bb-J
*\t*v\t*v
=:|!\t=:|!
8EEE-L 8EE-L\t8gL 8b-L 8ee-L
16AAA-Jk 16AA-Jk\t16e-Jk 16a-Jk 16ccJk
16AAA-L 16AA-L\t8.e- 8.a- 8.cc
16E-\t.
16D-J\t.
16CL\t8a-L
16AA-\t.
16E-J[\t16cJk
=!|:\t=!|:
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected

    def test_weird_edge_case2(self):
        transcription = """*^\t*^
*\t*^\t*\t*^
4r\t1d-[ 1D-[\t4A-\t4r\t4f\t1dd-
*\t*\t*\t*\t*v\t*v
4cc\t.\t4r\t8aa-L\t4dd- 4aa-
.\t.\t.\t8ddd-J\t.
4b-\t.\t4r\t8ggL\t4dd- 4gg
.\t.\t.\t8ddd-\t.
4b--\t.\t4r\t8gg-\t4dd- 4gg-
.\t.\t.\t8ddd-J\t.
*\t*v\t*v\t*\t*
*\t*\t*v\t*v
=\t=\t=
4a-\t4D-] 4d-]\t4ddd- 4ff 4dd-
4r\t4r\t4r
4D- 4DD-\t4r\t4ddd- 4aa- 4ff 4dd-
4r\t4r\t4r
*v\t*v\t*
==\t==
*met(c)\t*met(c)
4A\t16r
.\t16eL
.\t16f#
.\t16g#J
16r\t4a
16AAL\t.
16BB\t.
16C#J\t.
4D\t16r
.\t16aL
.\t16b
.\t16cc#J
16r\t4dd
16DL\t.
16E\t.
16F#J\t.
*-\t*-"""

        expected = """*^\t*^
*\t*^\t*\t*^
4r\t1d-[ 1D-[\t4A-\t4r\t4f\t1dd-
*\t*\t*\t*\t*v\t*v
4cc\t.\t4r\t8aa-L\t4dd- 4aa-
.\t.\t.\t8ddd-J\t.
4b-\t.\t4r\t8ggL\t4dd- 4gg
.\t.\t.\t8ddd-\t.
4b--\t.\t4r\t8gg-\t4dd- 4gg-
.\t.\t.\t8ddd-J\t.
*\t*v\t*v\t*\t*
*\t*\t*v\t*v
=\t=\t=
4a-\t4D-] 4d-]\t4ddd- 4ff 4dd-
4r\t4r\t4r
4D- 4DD-\t4r\t4ddd- 4aa- 4ff 4dd-
4r\t4r\t4r
*v\t*v\t*
==\t==
*-\t*-"""

        result = remove_bar_from_end(transcription)
        assert result == expected
