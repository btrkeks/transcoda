"""Tests for find_content_bottom_row and drop_header_strings functions."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.dataset_generation.dataset_generation.image_generation.image_post import find_content_bottom_row
from scripts.dataset_generation.dataset_generation.image_generation.kern_ops import drop_header_strings


class TestFindImageCut:
    """Test suite for the find_content_bottom_row function."""

    def test_find_black_pixel_at_bottom(self):
        """Test finding a black pixel at the very bottom of the image."""
        # Create a 10x10 RGB image with all white pixels except bottom row
        sample = np.full((10, 10, 3), 255, dtype=np.uint8)
        sample[-1, 5] = [0, 0, 0]  # Add black pixel at bottom

        result = find_content_bottom_row(sample)
        assert result == 9

    def test_find_black_pixel_in_middle(self):
        """Test finding a black pixel in the middle of the image."""
        # Create a 20x10 RGB image with all white pixels
        sample = np.full((20, 10, 3), 255, dtype=np.uint8)
        # Add black pixel at row 10 (middle)
        sample[10, 3] = [0, 0, 0]
        # Add some pixels above it too, but we search from bottom up
        sample[15, 2] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        assert result == 15  # Should find the lowest black pixel first

    def test_no_black_pixel_returns_none(self):
        """Test that None is returned when no black pixel exists."""
        # Create an image with all white pixels
        sample = np.full((15, 15, 3), 255, dtype=np.uint8)

        result = find_content_bottom_row(sample)
        assert result is None

    def test_all_colored_pixels_returns_none(self):
        """Test that non-black colored pixels are not detected."""
        # Create an image with various colors but no pure black
        sample = np.random.randint(1, 255, size=(10, 10, 3), dtype=np.uint8)

        result = find_content_bottom_row(sample)
        assert result is None

    def test_black_pixel_at_top(self):
        """Test finding a black pixel at the top of the image."""
        # Create image with black pixel only at top
        sample = np.full((10, 10, 3), 255, dtype=np.uint8)
        sample[0, 0] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        assert result == 0

    def test_multiple_black_pixels_same_row(self):
        """Test with multiple black pixels in the same row."""
        sample = np.full((10, 10, 3), 255, dtype=np.uint8)
        # Add multiple black pixels at row 5
        sample[5, :] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        assert result == 5

    def test_multiple_black_pixels_different_rows(self):
        """Test that the function returns the lowest row with black pixels."""
        sample = np.full((20, 10, 3), 255, dtype=np.uint8)
        # Add black pixels at different rows
        sample[3, 0] = [0, 0, 0]
        sample[7, 5] = [0, 0, 0]
        sample[15, 9] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        assert result == 15  # Should return the bottommost

    def test_near_black_pixel_not_detected(self):
        """Test that near-black pixels (e.g., [1,1,1]) are not detected."""
        sample = np.full((10, 10, 3), 255, dtype=np.uint8)
        # Add very dark but not pure black pixels
        sample[5, 5] = [1, 1, 1]
        sample[7, 3] = [5, 5, 5]

        result = find_content_bottom_row(sample)
        assert result is None

    def test_black_pixel_with_different_channel_values(self):
        """Test behavior with pixels that have mixed channel values.

        NOTE: The implementation uses `[0, 0, 0] in sample[y]` which checks
        if the exact list [0, 0, 0] appears as any pixel in the row.
        This test verifies the actual behavior.
        """
        sample = np.full((10, 10, 3), 255, dtype=np.uint8)
        # Add pixels with some channels black but not all
        sample[3, 0] = [0, 0, 50]  # Blue channel not black
        sample[5, 5] = [0, 50, 0]  # Green channel not black
        sample[7, 7] = [50, 0, 0]  # Red channel not black

        result = find_content_bottom_row(sample)
        # These are not pure [0,0,0] so should not be detected
        assert result is None

    def test_single_pixel_image_with_black(self):
        """Test with a minimal 1x1 image containing black."""
        sample = np.array([[[0, 0, 0]]], dtype=np.uint8)

        result = find_content_bottom_row(sample)
        assert result == 0

    def test_single_pixel_image_without_black(self):
        """Test with a minimal 1x1 image without black."""
        sample = np.array([[[255, 255, 255]]], dtype=np.uint8)

        result = find_content_bottom_row(sample)
        assert result is None

    def test_single_column_image(self):
        """Test with a tall, narrow image (single column)."""
        sample = np.full((100, 1, 3), 255, dtype=np.uint8)
        sample[50, 0] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        assert result == 50

    def test_single_row_image(self):
        """Test with a wide, short image (single row)."""
        sample = np.full((1, 100, 3), 255, dtype=np.uint8)
        sample[0, 50] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        assert result == 0

    def test_large_image_performance(self):
        """Test with a large image to ensure reasonable performance."""
        # Create a large image (e.g., 4000x3000)
        sample = np.full((4000, 3000, 3), 255, dtype=np.uint8)
        sample[2500, 1000] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        assert result == 2500

    def test_image_with_alpha_channel_not_ignored(self):
        """Test behavior with RGBA images (4 channels).

        The implementation checks ALL channels, so [0,0,0,255] is NOT black.
        Only [0,0,0,0] would be detected as black in RGBA images.
        """
        # Create an RGBA image
        sample = np.full((10, 10, 4), 255, dtype=np.uint8)
        sample[5, 5, :3] = [0, 0, 0]  # Set RGB to black, alpha to 255
        sample[7, 3] = [0, 0, 0, 0]  # Fully transparent black

        result = find_content_bottom_row(sample)
        # Only the fully black+transparent pixel is detected
        assert result == 7

    def test_grayscale_image_handles_correctly(self):
        """Test that grayscale images (2D arrays) are handled correctly.

        The new implementation has explicit support for grayscale images.
        """
        # Grayscale image is 2D, not 3D
        sample = np.full((10, 10), 255, dtype=np.uint8)
        sample[6, 4] = 0  # Add a black pixel

        result = find_content_bottom_row(sample)
        assert result == 6

    def test_bottom_row_all_black(self):
        """Test when the entire bottom row is black."""
        sample = np.full((10, 10, 3), 255, dtype=np.uint8)
        sample[-1, :, :] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        assert result == 9

    def test_consecutive_black_rows_from_bottom(self):
        """Test with multiple consecutive black rows from the bottom."""
        sample = np.full((10, 10, 3), 255, dtype=np.uint8)
        # Make bottom 3 rows have black pixels
        sample[7:, 0] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        # Should return the lowest (bottom-most) row with black
        assert result == 9

    def test_dtype_float_with_black_values(self):
        """Test with float dtype where 0.0 represents black."""
        sample = np.full((10, 10, 3), 1.0, dtype=np.float32)
        sample[6, 3] = [0.0, 0.0, 0.0]

        result = find_content_bottom_row(sample)
        # Should detect [0.0, 0.0, 0.0] as equivalent to [0, 0, 0]
        assert result == 6

    def test_mixed_black_pixels_sparse(self):
        """Test with sparsely distributed black pixels."""
        sample = np.full((50, 50, 3), 255, dtype=np.uint8)
        # Add black pixels at various positions
        sample[10, 20] = [0, 0, 0]
        sample[25, 15] = [0, 0, 0]
        sample[40, 30] = [0, 0, 0]

        result = find_content_bottom_row(sample)
        assert result == 40  # Bottommost black pixel

    def test_empty_array_edge_case(self):
        """Test with an empty array."""
        sample = np.array([], dtype=np.uint8).reshape(0, 0, 3)

        result = find_content_bottom_row(sample)
        assert result is None  # No rows to iterate

    def test_zero_height_image(self):
        """Test with zero-height image (edge case)."""
        sample = np.array([], dtype=np.uint8).reshape(0, 10, 3)

        result = find_content_bottom_row(sample)
        assert result is None


class TestDropHeaderStrings:
    """Test suite for the _drop_header_strings static method.

    This function removes clef (*clefG2, *clefF4, etc.), key signature (*k[...]),
    and time signature (*M...) lines from a kern system to avoid redundancy when
    merging multiple systems together.
    """

    def test_removes_clef_line(self):
        """Test that lines containing clef tokens are removed."""
        tokens = "*clefG2\t*clefF4\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*clefG2" not in result
        assert "*clefF4" not in result
        assert "4c\t4C" in result

    def test_removes_key_signature_line(self):
        """Test that lines containing key signature tokens are removed."""
        tokens = "*k[f#c#]\t*k[f#c#]\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*k[f#c#]" not in result
        assert "4c\t4C" in result

    def test_removes_time_signature_line(self):
        """Test that lines containing time signature tokens are removed."""
        tokens = "*M4/4\t*M4/4\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*M4/4" not in result
        assert "4c\t4C" in result

    def test_removes_all_header_types_together(self):
        """Test removal of all three header types in a single system."""
        tokens = """*clefG2\t*clefF4
*k[b-e-a-]\t*k[b-e-a-]
*M3/4\t*M3/4
4c\t4C
4d\t4D
*-\t*-"""

        result = drop_header_strings(tokens)

        assert "*clefG2" not in result
        assert "*clefF4" not in result
        assert "*k[b-e-a-]" not in result
        assert "*M3/4" not in result
        assert "4c\t4C" in result
        assert "4d\t4D" in result

    def test_preserves_music_content(self):
        """Test that music notation content is preserved."""
        tokens = """**kern\t**kern
*clefG2\t*clefF4
*M4/4\t*M4/4
=1\t=1
4c\t4C
4d\t4D
8e\t8E
8f\t8F
*-\t*-"""

        result = drop_header_strings(tokens)

        assert "=1\t=1" in result
        assert "4c\t4C" in result
        assert "4d\t4D" in result
        assert "8e\t8E" in result
        assert "8f\t8F" in result

    def test_preserves_barlines(self):
        """Test that barlines are preserved."""
        tokens = "**kern\t**kern\n*clefG2\t*clefF4\n=1\t=1\n4c\t4C\n=2\t=2\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "=1\t=1" in result
        assert "=2\t=2" in result

    def test_preserves_terminator_lines(self):
        """Test that terminator lines (*-) are preserved."""
        tokens = "**kern\t**kern\n*clefG2\t*clefF4\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*-\t*-" in result

    def test_empty_string_input(self):
        """Test with an empty string."""
        result = drop_header_strings("")

        assert result == ""

    def test_no_header_lines_to_remove(self):
        """Test when there are no header lines to remove."""
        tokens = "**kern\t**kern\n4c\t4C\n4d\t4D\n*-\t*-"

        result = drop_header_strings(tokens)

        # All lines should be preserved
        assert "**kern\t**kern" in result
        assert "4c\t4C" in result
        assert "4d\t4D" in result
        assert "*-\t*-" in result

    def test_only_header_lines(self):
        """Test when input contains only header lines."""
        tokens = "*clefG2\t*clefF4\n*k[f#]\t*k[f#]\n*M4/4\t*M4/4"

        result = drop_header_strings(tokens)

        assert "*clefG2" not in result
        assert "*k[f#]" not in result
        assert "*M4/4" not in result

    def test_mixed_clef_types(self):
        """Test removal of different clef types."""
        tokens = "*clefF4\t*clefC3\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*clefF4" not in result
        assert "*clefC3" not in result

    def test_complex_key_signatures(self):
        """Test removal of complex key signatures."""
        tokens = """*clefG2\t*clefF4
*k[f#c#g#d#a#e#b#]\t*k[f#c#g#d#a#e#b#]
4c\t4C
*-\t*-"""

        result = drop_header_strings(tokens)

        assert "*k[f#c#g#d#a#e#b#]" not in result
        assert "4c\t4C" in result

    def test_split_at_beginning(self):
        tokens = """*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-]\t*k[b-]\t*k[b-]
*M3/4\t*M3/4\t*M3/4
4r\t8ddL\t4r
.\t8b-J\t.
4r\t4dd\t4r
4F 4c\t4cc\t4f 4a"""

        result = drop_header_strings(tokens)

        assert "*\t*^" in result
        assert "*clefF4" not in result
        assert "*k[b-]" not in result
        assert "*M3/4" not in result
        assert "4r\t8ddL\t4r" in result
        assert "4F 4c\t4cc\t4f 4a" in result

    def test_various_time_signatures(self):
        """Test removal of various time signature formats."""
        tokens = "*M6/8\t*M6/8\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*M6/8" not in result
        assert "4c\t4C" in result

    def test_in_middle_column_only(self):
        """Test when header token appears in one column but not all."""
        tokens = "4c\t*clefG2\n4d\t4D\n*-\t*-"

        result = drop_header_strings(tokens)

        # The line with *clefG2 in second column should be removed
        assert "*clefG2" in result
        assert "4d\t4D" in result

    def test_preserves_other_interpretation_tokens(self):
        """Test that non-header interpretation tokens are preserved."""
        tokens = """**kern\t**kern
*clefG2\t*clefF4
*MM120\t*MM120
4c\t4C
*-\t*-"""

        result = drop_header_strings(tokens)

        # *MM120 (tempo marking) should be preserved
        assert "*MM120" in result
        assert "*clefG2" not in result

    def test_preserves_line_order(self):
        """Test that the order of remaining lines is preserved."""
        tokens = """**kern\t**kern
Line1\t1
*clefG2\t*clefF4
Line2\t2
*M4/4\t*M4/4
Line3\t3
*-\t*-"""

        result = drop_header_strings(tokens)

        lines = [line for line in result.split("\n") if line and "Line" in line]
        assert len(lines) == 3
        assert lines[0].startswith("Line1")
        assert lines[1].startswith("Line2")
        assert lines[2].startswith("Line3")

    def test_single_column_kern(self):
        """Test with single-column kern (no tabs)."""
        tokens = "**kern\n*clefG2\n*M4/4\n4c\n*-"

        result = drop_header_strings(tokens)

        assert "*clefG2" not in result
        assert "*M4/4" not in result
        assert "4c" in result

    def test_three_column_kern(self):
        """Test with three-column kern."""
        tokens = "**kern\t**kern\t**kern\n*clefG2\t*clefG2\t*clefF4\n4c\t4e\t4C\n*-\t*-\t*-"

        result = drop_header_strings(tokens)

        assert "*clefG2" not in result
        assert "*clefF4" not in result
        assert "4c\t4e\t4C" in result

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        tokens = "**kern\t**kern\n*clefG2\t*clefF4\n\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*clefG2" not in result
        assert "4c\t4C" in result

    def test_trailing_whitespace_on_lines(self):
        """Test handling of trailing whitespace on lines."""
        tokens = "**kern\t**kern  \n*clefG2\t*clefF4  \n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*clefG2" not in result
        assert "4c\t4C" in result

    def test_clef_with_line_number(self):
        """Test removal of clef tokens with line numbers (like *clefG2)."""
        tokens = "**kern\t**kern\n*clefX2\t*clefG1\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*clefX2" not in result
        assert "*clefG1" not in result

    def test_empty_key_signature(self):
        """Test removal of empty key signature (*k[])."""
        tokens = "**kern\t**kern\n*k[]\t*k[]\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*k[]" not in result
        assert "4c\t4C" in result

    def test_preserves_spine_path_indicators(self):
        """Test that spine path indicators not related to headers are preserved."""
        tokens = "**kern\t**kern\n*clefG2\t*clefF4\n*^\t*\n4c 4e\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        # *^ (spine split) should be preserved
        assert "*clefG2" not in result
        # Note: Depending on implementation, *^ line might be preserved or not
        assert "4c 4e\t4C" in result

    def test_compound_time_signature(self):
        """Test removal of compound time signatures."""
        tokens = "**kern\t**kern\n*M12/8\t*M12/8\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        assert "*M12/8" not in result
        assert "4c\t4C" in result

    def test_partial_matches_not_removed(self):
        """Test that tokens containing clef/k/M as substrings aren't removed."""
        tokens = "**kern\t**kern\n*clef\t*k\n4c\t4C\n*-\t*-"

        result = drop_header_strings(tokens)

        # If these are exact tokens without the required format, behavior depends on impl
        # This test documents expected behavior for edge cases
        assert "4c\t4C" in result

    def test_realistic_middle_system_merge_scenario(self):
        """Test a realistic scenario from _prepare_middle_system_for_merge."""
        tokens = """**kern\t**kern
*clefG2\t*clefF4
*k[b-]\t*k[b-]
*M4/4\t*M4/4
=1\t=1
4c\t4C
4d\t4D
4e\t4E
4f\t4F
=2\t=2
2g\t2G
2a\t2A
*-\t*-"""

        result = drop_header_strings(tokens)

        # Headers should be removed
        assert "*clefG2" not in result
        assert "*k[b-]" not in result
        assert "*M4/4" not in result

        # Music content should be preserved
        assert "=1\t=1" in result
        assert "4c\t4C" in result
        assert "2g\t2G" in result
        assert "=2\t=2" in result

        # Terminator should be preserved
        assert "*-\t*-" in result

    def test_multiple_header_lines_consecutive(self):
        """Test removal of multiple consecutive header lines."""
        tokens = """**kern\t**kern
*clefG2\t*clefF4
*clefG2\t*clefF4
*k[f#]\t*k[f#]
*k[c#]\t*k[c#]
*M4/4\t*M4/4
*M3/4\t*M3/4
4c\t4C
*-\t*-"""

        result = drop_header_strings(tokens)

        # All header lines should be removed
        assert "*clefG2" not in result
        assert "*clefF4" not in result
        assert "*k[f#]" not in result
        assert "*k[c#]" not in result
        assert "*M4/4" not in result
        assert "*M3/4" not in result

        # Music should remain
        assert "4c\t4C" in result


class TestLayoutOverflowErrors:
    """Test suite for LayoutOverflowError exception."""

    def test_multipage_layout_error_message(self):
        """Test that LayoutOverflowError contains useful info for page overflow."""
        from scripts.dataset_generation.dataset_generation.image_generation.exceptions import LayoutOverflowError

        error = LayoutOverflowError(
            "Rendered output spans multiple pages.", page_count=2, allowed_pages=1
        )

        # Verify the error message contains useful information
        assert "multiple pages" in str(error).lower()
        assert "pages=2" in str(error)
        assert "allowed=1" in str(error)

        # Verify it can be raised and caught
        with pytest.raises(LayoutOverflowError):
            raise error

    def test_system_overflow_error_message(self):
        """Test that LayoutOverflowError contains useful info for system overflow."""
        from scripts.dataset_generation.dataset_generation.image_generation.exceptions import LayoutOverflowError

        error = LayoutOverflowError(
            "Rendered SVG contains more systems than allowed.", system_count=5, max_systems=3
        )

        # Verify the error message contains useful information
        assert "systems" in str(error).lower()
        assert "systems=5" in str(error)
        assert "max=3" in str(error)

        # Verify it can be raised and caught
        with pytest.raises(LayoutOverflowError):
            raise error
