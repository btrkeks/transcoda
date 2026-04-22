"""Interactive dataset viewer for Jupyter notebooks."""

import html

import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from ipywidgets import Button, Checkbox, HBox, HTML, IntSlider, IntText, Layout, Output, VBox


class DatasetViewer:
    """Interactive viewer for browsing HuggingFace datasets with images and transcriptions."""

    def __init__(self, dataset, title: str = "Dataset"):
        self.dataset = dataset
        self.title = title
        self.current_idx = 0
        self.show_transcription = False

        # Create widgets
        self.slider = IntSlider(
            min=0,
            max=len(dataset) - 1,
            step=1,
            value=0,
            description="Index:",
            continuous_update=False,
            layout={"width": "600px"},
        )

        self.prev_btn = Button(description="◄ Previous", button_style="info")
        self.next_btn = Button(description="Next ►", button_style="info")
        self.jump_input = IntText(value=0, description="Jump to:", layout={"width": "150px"})
        self.transcription_checkbox = Checkbox(value=False, description="Show transcription")

        self.output = Output()

        # Set up event handlers
        self.slider.observe(self._on_slider_change, names="value")
        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)
        self.jump_input.observe(self._on_jump, names="value")
        self.transcription_checkbox.observe(self._on_transcription_toggle, names="value")

    def _on_slider_change(self, change):
        self.current_idx = change["new"]
        self.display_current()

    def _on_prev(self, _btn):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.slider.value = self.current_idx

    def _on_next(self, _btn):
        if self.current_idx < len(self.dataset) - 1:
            self.current_idx += 1
            self.slider.value = self.current_idx

    def _on_jump(self, change):
        idx = change["new"]
        if 0 <= idx < len(self.dataset):
            self.current_idx = idx
            self.slider.value = idx

    def _on_transcription_toggle(self, change):
        self.show_transcription = change["new"]
        self.display_current()

    def display_current(self):
        with self.output:
            clear_output(wait=True)

            # Get the example
            example = self.dataset[self.current_idx]

            # Get the PIL image directly
            img = example["image"]

            # Build title
            title_parts = [
                self.title,
                f"Image {self.current_idx + 1} / {len(self.dataset)}",
                f"Size: {img.size}",
            ]
            system_count = example.get("svg_system_count")
            if system_count is not None:
                title_parts.append(f"Systems: {system_count}")
            title_text = " | ".join(title_parts)
            display(HTML(f"<b>{title_text}</b>"))

            if "source" in example:
                display(HTML(f"<code>Source: {html.escape(str(example['source']))}</code>"))

            image_aspect = img.size[0] / img.size[1]  # width / height
            panel_height_px = 920
            image_height_px = panel_height_px
            image_width_px = max(360, int(image_height_px * image_aspect))

            image_output = Output(
                layout=Layout(width=f"{image_width_px}px", flex="0 0 auto")
            )
            with image_output:
                # Keep image height fixed while allowing width to follow image aspect ratio.
                figure_height = image_height_px / 100
                figure_width = image_width_px / 100
                _fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=100)
                ax.imshow(img)
                ax.axis("off")
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.show()

            if self.show_transcription:
                transcription_text = html.escape(str(example.get("transcription", "")))
                transcription_panel = HTML(
                    value=(
                        f"<div style='height:{panel_height_px}px; overflow:auto; border:1px solid #ddd; "
                        "border-radius:6px; padding:8px; background:#fafafa;'>"
                        "<div style='font-weight:600; margin-bottom:8px;'>Transcription</div>"
                        f"<pre style='margin:0; white-space:pre-wrap;'>{transcription_text}</pre>"
                        "</div>"
                    ),
                    layout=Layout(width="420px", flex="0 0 auto"),
                )
                display(
                    HBox(
                        [image_output, transcription_panel],
                        layout=Layout(align_items="stretch"),
                    )
                )
            else:
                display(image_output)

    def show(self):
        """Display the viewer widget."""
        # Layout
        nav_buttons = HBox([self.prev_btn, self.next_btn, self.jump_input])
        controls = VBox(
            [
                self.slider,
                nav_buttons,
                self.transcription_checkbox,
            ]
        )

        display(controls, self.output)
        self.display_current()
