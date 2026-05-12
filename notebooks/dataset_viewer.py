"""Interactive dataset viewer for Jupyter notebooks."""

import html

import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from ipywidgets import (
    Button,
    Checkbox,
    Dropdown,
    HBox,
    HTML,
    IntSlider,
    IntText,
    Layout,
    Output,
    VBox,
)


class DatasetViewer:
    """Interactive viewer for browsing HuggingFace datasets with images and transcriptions."""

    def __init__(self, dataset, title: str = "Dataset", enable_filters: bool = False):
        self.dataset = dataset
        self.title = title
        self.enable_filters = enable_filters
        self.current_idx = 0
        self.filtered_indices = list(range(len(dataset)))
        self.show_transcription = False
        self.initial_kern_spine_counts = self._read_dataset_column("initial_kern_spine_count")
        self.svg_system_counts = self._read_dataset_column("svg_system_count")
        self.derived_spine_classes = [
            self._derive_spine_class(count) for count in self.initial_kern_spine_counts
        ]

        # Create widgets
        self.slider = IntSlider(
            min=0,
            max=max(len(self.filtered_indices) - 1, 0),
            step=1,
            value=0,
            description="Match:" if enable_filters else "Index:",
            continuous_update=False,
            layout={"width": "600px"},
        )

        self.prev_btn = Button(description="◄ Previous", button_style="info")
        self.next_btn = Button(description="Next ►", button_style="info")
        self.jump_input = IntText(value=0, description="Jump to:", layout={"width": "150px"})
        self.transcription_checkbox = Checkbox(value=False, description="Show transcription")

        self.spine_class_dropdown = None
        self.svg_system_count_dropdown = None
        self.clear_filters_btn = None
        self.match_summary = None
        if enable_filters:
            self.spine_class_dropdown = Dropdown(
                options=[("All", None), ("1", "1"), ("2", "2"), ("3_plus", "3_plus")],
                value=None,
                description="spine_class:",
                layout={"width": "190px"},
            )
            self.svg_system_count_dropdown = Dropdown(
                options=[("All", None), *self._build_system_count_options()],
                value=None,
                description="svg_system_count:",
                layout={"width": "240px"},
            )
            self.clear_filters_btn = Button(description="Clear filters")
            self.match_summary = HTML()

        self.output = Output()

        # Set up event handlers
        self.slider.observe(self._on_slider_change, names="value")
        self.prev_btn.on_click(self._on_prev)
        self.next_btn.on_click(self._on_next)
        self.jump_input.observe(self._on_jump, names="value")
        self.transcription_checkbox.observe(self._on_transcription_toggle, names="value")
        if self.enable_filters:
            self.spine_class_dropdown.observe(self._on_filter_change, names="value")
            self.svg_system_count_dropdown.observe(self._on_filter_change, names="value")
            self.clear_filters_btn.on_click(self._on_clear_filters)

        self._sync_navigation_state()

    @staticmethod
    def _derive_spine_class(initial_kern_spine_count) -> str:
        count = max(1, int(initial_kern_spine_count or 1))
        if count == 1:
            return "1"
        if count == 2:
            return "2"
        return "3_plus"

    def _read_dataset_column(self, column_name: str) -> list[object | None]:
        if hasattr(self.dataset, "column_names") and column_name in self.dataset.column_names:
            return list(self.dataset[column_name])
        return [example.get(column_name) for example in self.dataset]

    def _build_system_count_options(self) -> list[tuple[str, int]]:
        system_counts = {
            int(system_count)
            for system_count in self.svg_system_counts
            if system_count is not None
        }
        return [(str(count), count) for count in sorted(system_counts)]

    def _compute_filtered_indices(
        self,
        spine_class: str | None = None,
        svg_system_count: int | None = None,
    ) -> list[int]:
        filtered_indices: list[int] = []
        for dataset_idx, (example_spine_class, example_system_count) in enumerate(
            zip(self.derived_spine_classes, self.svg_system_counts, strict=False)
        ):
            if spine_class is not None and example_spine_class != spine_class:
                continue
            if svg_system_count is not None and (
                example_system_count is None or int(example_system_count) != int(svg_system_count)
            ):
                continue
            filtered_indices.append(dataset_idx)
        return filtered_indices

    def _current_dataset_index(self) -> int | None:
        if not self.filtered_indices:
            return None
        return self.filtered_indices[self.current_idx]

    def _current_example(self):
        dataset_idx = self._current_dataset_index()
        if dataset_idx is None:
            return None
        return self.dataset[dataset_idx]

    def _active_filters(self) -> list[str]:
        if not self.enable_filters:
            return []
        active_filters: list[str] = []
        if self.spine_class_dropdown.value is not None:
            active_filters.append(f"spine_class={self.spine_class_dropdown.value}")
        if self.svg_system_count_dropdown.value is not None:
            active_filters.append(f"svg_system_count={self.svg_system_count_dropdown.value}")
        return active_filters

    def _update_match_summary(self) -> None:
        if self.match_summary is None:
            return
        active_filters = self._active_filters()
        filter_text = ""
        if active_filters:
            filter_text = " | " + " | ".join(
                f"<code>{html.escape(active_filter)}</code>" for active_filter in active_filters
            )
        self.match_summary.value = (
            f"<b>{len(self.filtered_indices)} / {len(self.dataset)} samples</b>{filter_text}"
        )

    def _set_navigation_inputs(self) -> None:
        self.slider.unobserve(self._on_slider_change, names="value")
        self.jump_input.unobserve(self._on_jump, names="value")
        try:
            self.slider.max = max(len(self.filtered_indices) - 1, 0)
            self.slider.value = self.current_idx
            self.jump_input.value = self.current_idx
        finally:
            self.slider.observe(self._on_slider_change, names="value")
            self.jump_input.observe(self._on_jump, names="value")

    def _sync_navigation_state(self) -> None:
        if self.filtered_indices and self.current_idx >= len(self.filtered_indices):
            self.current_idx = len(self.filtered_indices) - 1
        if not self.filtered_indices:
            self.current_idx = 0

        self._set_navigation_inputs()
        has_matches = bool(self.filtered_indices)
        self.slider.disabled = not has_matches
        self.jump_input.disabled = not has_matches
        self.prev_btn.disabled = not has_matches or self.current_idx == 0
        self.next_btn.disabled = not has_matches or self.current_idx >= len(self.filtered_indices) - 1
        self._update_match_summary()

    def _refresh_filtered_indices(self, *, reset_position: bool) -> None:
        if self.enable_filters:
            self.filtered_indices = self._compute_filtered_indices(
                spine_class=self.spine_class_dropdown.value,
                svg_system_count=self.svg_system_count_dropdown.value,
            )
        else:
            self.filtered_indices = list(range(len(self.dataset)))

        if reset_position or not self.filtered_indices:
            self.current_idx = 0
        self._sync_navigation_state()

    def _on_slider_change(self, change):
        self.current_idx = change["new"]
        self._sync_navigation_state()
        self.display_current()

    def _on_prev(self, _btn):
        if self.current_idx > 0:
            self.slider.value = self.current_idx - 1

    def _on_next(self, _btn):
        if self.current_idx < len(self.filtered_indices) - 1:
            self.slider.value = self.current_idx + 1

    def _on_jump(self, change):
        idx = change["new"]
        if 0 <= idx < len(self.filtered_indices):
            self.slider.value = idx

    def _on_transcription_toggle(self, change):
        self.show_transcription = change["new"]
        self.display_current()

    def _on_filter_change(self, _change):
        self._refresh_filtered_indices(reset_position=True)
        self.display_current()

    def _on_clear_filters(self, _btn):
        self.spine_class_dropdown.unobserve(self._on_filter_change, names="value")
        self.svg_system_count_dropdown.unobserve(self._on_filter_change, names="value")
        try:
            self.spine_class_dropdown.value = None
            self.svg_system_count_dropdown.value = None
        finally:
            self.spine_class_dropdown.observe(self._on_filter_change, names="value")
            self.svg_system_count_dropdown.observe(self._on_filter_change, names="value")
        self._refresh_filtered_indices(reset_position=True)
        self.display_current()

    def display_current(self):
        with self.output:
            clear_output(wait=True)

            example = self._current_example()
            if example is None:
                display(HTML(f"<b>{self.title} | No samples match current filters.</b>"))
                return

            dataset_idx = self._current_dataset_index()

            # Get the PIL image directly
            img = example["image"]

            # Build title
            title_parts = [self.title]
            if self.enable_filters:
                title_parts.extend(
                    [
                        f"Match {self.current_idx + 1} / {len(self.filtered_indices)}",
                        f"Dataset index {dataset_idx}",
                    ]
                )
            else:
                title_parts.append(f"Image {dataset_idx + 1} / {len(self.dataset)}")
            title_parts.append(f"Size: {img.size}")
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
        nav_buttons = HBox([self.prev_btn, self.next_btn, self.jump_input])
        control_items = []
        if self.enable_filters:
            filter_controls = HBox(
                [
                    self.spine_class_dropdown,
                    self.svg_system_count_dropdown,
                    self.clear_filters_btn,
                    self.match_summary,
                ],
                layout=Layout(align_items="center", flex_flow="row wrap", gap="8px"),
            )
            control_items.append(filter_controls)
        control_items.extend(
            [
                self.slider,
                nav_buttons,
                self.transcription_checkbox,
            ]
        )
        controls = VBox(control_items)

        display(controls, self.output)
        self.display_current()
