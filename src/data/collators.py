import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.preprocessing import NORMALIZED_PAD_VALUE


class Img2SeqCollator:
    def __init__(
        self,
        tokenizer,
        image_pad_value=NORMALIZED_PAD_VALUE,
        fixed_height: int | None = 1485,
        fixed_width: int | None = 1050,
        enforce_fixed_size: bool = True,
        fixed_decoder_length: int | None = None,
    ):
        self.tok = tokenizer
        self.image_pad_value = image_pad_value
        self.fixed_height = fixed_height
        self.fixed_width = fixed_width
        self.enforce_fixed_size = enforce_fixed_size
        self.fixed_decoder_length = fixed_decoder_length
        if self.enforce_fixed_size and (self.fixed_height is None or self.fixed_width is None):
            raise ValueError(
                "Img2SeqCollator requires fixed_height and fixed_width when enforce_fixed_size=True."
            )
        if self.fixed_decoder_length is not None and self.fixed_decoder_length <= 0:
            raise ValueError("Img2SeqCollator fixed_decoder_length must be > 0 when provided.")

    def __call__(self, batch):
        # --- labels ---
        labels = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
        if self.fixed_decoder_length is None:
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.tok.pad_token_id)
            decoder_attn_mask = labels_padded != self.tok.pad_token_id
            labels_padded[~decoder_attn_mask] = -100
        else:
            max_len = int(self.fixed_decoder_length)
            batch_size = len(labels)
            labels_padded = torch.full(
                (batch_size, max_len),
                fill_value=self.tok.pad_token_id,
                dtype=torch.long,
            )
            for idx, label in enumerate(labels):
                seq_len = int(label.numel())
                if seq_len > max_len:
                    sample = batch[idx]
                    sample_id = sample.get("sample_id", idx)
                    raise ValueError(
                        "Img2SeqCollator expected decoder length <= "
                        f"{max_len} but got {seq_len} for sample_id={sample_id} "
                        f"at batch_index={idx}."
                    )
                labels_padded[idx, :seq_len] = label

            decoder_attn_mask = labels_padded != self.tok.pad_token_id
            labels_padded[~decoder_attn_mask] = -100

        # --- images (C,H,W) ---
        imgs = [b["pixel_values"] for b in batch]  # tensors C×H×W
        if self.enforce_fixed_size:
            expected_h = int(self.fixed_height)
            expected_w = int(self.fixed_width)
            for idx, sample in enumerate(batch):
                tensor = sample["pixel_values"]
                observed_h, observed_w = int(tensor.shape[1]), int(tensor.shape[2])
                if observed_h != expected_h or observed_w != expected_w:
                    sample_id = sample.get("sample_id", idx)
                    raise ValueError(
                        "Img2SeqCollator expected fixed image size "
                        f"(H, W)=({expected_h}, {expected_w}) but got "
                        f"({observed_h}, {observed_w}) for sample_id={sample_id} "
                        f"at batch_index={idx}."
                    )

            pixel_values = torch.stack(imgs, dim=0)  # B×C×H×W
            image_sizes = torch.full((len(imgs), 2), 0, dtype=torch.long)
            image_sizes[:, 0] = expected_h
            image_sizes[:, 1] = expected_w
        else:
            Hmax = max(t.shape[1] for t in imgs)
            Wmax = max(t.shape[2] for t in imgs)
            out = []
            for t in imgs:
                pad_h = Hmax - t.shape[1]
                pad_w = Wmax - t.shape[2]
                # pad (left,right,top,bottom) = (0,w,0,h)
                padded = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=self.image_pad_value)
                out.append(padded)
            pixel_values = torch.stack(out, dim=0)  # B×C×Hmax×Wmax
            image_sizes = torch.tensor(
                [(t.shape[1], t.shape[2]) for t in imgs], dtype=torch.long
            )  # (B,2)

        result = {
            "pixel_values": pixel_values,
            "labels": labels_padded,
            "decoder_attention_mask": decoder_attn_mask,
            "image_sizes": image_sizes,
        }

        if "sample_id" in batch[0]:
            result["sample_ids"] = torch.tensor([b["sample_id"] for b in batch], dtype=torch.long)

        # Pass through source field for per-source metric tracking (if present)
        if "source" in batch[0]:
            result["source"] = [b["source"] for b in batch]

        return result
