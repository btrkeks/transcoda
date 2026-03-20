"""Model wrappers for unified inference interface.

Provides a protocol-based abstraction over different model architectures,
enabling fair comparison between the user's SMT model and PRAIG models.
"""

import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image
from rich.console import Console

from src.data.preprocessing import preprocess_pil_image
from src.model.generation_policy import build_generate_kwargs, settings_from_decoding_spec
from src.model.checkpoint_loader import load_model_from_checkpoint

console = Console()


@runtime_checkable
class ModelWrapper(Protocol):
    """Protocol for unified model inference interface.

    All model wrappers must implement this interface to be used with
    the EvaluationHarness.
    """

    @property
    def name(self) -> str:
        """Human-readable model name for output."""
        ...

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to model-specific tensor format.

        Args:
            image: PIL Image in RGB mode

        Returns:
            Model-specific tensor format ready for inference
        """
        ...

    def predict(self, image_tensor: torch.Tensor) -> str:
        """Run inference and return decoded **kern string.

        Args:
            image_tensor: Preprocessed image tensor (single sample, no batch dim)

        Returns:
            Decoded **kern string
        """
        ...


class UserSMTWrapper:
    """Wrapper for user's SMTModelForCausalLM.

    Uses the local model implementation with HuggingFace generate() method.
    Expects RGB uint8 input tensors.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        image_width: int | None = None,
    ):
        """Initialize wrapper by loading model from checkpoint.

        Args:
            checkpoint_path: Path to PyTorch Lightning .ckpt file
            device: Device for inference
            image_width: Target image width in pixels. If None, uses value from checkpoint.
        """
        self.device = device
        loaded = load_model_from_checkpoint(checkpoint_path, device)
        self.model = loaded.model
        self.i2w = loaded.i2w
        self.pad_token_id = loaded.pad_token_id
        self.artifact = loaded.artifact
        self._generation_settings = settings_from_decoding_spec(self.artifact.decoding)
        self._generation_max_length = self.artifact.decoding.max_len or self.model.config.maxlen

        # Get image_width from artifact if not specified
        if image_width is None:
            self.image_width = loaded.image_width
        else:
            self.image_width = image_width

        self.fixed_size = loaded.fixed_size

        self._name = "User SMT"

    @property
    def name(self) -> str:
        return self._name

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to normalized RGB tensor.

        Args:
            image: PIL Image (any mode, will be converted to RGB)

        Returns:
            Tensor of shape (3, H, W) as float32 in range [-1, 1]
        """
        tensor, _ = preprocess_pil_image(
            image=image,
            image_width=self.image_width,
            fixed_size=self.fixed_size,
        )
        return tensor

    def predict(self, image_tensor: torch.Tensor) -> str:
        """Run inference using HuggingFace generate().

        Args:
            image_tensor: RGB tensor (3, H, W) as float32 in [-1, 1]

        Returns:
            Decoded **kern string (without Humdrum header)
        """
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        image_sizes = torch.tensor(
            [(image_tensor.shape[-2], image_tensor.shape[-1])],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            generate_kwargs = build_generate_kwargs(
                pixel_values=image_tensor,
                image_sizes=image_sizes,
                max_length=self._generation_max_length,
                settings=self._generation_settings,
            )
            pred_ids = self.model.generate(**generate_kwargs)

        # Decode to string (without header for fair comparison)
        return self._token_ids_to_string(pred_ids[0].tolist())

    def batch_predict(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> list[str]:
        """Run batched inference.

        Args:
            pixel_values: Batched images (B, C, H, W) as uint8
            image_sizes: Original sizes (B, 2) as (H, W) per sample

        Returns:
            List of decoded **kern strings (one per sample)
        """
        # Move to device
        pixel_values = pixel_values.to(self.device)
        image_sizes = image_sizes.to(self.device)

        with torch.no_grad():
            generate_kwargs = build_generate_kwargs(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_length=self._generation_max_length,
                settings=self._generation_settings,
            )
            pred_ids = self.model.generate(**generate_kwargs)

        # Decode each sequence in the batch
        return [self._token_ids_to_string(seq.tolist()) for seq in pred_ids]

    def _token_ids_to_string(self, token_ids: list[int]) -> str:
        """Convert token IDs to **kern string without header."""
        tokens = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                break
            token = self.i2w.get(token_id, "")
            if token in ("<bos>", "<eos>"):
                continue
            tokens.append(token)

        return "".join(tokens)


class PRAIGModelWrapper:
    """Wrapper for PRAIG's SMT models from HuggingFace.

    Uses PRAIG's model implementation with their custom predict() method.
    Expects grayscale [0,1] float input tensors.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        praig_module_path: str = "models/external/praig-smt",
    ):
        """Initialize wrapper by loading PRAIG model from HuggingFace.

        Args:
            model_id: HuggingFace model ID (e.g., "PRAIG/smt-fp-grandstaff")
            device: Device for inference
            praig_module_path: Path to PRAIG SMT git submodule
        """
        # Add PRAIG module to path for imports
        praig_path = Path(praig_module_path).resolve()
        if str(praig_path) not in sys.path:
            sys.path.insert(0, str(praig_path))

        # Import PRAIG's model class
        from smt_model import SMTModelForCausalLM as PRAIGModel

        console.print(f"[cyan]Loading PRAIG model:[/cyan] {model_id}")
        self.model = PRAIGModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.model_id = model_id
        self._name = model_id.split("/")[-1]

    @property
    def name(self) -> str:
        return self._name

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to grayscale [0,1] float tensor.

        PRAIG models expect single-channel grayscale input normalized to [0,1].

        Args:
            image: PIL Image (any mode, will be converted to grayscale)

        Returns:
            Tensor of shape (1, H, W) as float32 in [0,1]
        """
        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")

        # Convert to tensor and normalize to [0,1]
        tensor = torch.from_numpy(np.array(image)).float() / 255.0

        # Add channel dimension: (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)

        return tensor

    def predict(self, image_tensor: torch.Tensor) -> str:
        """Run inference using PRAIG's predict() method.

        Args:
            image_tensor: Grayscale tensor (1, H, W) as float32

        Returns:
            Decoded **kern string with special tokens replaced
        """
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        image_sizes = torch.tensor(
            [(image_tensor.shape[-2], image_tensor.shape[-1])],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            text_sequence, _ = self.model.predict(
                image_tensor,
                image_sizes=image_sizes,
                convert_to_str=True,
            )

        # Join tokens and replace special tokens
        kern_str = self._decode_prediction_text(text_sequence)

        return kern_str

    def batch_predict(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> list[str]:
        pixel_values = pixel_values.to(self.device)
        image_sizes = image_sizes.to(self.device)

        with torch.no_grad():
            text_sequences, _ = self.model.predict(
                pixel_values,
                image_sizes=image_sizes,
                convert_to_str=True,
            )

        if pixel_values.shape[0] == 1:
            text_sequences = [text_sequences]

        return [self._decode_prediction_text(sequence) for sequence in text_sequences]

    @staticmethod
    def _decode_prediction_text(text_sequence: list[str]) -> str:
        kern_str = "".join(text_sequence)
        kern_str = kern_str.replace("<b>", "\n")
        kern_str = kern_str.replace("<s>", " ")
        kern_str = kern_str.replace("<t>", "\t")
        return kern_str
