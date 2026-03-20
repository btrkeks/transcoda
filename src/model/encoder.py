"""
Encoder loading factory for SMT model.

This module provides a flexible factory pattern for loading different vision encoder
backbones (ConvNeXt, ResNet, Swin, etc.) through a unified interface.
"""

import torch.nn as nn
from loguru import logger
from transformers import AutoModel

from .configuration_smt import SMTConfig


class EncoderLoader:
    """Factory class for loading vision encoders with automatic configuration detection."""

    @staticmethod
    def load(config: SMTConfig) -> tuple[nn.Module, int]:
        """
        Load an encoder based on the specified provider.

        Args:
            config: SMT model configuration

        Returns:
            tuple: (encoder module, encoder output dimension)

        Raises:
            ValueError: If the encoder provider is not supported
        """
        provider = config.encoder_provider.lower()

        if provider == "transformers":
            encoder, output_dim = EncoderLoader._load_transformers(config)
        else:
            raise ValueError(
                f"Unsupported encoder provider: {config.encoder_provider}. "
                f"Supported providers: ['transformers']"
            )

        # Apply freezing if configured
        if config.freeze_encoder_stages > 0:
            EncoderLoader._freeze_encoder_stages(encoder, config.freeze_encoder_stages)

        logger.info(
            f"Loaded encoder '{config.encoder_model_name_or_path}' "
            f"via {provider} with output dimension {output_dim}"
        )

        return encoder, output_dim

    @staticmethod
    def _load_transformers(config: SMTConfig) -> tuple[nn.Module, int]:
        """
        Load encoder using HuggingFace Transformers AutoModel.

        Args:
            config: SMT model configuration

        Returns:
            tuple: (encoder module, encoder output dimension)
        """
        # Load encoder using AutoModel to support various architectures
        encoder = AutoModel.from_pretrained(config.encoder_model_name_or_path)

        # Detect output dimension from encoder config
        output_dim = EncoderLoader._detect_encoder_output_dim(encoder.config)

        logger.info(
            f"Loaded transformers encoder: {config.encoder_model_name_or_path}, "
            f"output_dim={output_dim}"
        )

        return encoder, output_dim

    @staticmethod
    def _detect_encoder_output_dim(encoder_config) -> int:
        """
        Heuristic to detect the encoder's output dimension.

        Different encoder architectures expose their output dimension via different
        config attributes. This method checks common attribute names in order of
        preference.

        Args:
            encoder_config: HuggingFace model config object

        Returns:
            int: Detected output dimension

        Raises:
            ValueError: If output dimension cannot be detected
        """
        # Try common attribute names for output dimension
        # Order matters: try most specific first

        # ConvNeXt, ResNet, some CNNs: hidden_sizes[-1]
        if hasattr(encoder_config, "hidden_sizes") and encoder_config.hidden_sizes:
            output_dim = encoder_config.hidden_sizes[-1]
            return output_dim

        # SAM: nested vision_config.hidden_size
        if hasattr(encoder_config, "vision_config") and hasattr(
            encoder_config.vision_config, "hidden_size"
        ):
            output_dim = encoder_config.vision_config.hidden_size
            return output_dim

        # BERT, ViT, many transformers: hidden_size
        if hasattr(encoder_config, "hidden_size"):
            output_dim = encoder_config.hidden_size
            return output_dim

        # Some CNN models: num_channels
        if hasattr(encoder_config, "num_channels"):
            output_dim = encoder_config.num_channels
            return output_dim

        # Swin Transformer: embed_dim
        if hasattr(encoder_config, "embed_dim"):
            output_dim = encoder_config.embed_dim
            return output_dim

        # If we can't detect it, raise an error
        raise ValueError(
            f"Could not detect encoder output dimension from config. "
            f"Available config attributes: {dir(encoder_config)}"
        )

    @staticmethod
    def _freeze_encoder_stages(encoder: nn.Module, num_stages: int) -> None:
        """
        Freeze the embedding layer and first N stages of the encoder.

        Currently implements ConvNeXtV2-specific freezing logic. Future versions
        may support architecture detection and per-architecture freezing strategies.

        Args:
            encoder: The encoder module to freeze
            num_stages: Number of encoder stages to freeze (0-indexed)
        """
        # ConvNeXt-specific freezing logic
        # Assumes encoder has: encoder.embeddings and encoder.encoder.stages[i]

        # Freeze embeddings
        if hasattr(encoder, "embeddings"):
            for param in encoder.embeddings.parameters():
                param.requires_grad = False
            logger.info("Froze encoder embeddings.")
        else:
            logger.warning(
                "Encoder does not have 'embeddings' attribute. Skipping embeddings freezing."
            )

        # Freeze early stages
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "stages"):
            stages = encoder.encoder.stages
            for i in range(num_stages):
                if i < len(stages):
                    for param in stages[i].parameters():
                        param.requires_grad = False
                    logger.info(f"Froze encoder stage {i}.")
                else:
                    logger.warning(
                        f"Requested freezing stage {i}, but encoder only has "
                        f"{len(stages)} stages. Skipping."
                    )
        else:
            logger.warning(
                "Encoder does not have 'encoder.stages' structure. "
                "Cannot freeze stages. This is expected for non-ConvNeXt architectures."
            )
