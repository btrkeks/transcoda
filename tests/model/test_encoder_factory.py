"""Unit tests for the EncoderLoader factory."""

import pytest
import torch

from src.model.configuration_smt import SMTConfig
from src.model.encoder import EncoderLoader


class TestEncoderLoader:
    """Test suite for EncoderLoader factory."""

    def test_load_convnextv2_tiny(self):
        """Test loading ConvNeXtV2-tiny via factory with correct output dimension."""
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
            encoder_provider="transformers",
            freeze_encoder_stages=0,
        )

        encoder, output_dim = EncoderLoader.load(config)

        # ConvNeXtV2-tiny has output dimension of 768
        assert output_dim == 768, f"Expected output_dim=768, got {output_dim}"
        assert encoder is not None
        assert hasattr(encoder, "config")

        # Verify encoder can do a forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = encoder(pixel_values=dummy_input)
            assert output.last_hidden_state is not None
            # Output should be 4D: (B, C, H, W)
            assert len(output.last_hidden_state.shape) == 4
            assert output.last_hidden_state.shape[1] == output_dim

    def test_load_convnextv2_base(self):
        """Test loading ConvNeXtV2-base with different output dimension."""
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-base-1k-224",
            encoder_provider="transformers",
            freeze_encoder_stages=0,
        )

        encoder, output_dim = EncoderLoader.load(config)

        # ConvNeXtV2-base has output dimension of 1024
        assert output_dim == 1024, f"Expected output_dim=1024, got {output_dim}"

    def test_load_resnet50(self):
        """Test loading ResNet50 via transformers AutoModel."""
        config = SMTConfig(
            encoder_model_name_or_path="microsoft/resnet-50",
            encoder_provider="transformers",
            freeze_encoder_stages=0,
        )

        encoder, output_dim = EncoderLoader.load(config)

        # ResNet-50 has output dimension of 2048
        assert output_dim == 2048, f"Expected output_dim=2048, got {output_dim}"
        assert encoder is not None

        # Verify forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = encoder(pixel_values=dummy_input)
            assert output.last_hidden_state is not None

    def test_load_swin_tiny(self):
        """Test loading Swin Transformer via transformers AutoModel."""
        config = SMTConfig(
            encoder_model_name_or_path="microsoft/swin-tiny-patch4-window7-224",
            encoder_provider="transformers",
            freeze_encoder_stages=0,
        )

        encoder, output_dim = EncoderLoader.load(config)

        # Swin-tiny has output dimension of 768
        assert output_dim == 768, f"Expected output_dim=768, got {output_dim}"
        assert encoder is not None

        # Verify forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = encoder(pixel_values=dummy_input)
            assert output.last_hidden_state is not None

    def test_invalid_provider(self):
        """Test that invalid encoder provider raises ValueError."""
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
            encoder_provider="invalid_provider",
            freeze_encoder_stages=0,
        )

        with pytest.raises(ValueError, match="Unsupported encoder provider"):
            EncoderLoader.load(config)

    def test_freeze_encoder_stages_zero(self):
        """Test that freeze_encoder_stages=0 leaves all params trainable."""
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
            encoder_provider="transformers",
            freeze_encoder_stages=0,
        )

        encoder, _ = EncoderLoader.load(config)

        # Check that all parameters are trainable
        for name, param in encoder.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be trainable"

    def test_freeze_encoder_stages_embeddings(self):
        """Test that freeze_encoder_stages=1 freezes embeddings and stage 0."""
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
            encoder_provider="transformers",
            freeze_encoder_stages=1,
        )

        encoder, _ = EncoderLoader.load(config)

        # Check that embeddings are frozen
        for param in encoder.embeddings.parameters():
            assert not param.requires_grad, "Embeddings should be frozen"

        # Check that stage 0 is frozen
        for param in encoder.encoder.stages[0].parameters():
            assert not param.requires_grad, "Stage 0 should be frozen"

        # Check that later stages are still trainable
        for param in encoder.encoder.stages[1].parameters():
            assert param.requires_grad, "Stage 1 should be trainable"

    def test_freeze_encoder_stages_multiple(self):
        """Test that freeze_encoder_stages=2 freezes embeddings and first 2 stages."""
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
            encoder_provider="transformers",
            freeze_encoder_stages=2,
        )

        encoder, _ = EncoderLoader.load(config)

        # Check that embeddings are frozen
        for param in encoder.embeddings.parameters():
            assert not param.requires_grad, "Embeddings should be frozen"

        # Check that stages 0 and 1 are frozen
        for i in range(2):
            for param in encoder.encoder.stages[i].parameters():
                assert not param.requires_grad, f"Stage {i} should be frozen"

        # Check that stage 2 is still trainable
        for param in encoder.encoder.stages[2].parameters():
            assert param.requires_grad, "Stage 2 should be trainable"

    def test_detect_encoder_output_dim_hidden_sizes(self):
        """Test output dimension detection via hidden_sizes attribute."""

        # Create a mock config with hidden_sizes
        class MockConfig:
            hidden_sizes = [96, 192, 384, 768]

        output_dim = EncoderLoader._detect_encoder_output_dim(MockConfig())
        assert output_dim == 768

    def test_detect_encoder_output_dim_hidden_size(self):
        """Test output dimension detection via hidden_size attribute."""

        # Create a mock config with hidden_size
        class MockConfig:
            hidden_size = 512

        output_dim = EncoderLoader._detect_encoder_output_dim(MockConfig())
        assert output_dim == 512

    def test_detect_encoder_output_dim_num_channels(self):
        """Test output dimension detection via num_channels attribute."""

        # Create a mock config with num_channels
        class MockConfig:
            num_channels = 256

        output_dim = EncoderLoader._detect_encoder_output_dim(MockConfig())
        assert output_dim == 256

    def test_detect_encoder_output_dim_embed_dim(self):
        """Test output dimension detection via embed_dim attribute."""

        # Create a mock config with embed_dim
        class MockConfig:
            embed_dim = 768

        output_dim = EncoderLoader._detect_encoder_output_dim(MockConfig())
        assert output_dim == 768

    def test_detect_encoder_output_dim_vision_config(self):
        """Test output dimension detection via nested vision_config.hidden_size (SAM pattern)."""

        # Create a mock config with nested vision_config (SAM-style)
        class MockVisionConfig:
            hidden_size = 768

        class MockConfig:
            vision_config = MockVisionConfig()

        output_dim = EncoderLoader._detect_encoder_output_dim(MockConfig())
        assert output_dim == 768

    def test_detect_encoder_output_dim_precedence(self):
        """Test that hidden_sizes takes precedence over other attributes."""

        # Create a mock config with multiple attributes
        class MockConfig:
            hidden_sizes = [96, 192, 384, 768]
            hidden_size = 512
            embed_dim = 256

        output_dim = EncoderLoader._detect_encoder_output_dim(MockConfig())
        # Should use hidden_sizes[-1] = 768, not hidden_size or embed_dim
        assert output_dim == 768

    def test_detect_encoder_output_dim_missing(self):
        """Test that missing output dimension raises ValueError."""

        # Create a mock config with no relevant attributes
        class MockConfig:
            some_other_attribute = 123

        with pytest.raises(ValueError, match="Could not detect encoder output dimension"):
            EncoderLoader._detect_encoder_output_dim(MockConfig())

    def test_freeze_non_convnext_encoder(self):
        """Test that freezing on non-ConvNeXt encoder logs warnings but doesn't crash."""
        # ResNet doesn't have the same structure as ConvNeXt
        config = SMTConfig(
            encoder_model_name_or_path="microsoft/resnet-50",
            encoder_provider="transformers",
            freeze_encoder_stages=1,
        )

        # Should not raise an error, just log warnings
        encoder, output_dim = EncoderLoader.load(config)

        assert encoder is not None
        assert output_dim == 2048

    def test_encoder_compatibility_with_automodel(self):
        """Test that AutoModel-loaded encoder maintains same structure as direct import."""
        # This test verifies backward compatibility with LLRD
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
            encoder_provider="transformers",
            freeze_encoder_stages=0,
        )

        encoder, _ = EncoderLoader.load(config)

        # Verify that AutoModel wraps the same underlying ConvNeXtV2 structure
        assert hasattr(encoder, "embeddings"), "Encoder should have embeddings"
        assert hasattr(encoder, "encoder"), "Encoder should have encoder attribute"
        assert hasattr(encoder.encoder, "stages"), "Encoder should have stages"

        # Verify parameter names match expected pattern for LLRD
        param_names = [name for name, _ in encoder.named_parameters()]
        assert any("embeddings" in name for name in param_names)
        assert any("stages" in name for name in param_names)

    def test_load_transformers_method_directly(self):
        """Test calling _load_transformers method directly."""
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
            encoder_provider="transformers",
        )

        encoder, output_dim = EncoderLoader._load_transformers(config)

        assert encoder is not None
        assert output_dim == 768

    def test_freeze_encoder_stages_method_directly(self):
        """Test calling _freeze_encoder_stages method directly."""
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
            encoder_provider="transformers",
            freeze_encoder_stages=0,  # Don't freeze in load()
        )

        encoder, _ = EncoderLoader.load(config)

        # All params should be trainable initially
        assert all(p.requires_grad for p in encoder.parameters())

        # Now freeze 1 stage
        EncoderLoader._freeze_encoder_stages(encoder, 1)

        # Check that embeddings and stage 0 are frozen
        for param in encoder.embeddings.parameters():
            assert not param.requires_grad

        for param in encoder.encoder.stages[0].parameters():
            assert not param.requires_grad

    def test_encoder_output_shape(self):
        """Test that encoder output has the expected 4D shape (B, C, H, W)."""
        config = SMTConfig(
            encoder_model_name_or_path="facebook/convnextv2-tiny-1k-224",
            encoder_provider="transformers",
            freeze_encoder_stages=0,
        )

        encoder, output_dim = EncoderLoader.load(config)

        # Create a dummy input and verify output shape
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 224, 224)  # Batch size 2
            output = encoder(pixel_values=dummy_input)

            assert output.last_hidden_state is not None
            # Should be 4D tensor
            assert len(output.last_hidden_state.shape) == 4
            # Batch dimension
            assert output.last_hidden_state.shape[0] == 2
            # Channel dimension should match detected output_dim
            assert output.last_hidden_state.shape[1] == output_dim
