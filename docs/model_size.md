## With ConvNextV2-tiny

Encoder (ConvNextV2-tiny): 27,866,496
Bridge (MLP): 2,624,000
Decoder (vanilla Transformer): 28,311,480
Total: 58,801,976

learning_rate: 0.001
encoder_lr_factor: 0.3
lr_scheduler: cosine
warmup_steps: 500

effective_batch_size: 72

## With ConvNextV2-base

Encoder (ConvNextV2-base): 87,692,800
Bridge (MLP): 3,148,288
Decoder (vanilla Transformer): 28,311,480
Total: 119,152,568
