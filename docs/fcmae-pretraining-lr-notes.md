# FCMAE Pretraining LR Notes

FCMAE pretraining scales the configured learning rate by effective batch size:

```text
actual_lr = base_learning_rate * batch_size * accumulate_grad_batches * world_size / 256
```

## Runs Tried

| Run | Config | Actual LR | Observation |
| --- | --- | ---: | --- |
| `3fp7uqvz` | `base_learning_rate=0.00015`, effective batch `8` | `4.6875e-6` | Stable but slow. Loss dropped early, then mostly hovered around `0.65-0.8`; diagnostic tail mean was about `0.764`. |
| `m4xss0vt` | `base_learning_rate=0.00064`, effective batch `8` | `2.0e-5` | Better. Loss continued improving through about `11k` steps, with later samples often around `0.26-0.47` and latest observed summary loss about `0.243`. No obvious instability. |

## Current Recommendation

Keep the next run near **actual LR `2e-5`** until reconstruction previews can be inspected.

For effective batch `8`, use:

```text
base_learning_rate = 0.00064
```

For effective batch `16`, use:

```text
base_learning_rate = 0.00032
```

The local base config may use a different effective batch size than the W&B runs above; check the formula before restarting.

## Next Sweep Point

If the new reconstruction previews look healthy and loss keeps descending smoothly, try actual LR `3e-5` next.

For effective batch `8`, that is:

```text
base_learning_rate = 0.00096
```

For effective batch `16`, that is:

```text
base_learning_rate = 0.00048
```
