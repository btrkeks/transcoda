"""Failure-policy settings for the rewrite executor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailurePolicySettings:
    name: str
    task_timeout_seconds: int
    max_task_retries_timeout: int
    max_task_retries_expired: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "task_timeout_seconds": self.task_timeout_seconds,
            "max_task_retries_timeout": self.max_task_retries_timeout,
            "max_task_retries_expired": self.max_task_retries_expired,
        }


def resolve_failure_policy(failure_policy: str) -> FailurePolicySettings:
    normalized = str(failure_policy).strip().lower()
    policy_map = {
        "throughput": FailurePolicySettings(
            name="throughput",
            task_timeout_seconds=5,
            max_task_retries_timeout=0,
            max_task_retries_expired=0,
        ),
        "balanced": FailurePolicySettings(
            name="balanced",
            task_timeout_seconds=30,
            max_task_retries_timeout=1,
            max_task_retries_expired=0,
        ),
        "coverage": FailurePolicySettings(
            name="coverage",
            task_timeout_seconds=60,
            max_task_retries_timeout=1,
            max_task_retries_expired=1,
        ),
    }
    if normalized not in policy_map:
        raise ValueError("failure_policy must be one of: throughput, balanced, coverage")
    return policy_map[normalized]
