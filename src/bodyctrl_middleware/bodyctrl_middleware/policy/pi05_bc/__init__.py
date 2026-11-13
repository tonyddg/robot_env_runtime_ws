"""Pi0.5 trajectory plus local temporal behavior-cloning policy."""

from bodyctrl_middleware.policy.pi05_bc.config import Pi05BCPolicyConfig

__all__ = [
    "Pi05BCPolicyConfig",
    "Pi05BCTemporalPolicy",
]


def __getattr__(name: str):
    """Import the heavy policy module only when explicitly requested."""
    if name == "Pi05BCTemporalPolicy":
        from bodyctrl_middleware.policy.pi05_bc.policy import (
            Pi05BCTemporalPolicy,
        )

        return Pi05BCTemporalPolicy
    raise AttributeError(name)
