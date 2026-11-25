"""Pi0.5 direct-action runtime policy."""

from bodyctrl_middleware.policy.pi05_direct.config import (
    Pi05DirectPolicyConfig,
)
from bodyctrl_middleware.policy.pi05_direct.client import (
    OpenPiDirectActionClient,
)

__all__ = [
    "OpenPiDirectActionClient",
    "Pi05DirectActionPolicy",
    "Pi05DirectPolicyConfig",
]


def __getattr__(name: str):
    """Import the policy implementation only when requested."""
    if name == "Pi05DirectActionPolicy":
        from bodyctrl_middleware.policy.pi05_direct.policy import (
            Pi05DirectActionPolicy,
        )

        return Pi05DirectActionPolicy
    raise AttributeError(name)
