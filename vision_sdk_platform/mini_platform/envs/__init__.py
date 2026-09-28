from .env_base import AirHockeyBase
from .env_defend import AirHockeyDefend
from .env_double import AirHockeyDouble
from .env_hitting import AirHockeyHit
from .env_prepare import AirHockeyPrepare
from .env_single import AirHockeySingle
from .position_control import (MetalPositionDefend, MetalPositionDouble, MetalPositionHit,
                               MetalPositionPrepare, PositionControlMetal,
                               MetalSdkDefend, MetalSdkDouble, MetalSdkHit, MetalSdkPrepare,
                               SdkPositionControlMetal,
                               MetalPositionHit2P, MetalSdkHit2P)

#: Task name -> environment class, in the shape `AirHockeyChallengeWrapper.env_dict` expects.
#: The wrapper imports this lazily when it sees a "mini-" task, so the framework carries no
#: import-time dependency on this package.
#:
#: The `-sdk` variants are identical in every respect except the low-level joint controller:
#: they run the METAL arm SDK's own law (core/sdk_control.py) instead of the challenge stack's
#: hand-tuned PD. Same planner, same FSM, same action contract -- so switching between them is
#: a controlled A/B on one variable.
ENV_DICT = {
    "mini-hit": MetalPositionHit,
    "mini-defend": MetalPositionDefend,
    "mini-prepare": MetalPositionPrepare,
    "mini-hit-sdk": MetalSdkHit,
    "mini-defend-sdk": MetalSdkDefend,
    "mini-prepare-sdk": MetalSdkPrepare,
    # Both arms contest the puck: the baseline agent plays the opponent side too, instead
    # of the scripted lissajous wander that never touches it.
    "mini-hit-2p": MetalPositionHit2P,
    "mini-hit-sdk-2p": MetalSdkHit2P,
}

__all__ = [
    "AirHockeyBase", "AirHockeyDefend", "AirHockeyDouble", "AirHockeyHit",
    "AirHockeyPrepare", "AirHockeySingle", "PositionControlMetal",
    "MetalPositionHit", "MetalPositionDefend", "MetalPositionPrepare", "MetalPositionDouble",
    "SdkPositionControlMetal",
    "MetalSdkHit", "MetalSdkDefend", "MetalSdkPrepare", "MetalSdkDouble",
    "MetalPositionHit2P", "MetalSdkHit2P",
    "ENV_DICT",
]
