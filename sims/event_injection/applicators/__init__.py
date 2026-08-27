"""
Built-in event applicators.

BUILTIN_APPLICATORS is a dict mapping event_id → applicator instance,
ready to pass to EventScheduler.

Only physics-level injections are included, sensor-only corruptions
(Transient Spike/Dip) were removed because they don't affect the
actual simulation.
"""

from event_injection.applicators.payload import PayloadAdditionApplicator
from event_injection.applicators.friction import FrictionDecreaseApplicator
from event_injection.applicators.motor import MotorMiscommutationApplicator
from event_injection.applicators.gripper_failure import GripperActivationFailureApplicator
from event_injection.applicators.gripper_release import GripperReleaseApplicator
from event_injection.applicators.collision import CollisionApplicator
BUILTIN_APPLICATORS = {
    6: PayloadAdditionApplicator(),
    11: FrictionDecreaseApplicator(),
    12: MotorMiscommutationApplicator(),
    13: GripperActivationFailureApplicator(),
    14: GripperReleaseApplicator(),
    16: CollisionApplicator(),
}

__all__ = [
    "BUILTIN_APPLICATORS",
    "PayloadAdditionApplicator",
    "FrictionDecreaseApplicator",
    "MotorMiscommutationApplicator",
    "GripperActivationFailureApplicator",
    "GripperReleaseApplicator",
    "CollisionApplicator",
]
