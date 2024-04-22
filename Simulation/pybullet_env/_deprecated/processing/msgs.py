from dataclasses import dataclass
from typing import List

from imm.pybullet_util.typing_extra import QuaternionT, TranslationT


@dataclass(frozen=True)
class ActionInfo:
    action: str
    pos: TranslationT
    orn: QuaternionT

@dataclass(frozen=True)
class FetchingRequestMsg:
    initial_state_info: List
    action_info: ActionInfo
    

@dataclass(frozen=True)
class ObservationInfo:
    observation: object
    likelihood: float

@dataclass(frozen=True)
class FetchingResponseMsg:
    result_state_info: List
    observation_info: ObservationInfo