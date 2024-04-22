#!/usr/bin/env python3

import numpy as np
import logging

from collections import defaultdict
from typing import Tuple, Callable, Iterable, Optional, Dict, Any, List
from dataclasses import dataclass
import pybullet as pb

from imm.pybullet_util.common import (
    get_relative_transform,
    get_link_pose,
    get_name_from_index
)
from imm.pybullet_util.bullet_client import BulletClient


@dataclass(frozen=True)
class LinkPair:
    body_id_a: int
    link_id_a: int
    body_id_b: int
    link_id_b: int


def is_allowed(allowlist: Iterable[LinkPair], contact: List):
    """Checks whether a given contact is allowed."""
    for C in allowlist:
        if contact[1] != C.body_id_a:
            continue
        if contact[2] != C.body_id_b:
            continue

        if (C.link_id_a is not None):
            if C.link_id_a != contact[3]:
                continue

        if (C.link_id_b is not None):
            if C.link_id_b != contact[4]:
                continue
        # allowed
        return True
    # not allowed
    return False


class SetRobotState:
    def __init__(self, bc: BulletClient,
                 robot_id: int,
                 joint_ids: Iterable[int],
                 attachlist: Iterable[LinkPair]):
        self.bc = bc
        self.robot_id = robot_id
        self.joint_ids = joint_ids

        # Lookup relative transforms
        # of attachments.
        self.attach_xfms = {
            C: get_relative_transform(bc,
                                      C.body_id_a,
                                      C.link_id_a,
                                      C.link_id_b,
                                      C.body_id_b,
                                      inertial=False)
            for C in attachlist}

    def __call__(self, q: np.ndarray):
        bc = self.bc

        # TODO(ycho):
        # if pedantic, check for joint limits too.
        # TODO(ycho):
        # Consider setting base poses as well.

        # Set kinematic states.
        for i, v in zip(self.joint_ids, q):
            bc.resetJointState(self.robot_id, i, v)

        # Update transforms of attached bodies.
        for C, xfm in self.attach_xfms.items():
            # NOTE(ycho): as long as we're consistent
            # about `inertial` keyword, we're fine.
            pose = get_link_pose(bc, C.body_id_a, C.link_id_a,
                                 inertial=False)
            pose = bc.multiplyTransforms(pose[0], pose[1],
                                         xfm[0], xfm[1])
            bc.resetBasePositionAndOrientation(
                C.body_id_b, pose[0], pose[1])


class ContactBasedCollision:
    """General contact-based collision checker.

    see pb.getContactPoints()
    """

    def __init__(
            self, bc: BulletClient, robot_id: int, joint_ids: Iterable[int],
            allowlist: Iterable[LinkPair],
            attachlist: Iterable[LinkPair],
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            tol: Optional[Dict[int, float]] = None):
        self.bc = bc
        self.robot_id = robot_id
        self.joint_ids = joint_ids
        self.attachlist = attachlist
        self.joint_limits = joint_limits
        self.tol = tol
        self.set_state = SetRobotState(bc, robot_id, joint_ids, attachlist)

        # Split by `body_id_a` for convenience.
        self.allowlist = defaultdict(list)
        for C in allowlist:
            self.allowlist[C.body_id_a].append(C)

        # Lookup relative transforms
        # of attachments.
        self.attach_xfms = {
            C: get_relative_transform(bc,
                                      C.body_id_a,
                                      C.link_id_a,
                                      C.link_id_b,
                                      C.body_id_b)
            for C in attachlist}

    def __call__(self, q: np.ndarray, debug: bool = False) -> bool:
        # TODO(ycho): consider also returning `info`
        bc = self.bc
        robot_id: int = self.robot_id
        joint_limits = self.joint_limits

        # Check if exceeding joint limits.
        if (q < joint_limits[0]).any():
            logging.debug(F'Hit lower jlim: {q} < {joint_limits[0]}')
            return True
        if (q >= joint_limits[1]).any():
            logging.debug(F'Hit upper jlim: {q} >= {joint_limits[1]}')
            return True

        # Set robot state.
        self.set_state(q)

        # Perform collision detection.
        bc.performCollisionDetection()

        # Check collisions.
        # We primarily check the robot and the attached bodies.
        bodies = [robot_id] + [C.body_id_b for C in self.attachlist]

        # Configure tolerances.
        if self.tol is None:
            tol = {}
        else:
            tol = self.tol
        for body_id in bodies:
            tol.setdefault(body_id, 0.0)

        for body in bodies:
            contacts = bc.getContactPoints(bodyA=body)

            filtered_contacts = []
            allowlist = self.allowlist.get(body, [])
            for contact in contacts:
                if contact[8] >= tol[body]:
                    continue
                if is_allowed(allowlist, contact):
                    continue
                filtered_contacts.append(contact)
            contacts = filtered_contacts

            if len(contacts) > 0:
                msg = ''
                # In case of contact, optionally output debug messages.
                if debug:
                    for pt in contacts:
                        try:
                            names_a = get_name_from_index(
                                pt[1], bc.sim_id, [pt[3]], link=True)
                            names_b = get_name_from_index(
                                pt[2], bc.sim_id, [pt[4]], link=True)
                            msg += F'names_a = {names_a}, names_b = {names_b}\n'
                        except pb.error:
                            msg += F'{pt[1], pt[2], pt[3], pt[4]}\n'
                            continue
                    logging.debug(msg)
                return True
        return False
