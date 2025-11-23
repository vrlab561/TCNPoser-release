# MIT License
# Copyright (c) 2022 ETH Sensing, Interaction & Perception Lab
#
# This code is based on https://github.com/eth-siplab/AvatarPoser
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import math
import numpy as np
from human_body_prior.tools import tgm_conversion as tgm
from human_body_prior.tools.rotation_tools import aa2matrot, matrot2aa
from torch.nn import functional as F


# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0


def bgs(d6s):
    d6s = d6s.reshape(-1, 2, 3).permute(0, 2, 1)
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    c = torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1
    b2 = F.normalize(a2 - c, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrot2sixd(pose_matrot):
    """
    :param pose_matrot: Nx3x3
    :return: pose_6d: Nx6
    """
    pose_6d = torch.cat([pose_matrot[:, :3, 0], pose_matrot[:, :3, 1]], dim=1)
    return pose_6d


def aa2sixd(pose_aa):
    """
    :param pose_aa Nx3
    :return: pose_6d: Nx6
    """
    pose_matrot = aa2matrot(pose_aa)
    pose_6d = matrot2sixd(pose_matrot)
    return pose_6d


def sixd2matrot(pose_6d):
    """
    :param pose_6d: Nx6
    :return: pose_matrot: Nx3x3
    """
    rot_vec_1 = pose_6d[:, :3]
    rot_vec_2 = pose_6d[:, 3:6]
    rot_vec_3 = torch.linalg.cross(rot_vec_1, rot_vec_2)
    pose_matrot = torch.stack([rot_vec_1, rot_vec_2, rot_vec_3], dim=-1)
    return pose_matrot


def sixd2aa(pose_6d, batch=False):
    """
    :param pose_6d: Nx6
    :return: pose_aa: Nx3
    """
    if batch:
        B, J, C = pose_6d.shape
        pose_6d = pose_6d.reshape(-1, 6)
    pose_matrot = sixd2matrot(pose_6d)
    pose_aa = matrot2aa(pose_matrot)
    pose_aa_np=pose_aa.cpu().detach().numpy()
    if batch:
        pose_aa = pose_aa.reshape(B, J, 3)
    return pose_aa


def sixd2quat(pose_6d):
    """
    :param pose_6d: Nx6
    :return: pose_quaternion: Nx4
    """
    pose_mat = sixd2matrot(pose_6d)
    pose_mat_34 = torch.cat(
        (pose_mat, torch.zeros(pose_mat.size(0), pose_mat.size(1), 1)), dim=-1
    )
    pose_quaternion = tgm.rotation_matrix_to_quaternion(pose_mat_34)
    return pose_quaternion


def quat2aa(pose_quat):
    """
    :param pose_quat: Nx4
    :return: pose_aa: Nx3
    """
    return tgm.quaternion_to_angle_axis(pose_quat)


def mat2euler(mat, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    Note that many Euler angle triplets can describe one matrix.

    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    #
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS4:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS4:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az
