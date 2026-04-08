from __future__ import annotations

import torch

_NEAR_ZERO_EPS = {
    torch.float32: 0.01,
    torch.float64: 0.005,
}
_NEAR_PI_EPS = {
    torch.float32: 0.01,
    torch.float64: 1.0e-7,
}
_MATRIX_EPS = {
    torch.float32: 4.0e-4,
    torch.float64: 1.0e-6,
}
_NON_ZERO = 1.0


def _near_zero_eps(dtype: torch.dtype) -> float:
    return _NEAR_ZERO_EPS.get(dtype, 0.01)


def _near_pi_eps(dtype: torch.dtype) -> float:
    return _NEAR_PI_EPS.get(dtype, 0.01)


def _matrix_eps(dtype: torch.dtype) -> float:
    return _MATRIX_EPS.get(dtype, 4.0e-4)


def _require_tangent(tangent: torch.Tensor) -> None:
    if tangent.ndim < 1 or tangent.shape[-1] != 3:
        raise ValueError("tangent shape must be (..., 3)")


def _require_rotation(rotation: torch.Tensor) -> None:
    if rotation.ndim < 2 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation shape must be (..., 3, 3)")


def _exp_impl_helper(tangent: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(tangent, dim=-1, keepdim=True).unsqueeze(-1)
    theta2 = theta * theta

    near_zero = theta < _near_zero_eps(tangent.dtype)
    theta_nz = torch.where(near_zero, torch.ones_like(theta), theta)
    theta2_nz = torch.where(near_zero, torch.ones_like(theta2), theta2)

    cosine = torch.where(near_zero, 8.0 / (4.0 + theta2) - 1.0, theta.cos())
    sine = theta.sin()
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(
        near_zero,
        0.5 * sine_by_theta,
        (1.0 - cosine) / theta2_nz,
    )

    size = tangent.shape[:-1]
    result = (
        one_minus_cosine_by_theta2
        * tangent.view(*size, 3, 1)
        @ tangent.view(*size, 1, 3)
    )
    result[..., 0, 0] += cosine.view(size)
    result[..., 1, 1] += cosine.view(size)
    result[..., 2, 2] += cosine.view(size)

    sine_axis = sine_by_theta.view(*size, 1) * tangent
    result[..., 0, 1] -= sine_axis[..., 2]
    result[..., 1, 0] += sine_axis[..., 2]
    result[..., 0, 2] += sine_axis[..., 1]
    result[..., 2, 0] -= sine_axis[..., 1]
    result[..., 1, 2] -= sine_axis[..., 0]
    result[..., 2, 1] += sine_axis[..., 0]
    return result


def _log_impl_helper(rotation: torch.Tensor) -> torch.Tensor:
    size = rotation.shape[:-2]

    sine_axis = rotation.new_zeros(*size, 3)
    sine_axis[..., 0] = 0.5 * (rotation[..., 2, 1] - rotation[..., 1, 2])
    sine_axis[..., 1] = 0.5 * (rotation[..., 0, 2] - rotation[..., 2, 0])
    sine_axis[..., 2] = 0.5 * (rotation[..., 1, 0] - rotation[..., 0, 1])

    cosine = 0.5 * (rotation.diagonal(dim1=-1, dim2=-2).sum(dim=-1) - 1.0)
    cosine = cosine.clamp(-1.0, 1.0)
    sine = sine_axis.norm(dim=-1)
    theta = torch.atan2(sine, cosine)

    near_zero = theta < _near_zero_eps(rotation.dtype)
    near_pi = 1.0 + cosine <= _near_pi_eps(rotation.dtype)
    near_zero_or_near_pi = torch.logical_or(near_zero, near_pi)

    sine_nz = torch.where(near_zero_or_near_pi, torch.ones_like(sine), sine)
    scale = torch.where(
        near_zero_or_near_pi,
        1.0 + sine * sine / 6.0,
        theta / sine_nz,
    )
    result = sine_axis * scale.view(*size, 1)

    ddiag = torch.diagonal(rotation, dim1=-1, dim2=-2)
    major = torch.logical_and(
        ddiag[..., 1] > ddiag[..., 0], ddiag[..., 1] > ddiag[..., 2]
    ) + 2 * torch.logical_and(
        ddiag[..., 2] > ddiag[..., 0], ddiag[..., 2] > ddiag[..., 1]
    )

    flat_rot = rotation.reshape(-1, 3, 3)
    flat_major = major.reshape(-1)
    rows = torch.arange(flat_rot.shape[0], device=rotation.device)
    sel_rows = 0.5 * (flat_rot[rows, flat_major] + flat_rot[rows, :, flat_major])
    sel_rows[rows, flat_major] -= cosine.reshape(-1)
    sel_rows = sel_rows.reshape(*size, 3)

    denom = torch.where(
        near_zero,
        torch.full_like(cosine, _NON_ZERO),
        sel_rows.norm(dim=-1),
    )
    axis = sel_rows / denom.unsqueeze(-1)

    sign_tmp = sine_axis.reshape(-1, 3)[rows, flat_major].reshape(*size).sign()
    sign = torch.where(sign_tmp != 0, sign_tmp, torch.ones_like(sign_tmp))
    tangent_pi = axis * (theta * sign).unsqueeze(-1)
    return torch.where(near_pi.unsqueeze(-1), tangent_pi, result)


def so3_exp(tangent: torch.Tensor) -> torch.Tensor:
    _require_tangent(tangent)
    return _exp_impl_helper(tangent)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    _require_rotation(rotation)
    return _log_impl_helper(rotation)


def so3_compose(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    _require_rotation(left)
    _require_rotation(right)
    return left @ right


def so3_inverse(rotation: torch.Tensor) -> torch.Tensor:
    _require_rotation(rotation)
    return rotation.transpose(-1, -2).clone()


def so3_geodesic_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _require_rotation(pred)
    _require_rotation(target)
    rel = so3_compose(so3_inverse(pred), target)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos_theta)


def so3_validate(rotation: torch.Tensor) -> dict[str, object]:
    _require_rotation(rotation)
    check_tensor = rotation if rotation.dtype == torch.float64 else rotation.double()
    eye = torch.eye(3, dtype=check_tensor.dtype, device=check_tensor.device)
    ortho_err = (check_tensor @ check_tensor.transpose(-1, -2) - eye).abs().amax(dim=(-2, -1))
    det_err = (torch.linalg.det(check_tensor) - 1.0).abs()
    eps = _matrix_eps(rotation.dtype)
    return {
        "is_valid": bool(((ortho_err < eps) & (det_err < eps)).all().item()),
        "max_orthogonality_error": float(ortho_err.max().item()),
        "max_det_error": float(det_err.max().item()),
    }


def so3_normalize(rotation: torch.Tensor) -> torch.Tensor:
    _require_rotation(rotation)
    u, _, v = torch.svd(rotation)
    sign = torch.det(u @ v).view(*rotation.shape[:-2], 1, 1)
    vt = torch.cat(
        (v[..., :2], torch.where(sign > 0, v[..., 2:], -v[..., 2:])),
        dim=-1,
    ).transpose(-1, -2)
    return u @ vt


def so3_batch_info(rotation: torch.Tensor) -> dict[str, object]:
    _require_rotation(rotation)
    return {
        "batch_shape": tuple(rotation.shape[:-2]),
        "dtype": str(rotation.dtype),
        "device": str(rotation.device),
    }


class SO3:
    def __init__(self, tensor: torch.Tensor, strict_checks: bool = False, disable_checks: bool = False):
        if disable_checks:
            self.tensor = tensor
            return
        info = so3_validate(tensor)
        if info["is_valid"]:
            self.tensor = tensor
            return
        if strict_checks:
            raise ValueError("Input tensor is not a valid SO3 rotation")
        self.tensor = so3_normalize(tensor)

    @staticmethod
    def exp(tangent: torch.Tensor) -> "SO3":
        return SO3(so3_exp(tangent), disable_checks=True)

    def log(self) -> torch.Tensor:
        return so3_log(self.tensor)

    def compose(self, other: "SO3") -> "SO3":
        return SO3(so3_compose(self.tensor, other.tensor), disable_checks=True)

    def inverse(self) -> "SO3":
        return SO3(so3_inverse(self.tensor), disable_checks=True)

    def geodesic_error(self, other: "SO3") -> torch.Tensor:
        return so3_geodesic_error(self.tensor, other.tensor)

    def validate(self) -> dict[str, object]:
        return so3_validate(self.tensor)

    def info(self) -> dict[str, object]:
        return so3_batch_info(self.tensor)


def transform_by_pose(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return torch.einsum("...ij,...j->...i", rotation, points) + translation


def rotate(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", rotation, vector)


def exp_map(vector: torch.Tensor) -> torch.Tensor:
    return so3_exp(vector)


def log_map(rotation: torch.Tensor) -> torch.Tensor:
    return so3_log(rotation)


def inv(rotation: torch.Tensor) -> torch.Tensor:
    return so3_inverse(rotation)


def compose(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    return so3_compose(mat1, mat2)


def transform(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    return torch.einsum("...ij,...j->...i", rotation, vector) + translation


def check_orthogonality(rotation: torch.Tensor, eps: float = 1.0e-4) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = rotation.shape[0]
    identity = torch.eye(3, device=rotation.device).expand(batch_size, 3, 3)
    rr_t = torch.bmm(rotation, rotation.transpose(-1, -2))
    error = torch.norm(rr_t - identity, p="fro", dim=(1, 2))
    return error < eps, error


def mat_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Input matrix must be 3x3")
    return matrix[..., :3, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def rot6d_to_mat(rot6d: torch.Tensor) -> torch.Tensor:
    if rot6d.shape[-1] != 6:
        raise ValueError("Input 6D vector must have a last dimension of 6")
    x_raw, y_raw = rot6d[..., :3], rot6d[..., 3:]
    x = torch.nn.functional.normalize(x_raw, p=2, dim=-1)
    y_ortho = y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x
    norm_y_ortho = torch.linalg.vector_norm(y_ortho, dim=-1, keepdim=True)

    is_singular = norm_y_ortho < 1.0e-8
    z_axis = torch.zeros_like(x)
    z_axis[..., 2] = 1.0
    x_axis = torch.zeros_like(x)
    x_axis[..., 0] = 1.0
    is_x_aligned_with_z = torch.abs(x[..., 2:3]) > 0.99
    new_y = torch.where(is_x_aligned_with_z, x_axis, z_axis)
    y_safe = torch.where(is_singular, new_y, y_raw)
    y_ortho_safe = y_safe - (x * y_safe).sum(dim=-1, keepdim=True) * x
    y = torch.nn.functional.normalize(y_ortho_safe, p=2, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def pose9d_to_mat(pose9d: torch.Tensor) -> torch.Tensor:
    if pose9d.shape[-1] != 9:
        raise ValueError("Input pose9d must have a last dimension of 9")
    batch_shape = pose9d.shape[:-1]
    transform_mat = torch.eye(4, device=pose9d.device, dtype=pose9d.dtype).expand(*batch_shape, 4, 4).clone()
    transform_mat[..., :3, :3] = rot6d_to_mat(pose9d[..., 3:9])
    transform_mat[..., :3, 3] = pose9d[..., :3]
    return transform_mat


def mat_to_pose9d(matrix: torch.Tensor) -> torch.Tensor:
    position = matrix[..., :3, 3]
    rot6d = mat_to_rot6d(matrix[..., :3, :3])
    return torch.cat([position, rot6d], dim=-1)


def rR_to_T(position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    batch_shape = position.shape[:-1]
    transform_mat = torch.eye(4, device=position.device, dtype=position.dtype).expand(*batch_shape, 4, 4).clone()
    transform_mat[..., :3, :3] = rotation
    transform_mat[..., :3, 3] = position
    return transform_mat


def T_to_rR(transform_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return transform_mat[..., :3, 3], transform_mat[..., :3, :3]


def se3_mat_inv(transform_mat: torch.Tensor) -> torch.Tensor:
    if transform_mat.shape[-2:] != (4, 4):
        raise ValueError("Input transform must be (..., 4, 4)")
    rotation = transform_mat[..., :3, :3]
    translation = transform_mat[..., :3, 3]
    rotation_inv = rotation.transpose(-2, -1)
    translation_inv = -torch.matmul(rotation_inv, translation.unsqueeze(-1)).squeeze(-1)
    result = torch.eye(4, device=transform_mat.device, dtype=transform_mat.dtype).expand(*transform_mat.shape[:-2], 4, 4).clone()
    result[..., :3, :3] = rotation_inv
    result[..., :3, 3] = translation_inv
    return result


def rotation_matrix_to_angles(rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_shape = rotation.shape[:-2]
    flat_rotation = rotation.reshape(-1, 3, 3)
    batch_size = flat_rotation.shape[0]
    device = flat_rotation.device

    sy = -flat_rotation[:, 2, 0]
    elevation = torch.asin(sy.clamp(-1.0, 1.0))

    gimbal_lock_mask = torch.abs(sy) > 0.99999
    not_gimbal_lock_mask = ~gimbal_lock_mask

    azimuth = torch.zeros(batch_size, device=device, dtype=rotation.dtype)
    roll = torch.zeros(batch_size, device=device, dtype=rotation.dtype)

    if not_gimbal_lock_mask.any():
        azimuth[not_gimbal_lock_mask] = torch.atan2(
            flat_rotation[not_gimbal_lock_mask, 1, 0],
            flat_rotation[not_gimbal_lock_mask, 0, 0],
        )
        roll[not_gimbal_lock_mask] = torch.atan2(
            flat_rotation[not_gimbal_lock_mask, 2, 1],
            flat_rotation[not_gimbal_lock_mask, 2, 2],
        )

    if gimbal_lock_mask.any():
        azimuth[gimbal_lock_mask] = torch.atan2(
            -flat_rotation[gimbal_lock_mask, 0, 1],
            flat_rotation[gimbal_lock_mask, 1, 1],
        )

    return (
        azimuth.reshape(*batch_shape),
        elevation.reshape(*batch_shape),
        roll.reshape(*batch_shape),
    )

__all__ = [
    "SO3",
    "check_orthogonality",
    "compose",
    "exp_map",
    "inv",
    "log_map",
    "mat_to_pose9d",
    "mat_to_rot6d",
    "pose9d_to_mat",
    "rR_to_T",
    "rotate",
    "rotation_matrix_to_angles",
    "rot6d_to_mat",
    "se3_mat_inv",
    "so3_batch_info",
    "so3_compose",
    "so3_exp",
    "so3_geodesic_error",
    "so3_inverse",
    "so3_log",
    "so3_normalize",
    "so3_validate",
    "T_to_rR",
    "transform",
    "transform_by_pose",
]
