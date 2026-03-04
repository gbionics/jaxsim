import jax
import jax.numpy as jnp
import pytest
from jax.errors import JaxRuntimeError
from numpy.testing import assert_array_equal

import jaxsim.api as js
from jaxsim import VelRepr
from jaxsim.math.quaternion import Quaternion

from . import utils
from .utils import assert_allclose


def test_frame_index(jaxsim_models_types: js.model.JaxSimModel):

    model = jaxsim_models_types

    # =====
    # Tests
    # =====

    n_l = model.number_of_links()
    n_f = len(model.frame_names())

    for idx, frame_name in enumerate(model.frame_names()):
        frame_index = n_l + idx
        assert js.frame.name_to_idx(model=model, frame_name=frame_name) == frame_index
        assert js.frame.idx_to_name(model=model, frame_index=frame_index) == frame_name
        assert (
            js.frame.idx_of_parent_link(model=model, frame_index=frame_index)
            < model.number_of_links()
        )

    # See discussion in https://github.com/ami-iit/jaxsim/pull/280
    assert_array_equal(
        js.frame.names_to_idxs(model=model, frame_names=model.frame_names()),
        jnp.arange(n_l, n_l + n_f),
    )

    assert (
        js.frame.idxs_to_names(
            model=model,
            frame_indices=tuple(
                js.frame.names_to_idxs(
                    model=model, frame_names=model.frame_names()
                ).tolist()
            ),
        )
        == model.frame_names()
    )

    with pytest.raises(ValueError):
        _ = js.frame.name_to_idx(model=model, frame_name="non_existent_frame")

    with pytest.raises(JaxRuntimeError):
        _ = js.frame.idx_to_name(model=model, frame_index=-1)

    with pytest.raises(JaxRuntimeError):
        _ = js.frame.idx_to_name(model=model, frame_index=n_l - 1)

    with pytest.raises(JaxRuntimeError):
        _ = js.frame.idx_to_name(model=model, frame_index=n_l + n_f)

    with pytest.raises(JaxRuntimeError):
        _ = js.frame.idx_of_parent_link(model=model, frame_index=-1)

    with pytest.raises(JaxRuntimeError):
        _ = js.frame.idx_of_parent_link(model=model, frame_index=n_l - 1)

    with pytest.raises(JaxRuntimeError):
        _ = js.frame.idx_of_parent_link(model=model, frame_index=n_l + n_f)


def test_frame_transforms(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=VelRepr.Inertial
    )

    kin_dyn = utils.build_kindyncomputations_from_jaxsim_model(model=model, data=data)

    # Get all names of frames in the iDynTree model.
    frame_names = [
        frame.name
        for frame in model.description.frames
        if frame.name in kin_dyn.frame_names()
    ]

    # Skip some entry of models with many frames.
    frame_names = [
        name
        for name in frame_names
        if "skin" not in name or "laser" not in name or "depth" not in name
    ]

    # Get indices of frames.
    frame_indices = tuple(
        frame.index
        for frame in model.description.frames
        if frame.index is not None and frame.name in frame_names
    )

    # =====
    # Tests
    # =====

    assert len(frame_indices) == len(frame_names)

    for frame_name in frame_names:

        W_H_F_js = js.frame.transform(
            model=model,
            data=data,
            frame_index=js.frame.name_to_idx(model=model, frame_name=frame_name),
        )
        W_H_F_idt = kin_dyn.frame_transform(frame_name=frame_name)
        assert_allclose(
            W_H_F_js, W_H_F_idt, atol=1e-6, err_msg=f"Mismatch in {frame_name}"
        )


def test_frame_jacobians(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    kin_dyn = utils.build_kindyncomputations_from_jaxsim_model(model=model, data=data)

    # Get all names of frames in the iDynTree model.
    frame_names = [
        frame.name
        for frame in model.description.frames
        if frame.name in kin_dyn.frame_names()
    ]

    # Lower the number of frames for models with many frames.
    if model.name().lower() == "ergocub":
        assert any("sole" in name for name in frame_names)
        frame_names = [name for name in frame_names if "sole" in name]

    # Get indices of frames.
    frame_indices = tuple(
        frame.index
        for frame in model.description.frames
        if frame.index is not None and frame.name in frame_names
    )

    # =====
    # Tests
    # =====

    assert len(frame_indices) == len(frame_names)

    for frame_name, frame_index in zip(frame_names, frame_indices, strict=True):

        J_WL_js = js.frame.jacobian(model=model, data=data, frame_index=frame_index)
        J_WL_idt = kin_dyn.jacobian_frame(frame_name=frame_name)
        assert_allclose(J_WL_js, J_WL_idt, err_msg=f"Mismatch in {frame_name}")

    for frame_name, frame_index in zip(frame_names, frame_indices, strict=True):

        v_WF_idt = kin_dyn.frame_velocity(frame_name=frame_name)
        v_WF_js = js.frame.velocity(model=model, data=data, frame_index=frame_index)
        assert_allclose(v_WF_js, v_WF_idt, err_msg=f"Mismatch in {frame_name}")


def test_frame_jacobian_derivative(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):
    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    kin_dyn = utils.build_kindyncomputations_from_jaxsim_model(model=model, data=data)

    # Get all names of frames in the iDynTree model.
    frame_names = [
        frame.name
        for frame in model.description.frames
        if frame.name in kin_dyn.frame_names()
    ]

    # Skip some entry of models with many frames.
    frame_names = [
        name
        for name in frame_names
        if "skin" not in name or "laser" not in name or "depth" not in name
    ]

    frame_idxs = js.frame.names_to_idxs(model=model, frame_names=tuple(frame_names))

    # ===============
    # Test against AD
    # ===============

    # Get the generalized velocity.
    I_ν = data.generalized_velocity()

    # Compute J̇.
    O_J̇_WF_I = jax.vmap(
        lambda frame_index: js.frame.jacobian_derivative(
            model=model, data=data, frame_index=frame_index
        )
    )(frame_idxs)

    assert O_J̇_WF_I.shape == (len(frame_names), 6, 6 + model.dofs())

    # Compute the plain Jacobian.
    # This function will be used to compute the Jacobian derivative with AD.
    def J(q, frame_idxs) -> jax.Array:
        data_ad = js.data.JaxSimModelData.build(
            model=model,
            velocity_representation=data.velocity_representation,
            base_position=q[:3],
            base_quaternion=q[3:7],
            joint_positions=q[7:],
        )

        O_J_ad_WF_I = jax.vmap(
            lambda model, data, frame_index: js.frame.jacobian(
                model=model, data=data, frame_index=frame_index
            ),
            in_axes=(None, None, 0),
        )(model, data_ad, frame_idxs)

        return O_J_ad_WF_I

    def compute_q(data: js.data.JaxSimModelData) -> jax.Array:
        q = jnp.hstack(
            [
                data.base_position,
                data.base_orientation,
                data.joint_positions,
            ]
        )

        return q

    def compute_q̇(data: js.data.JaxSimModelData) -> jax.Array:
        B_ω_WB = data.base_velocity(VelRepr.Body)[3:6]

        W_ṗ_B = data.base_velocity(VelRepr.Mixed)[0:3]

        W_Q̇_B = Quaternion.derivative(
            quaternion=data.base_orientation,
            omega=B_ω_WB,
            omega_in_body_fixed=True,
            K=0.0,
        ).squeeze()

        q̇ = jnp.hstack([W_ṗ_B, W_Q̇_B, data.joint_velocities])

        return q̇

    # Compute q and q̇.
    q = compute_q(data)
    q̇ = compute_q̇(data)

    # Compute dJ/dt with AD.
    dJ_dq = jax.jacfwd(J, argnums=0)(q, frame_idxs)
    O_J̇_ad_WF_I = jnp.einsum("ijkq,q->ijk", dJ_dq, q̇)

    assert_allclose(O_J̇_WF_I, O_J̇_ad_WF_I)

    # =====================
    # Test against iDynTree
    # =====================

    # Compute the product J̇ν.
    O_a_bias_WF = jax.vmap(
        lambda O_J̇_WF_I, I_ν: O_J̇_WF_I @ I_ν,
        in_axes=(0, None),
    )(O_J̇_WF_I, I_ν)

    # Compare the two computations.
    for index, name in enumerate(frame_names):
        J̇ν_idt = kin_dyn.frame_bias_acc(frame_name=name)
        J̇ν_js = O_a_bias_WF[index]
        assert_allclose(J̇ν_js, J̇ν_idt, err_msg=f"Mismatch in {name}")
