import jax
import jax.numpy as jnp
import pytest
from jax.errors import JaxRuntimeError
from numpy.testing import assert_array_equal

import jaxsim.api as js
import jaxsim.math
from jaxsim import VelRepr

from . import utils
from .utils import assert_allclose


def test_link_index(
    jaxsim_models_types: js.model.JaxSimModel,
):

    model = jaxsim_models_types

    # =====
    # Tests
    # =====

    for idx, link_name in enumerate(model.link_names()):
        assert js.link.name_to_idx(model=model, link_name=link_name) == idx
        assert js.link.idx_to_name(model=model, link_index=idx) == link_name

    # See discussion in https://github.com/ami-iit/jaxsim/pull/280
    assert_array_equal(
        js.link.names_to_idxs(model=model, link_names=model.link_names()),
        jnp.arange(model.number_of_links()),
    )

    assert (
        js.link.idxs_to_names(
            model=model,
            link_indices=tuple(
                js.link.names_to_idxs(
                    model=model, link_names=model.link_names()
                ).tolist()
            ),
        )
        == model.link_names()
    )

    with pytest.raises(ValueError):
        _ = js.link.name_to_idx(model=model, link_name="non_existent_link")

    with pytest.raises(JaxRuntimeError):
        _ = js.link.idx_to_name(model=model, link_index=-1)

    with pytest.raises(IndexError):
        _ = js.link.idx_to_name(model=model, link_index=model.number_of_links())


def test_link_inertial_properties(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=VelRepr.Inertial,
    )

    kin_dyn = utils.build_kindyncomputations_from_jaxsim_model(model=model, data=data)

    # =====
    # Tests
    # =====

    for link_name, link_idx in zip(
        model.link_names(),
        jnp.arange(model.number_of_links()),
        strict=True,
    ):
        if link_name == model.base_link():
            continue

        assert_allclose(
            js.link.mass(model=model, link_index=link_idx),
            kin_dyn.link_mass(link_name=link_name),
            err_msg=f"Mismatch in {link_name}",
        )

        assert_allclose(
            js.link.spatial_inertia(model=model, link_index=link_idx),
            kin_dyn.link_spatial_inertia(link_name=link_name),
            err_msg=f"Mismatch in {link_name}",
        )


def test_link_transforms(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=VelRepr.Inertial,
    )

    kin_dyn = utils.build_kindyncomputations_from_jaxsim_model(model=model, data=data)

    # =====
    # Tests
    # =====

    W_H_LL_model = data._link_transforms

    W_H_LL_links = jax.vmap(
        lambda idx: js.link.transform(model=model, data=data, link_index=idx)
    )(jnp.arange(model.number_of_links()))

    assert_allclose(W_H_LL_model, W_H_LL_links)

    for W_H_L, link_name in zip(W_H_LL_links, model.link_names(), strict=True):

        assert_allclose(
            W_H_L,
            kin_dyn.frame_transform(frame_name=link_name),
            err_msg=f"Mismatch in {link_name}",
        )


def test_link_jacobians(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=velocity_representation,
    )

    kin_dyn = utils.build_kindyncomputations_from_jaxsim_model(model=model, data=data)

    # =====
    # Tests
    # =====

    J_WL_links = jax.vmap(
        lambda idx: js.link.jacobian(model=model, data=data, link_index=idx)
    )(jnp.arange(model.number_of_links()))

    for J_WL, link_name in zip(J_WL_links, model.link_names(), strict=True):
        assert_allclose(
            J_WL,
            kin_dyn.jacobian_frame(frame_name=link_name),
            err_msg=f"Mismatch in {link_name}",
        )

    # The following is true only in inertial-fixed representation.
    J_WL_model = js.model.generalized_free_floating_jacobian(model=model, data=data)

    assert_allclose(J_WL_model, J_WL_links)

    for link_name, link_idx in zip(
        model.link_names(),
        jnp.arange(model.number_of_links()),
        strict=True,
    ):
        v_WL_idt = kin_dyn.frame_velocity(frame_name=link_name)
        v_WL_js = js.link.velocity(model=model, data=data, link_index=link_idx)

        assert_allclose(v_WL_js, v_WL_idt, err_msg=f"Mismatch in {link_name}")

    # Test conversion to a different output velocity representation.
    for other_repr in {VelRepr.Inertial, VelRepr.Body, VelRepr.Mixed}.difference(
        {data.velocity_representation}
    ):

        with data.switch_velocity_representation(other_repr):
            kin_dyn_other_repr = utils.build_kindyncomputations_from_jaxsim_model(
                model=model, data=data
            )

        for link_name, link_idx in zip(
            model.link_names(),
            jnp.arange(model.number_of_links()),
            strict=True,
        ):
            v_WL_idt = kin_dyn_other_repr.frame_velocity(frame_name=link_name)
            v_WL_js = js.link.velocity(
                model=model, data=data, link_index=link_idx, output_vel_repr=other_repr
            )

            assert_allclose(v_WL_js, v_WL_idt, err_msg=f"Mismatch in {link_name}")


def test_link_bias_acceleration(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=velocity_representation,
    )

    kin_dyn = utils.build_kindyncomputations_from_jaxsim_model(model=model, data=data)

    # =====
    # Tests
    # =====

    for name, index in zip(
        model.link_names(),
        jnp.arange(model.number_of_links()),
        strict=True,
    ):
        Jν_idt = kin_dyn.frame_bias_acc(frame_name=name)
        Jν_js = js.link.bias_acceleration(model=model, data=data, link_index=index)

        assert_allclose(Jν_js, Jν_idt, err_msg=f"Mismatch in {name}")

    # Test that the conversion of the link bias acceleration works as expected.
    match data.velocity_representation:

        # We exclude the mixed representation because converting the acceleration is
        # more complex than using the plain 6D transform matrix.
        case VelRepr.Mixed:
            pass

        # Inertial-fixed to body-fixed conversion.
        case VelRepr.Inertial:

            W_H_L = data._link_transforms

            W_a_bias_WL = js.model.link_bias_accelerations(model=model, data=data)

            with data.switch_velocity_representation(VelRepr.Body):

                W_X_L = jax.vmap(
                    lambda W_H_L: jaxsim.math.Adjoint.from_transform(transform=W_H_L)
                )(W_H_L)

                L_a_bias_WL = js.model.link_bias_accelerations(model=model, data=data)

                W_a_bias_WL_converted = jax.vmap(
                    lambda W_X_L, L_a_bias_WL: W_X_L @ L_a_bias_WL
                )(W_X_L, L_a_bias_WL)

            assert_allclose(W_a_bias_WL, W_a_bias_WL_converted)

        # Body-fixed to inertial-fixed conversion.
        case VelRepr.Body:

            W_H_L = data._link_transforms

            L_a_bias_WL = js.model.link_bias_accelerations(model=model, data=data)

            with data.switch_velocity_representation(VelRepr.Inertial):

                L_X_W = jax.vmap(
                    lambda W_H_L: jaxsim.math.Adjoint.from_transform(
                        transform=W_H_L, inverse=True
                    )
                )(W_H_L)

                W_a_bias_WL = js.model.link_bias_accelerations(model=model, data=data)

                L_a_bias_WL_converted = jax.vmap(
                    lambda L_X_W, W_a_bias_WL: L_X_W @ W_a_bias_WL
                )(L_X_W, W_a_bias_WL)

            assert_allclose(L_a_bias_WL, L_a_bias_WL_converted)


def test_link_jacobian_derivative(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model,
        key=subkey,
        velocity_representation=velocity_representation,
    )

    # =====
    # Tests
    # =====

    # Get the generalized velocity.
    I_ν = data.generalized_velocity

    # Compute J̇.
    O_J̇_WL_I = jax.vmap(
        lambda link_index: js.link.jacobian_derivative(
            model=model, data=data, link_index=link_index
        )
    )(jnp.arange(model.number_of_links()))

    # Compute the product J̇ν.
    O_a_bias_WL = js.model.link_bias_accelerations(model=model, data=data)

    # Compare the two computations.
    assert_allclose(jnp.einsum("l6g,g->l6", O_J̇_WL_I, I_ν), O_a_bias_WL)

    # Compute the plain Jacobian.
    # This function will be used to compute the Jacobian derivative with AD.
    # Given q, computing J̇ by AD-ing this function should work out-of-the-box with
    # all velocity representations, that are handled internally when computing J.
    def J(q) -> jax.Array:

        data_ad = js.data.JaxSimModelData.build(
            model=model,
            velocity_representation=data.velocity_representation,
            base_position=q[:3],
            base_quaternion=q[3:7],
            joint_positions=q[7:],
        )

        O_J_WL_I = js.model.generalized_free_floating_jacobian(
            model=model, data=data_ad
        )

        return O_J_WL_I

    def compute_q(data: js.data.JaxSimModelData) -> jax.Array:

        q = jnp.hstack(
            [data.base_position, data.base_orientation, data.joint_positions]
        )

        return q

    def compute_q̇(data: js.data.JaxSimModelData) -> jax.Array:

        with data.switch_velocity_representation(VelRepr.Body):
            B_ω_WB = data.base_velocity[3:6]

        with data.switch_velocity_representation(VelRepr.Mixed):
            W_ṗ_B = data.base_velocity[0:3]

        W_Q̇_B = jaxsim.math.Quaternion.derivative(
            quaternion=data.base_orientation,
            omega=B_ω_WB,
            K=0.0,
        ).squeeze()

        q̇ = jnp.hstack([W_ṗ_B, W_Q̇_B, data.joint_velocities])

        return q̇

    # Compute q and q̇.
    q = compute_q(data)
    q̇ = compute_q̇(data)

    # Compute dJ/dt with AD.
    dJ_dq = jax.jacfwd(J, argnums=0)(q)
    O_J̇_ad_WL_I = jnp.einsum("ijkq,q->ijk", dJ_dq, q̇)

    assert_allclose(O_J̇_WL_I, O_J̇_ad_WL_I)
    assert_allclose(
        jnp.einsum("l6g,g->l6", O_J̇_ad_WL_I, I_ν),
        jnp.einsum("l6g,g->l6", O_J̇_WL_I, I_ν),
    )
