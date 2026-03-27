import jax

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils
from .utils import assert_allclose


def test_com_properties(
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

    # =====
    # Tests
    # =====

    p_com_idt = kin_dyn.com_position()
    p_com_js = js.com.com_position(model=model, data=data)

    assert_allclose(p_com_idt, p_com_js)

    J_Gh_idt = kin_dyn.centroidal_momentum_jacobian()
    J_Gh_js = js.com.centroidal_momentum_jacobian(model=model, data=data)

    assert_allclose(J_Gh_idt, J_Gh_js)

    h_com_idt = kin_dyn.centroidal_momentum()
    h_com_js = js.com.centroidal_momentum(model=model, data=data)

    assert_allclose(h_com_idt, h_com_js)

    M_com_locked_idt = kin_dyn.locked_centroidal_spatial_inertia()
    M_com_locked_js = js.com.locked_centroidal_spatial_inertia(model=model, data=data)

    assert_allclose(M_com_locked_idt, M_com_locked_js)

    J_avg_com_idt = kin_dyn.average_centroidal_velocity_jacobian()
    J_avg_com_js = js.com.average_centroidal_velocity_jacobian(model=model, data=data)

    assert_allclose(J_avg_com_idt, J_avg_com_js)

    v_avg_com_idt = kin_dyn.average_centroidal_velocity()
    v_avg_com_js = js.com.average_centroidal_velocity(model=model, data=data)

    assert_allclose(v_avg_com_idt, v_avg_com_js)

    # https://github.com/gbionics/jaxsim/pull/117#discussion_r1535486123
    if data.velocity_representation is not VelRepr.Body:
        vl_com_idt = kin_dyn.com_velocity()
        vl_com_js = js.com.com_linear_velocity(model=model, data=data)

        assert_allclose(vl_com_idt, vl_com_js)

    # iDynTree provides the bias acceleration in G[W] frame regardless of the velocity
    # representation. JaxSim, instead, returns the bias acceleration in G[B] when the
    # active representation is VelRepr.Body.
    if data.velocity_representation is not VelRepr.Body:
        G_v̇_bias_WG_idt = kin_dyn.com_bias_acceleration()
        G_v̇_bias_WG_js = js.com.bias_acceleration(model=model, data=data)

        assert_allclose(G_v̇_bias_WG_idt, G_v̇_bias_WG_js)
