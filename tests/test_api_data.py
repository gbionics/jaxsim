import jax
from numpy.testing import assert_raises

import jaxsim.api as js
from jaxsim import VelRepr

from . import utils
from .utils import assert_allclose


def test_data_valid(
    jaxsim_models_all: js.model.JaxSimModel,
):

    model = jaxsim_models_all
    data = js.data.JaxSimModelData.build(model=model)

    assert data.valid(model=model)


def test_data_change_velocity_representation(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=VelRepr.Inertial
    )

    # =====
    # Tests
    # =====

    kin_dyn_inertial = utils.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data
    )

    kin_dyn_mixed = utils.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data, vel_repr=VelRepr.Mixed
    )

    kin_dyn_body = utils.build_kindyncomputations_from_jaxsim_model(
        model=model, data=data, vel_repr=VelRepr.Body
    )

    assert_allclose(data.base_velocity(), kin_dyn_inertial.base_velocity())

    if not model.floating_base():
        return

    assert_allclose(data.base_velocity(VelRepr.Mixed), kin_dyn_mixed.base_velocity())
    assert_raises(
        AssertionError,
        assert_allclose,
        data.base_velocity(VelRepr.Mixed)[0:3],
        data._base_linear_velocity,
    )
    assert_allclose(data.base_velocity(VelRepr.Mixed)[3:6], data._base_angular_velocity)

    assert_allclose(data.base_velocity(VelRepr.Body), kin_dyn_body.base_velocity())
    assert_raises(
        AssertionError,
        assert_allclose,
        data.base_velocity(VelRepr.Body)[0:3],
        data._base_linear_velocity,
    )
    assert_raises(
        AssertionError,
        assert_allclose,
        data.base_velocity(VelRepr.Body)[3:6],
        data._base_angular_velocity,
    )
