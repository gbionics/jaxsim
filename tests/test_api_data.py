import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_raises

import jaxsim.api as js
from jaxsim import VelRepr
from jaxsim.utils import Mutability

from . import utils
from .utils import assert_allclose


def test_data_valid(
    jaxsim_models_all: js.model.JaxSimModel,
):

    model = jaxsim_models_all
    data = js.data.JaxSimModelData.build(model=model)

    assert data.valid(model=model)


def test_data_switch_velocity_representation(
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

    new_base_linear_velocity = jnp.array([1.0, -2.0, 3.0])
    old_base_linear_velocity = data._base_linear_velocity

    # The following should not change the original `data` object since it raises.
    with pytest.raises(RuntimeError):
        with data.switch_velocity_representation(
            velocity_representation=VelRepr.Inertial
        ):
            with data.mutable_context(mutability=Mutability.MUTABLE):
                data._base_linear_velocity = new_base_linear_velocity
            raise RuntimeError("This is raised on purpose inside this context")

    assert_allclose(data._base_linear_velocity, old_base_linear_velocity)

    # The following instead should result to an updated `data` object.
    with (
        data.switch_velocity_representation(velocity_representation=VelRepr.Inertial),
        data.mutable_context(mutability=Mutability.MUTABLE),
    ):
        data._base_linear_velocity = new_base_linear_velocity

    assert_allclose(data._base_linear_velocity, new_base_linear_velocity)


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

    with data.switch_velocity_representation(VelRepr.Mixed):
        kin_dyn_mixed = utils.build_kindyncomputations_from_jaxsim_model(
            model=model, data=data
        )

    with data.switch_velocity_representation(VelRepr.Body):
        kin_dyn_body = utils.build_kindyncomputations_from_jaxsim_model(
            model=model, data=data
        )

    assert_allclose(data.base_velocity, kin_dyn_inertial.base_velocity())

    if not model.floating_base():
        return

    with data.switch_velocity_representation(VelRepr.Mixed):
        assert_allclose(data.base_velocity, kin_dyn_mixed.base_velocity())
        assert_raises(
            AssertionError,
            assert_allclose,
            data.base_velocity[0:3],
            data._base_linear_velocity,
        )
        assert_allclose(data.base_velocity[3:6], data._base_angular_velocity)

    with data.switch_velocity_representation(VelRepr.Body):
        assert_allclose(data.base_velocity, kin_dyn_body.base_velocity())
        assert_raises(
            AssertionError,
            assert_allclose,
            data.base_velocity[0:3],
            data._base_linear_velocity,
        )
        assert_raises(
            AssertionError,
            assert_allclose,
            data.base_velocity[3:6],
            data._base_angular_velocity,
        )


def _one_joint_model_description(lower: float, upper: float) -> str:
    """Build a two-link model whose only joint has the given position limits."""

    inertial = (
        "<inertial><mass>1.0</mass><inertia>"
        "<ixx>0.1</ixx><ixy>0</ixy><ixz>0</ixz>"
        "<iyy>0.1</iyy><iyz>0</iyz><izz>0.1</izz>"
        "</inertia></inertial>"
    )

    return f"""<sdf version="1.7">
      <model name="one_joint">
        <link name="base">{inertial}</link>
        <link name="arm"><pose>0 0 0.5 0 0 0</pose>{inertial}</link>
        <joint name="shoulder" type="revolute">
          <parent>base</parent>
          <child>arm</child>
          <axis>
            <xyz>0 0 1</xyz>
            <limit>
              <lower>{lower}</lower>
              <upper>{upper}</upper>
              <effort>10</effort>
              <velocity>10</velocity>
            </limit>
          </axis>
        </joint>
      </model>
    </sdf>"""


@pytest.mark.parametrize(
    "lower, upper, expected_position",
    [
        # Zero is below the limits, so the joint starts at its lower limit
        (0.5, 1.5, 0.5),
        # Zero is above the limits, so the joint starts at its upper limit
        (-1.5, -0.5, -0.5),
        # Zero is a valid position, so the joint starts there as it always did
        (-1.0, 1.0, 0.0),
    ],
)
def test_data_default_joint_positions_are_within_limits(
    lower: float,
    upper: float,
    expected_position: float,
):

    model = js.model.JaxSimModel.build_from_model_description(
        model_description=_one_joint_model_description(lower, upper),
        is_urdf=False,
    )

    # =====
    # Tests
    # =====

    data = js.data.JaxSimModelData.build(model=model)

    s_min, s_max = js.joint.position_limits(model=model)
    assert bool(jnp.all(data.joint_positions >= s_min))
    assert bool(jnp.all(data.joint_positions <= s_max))

    # The default stays as close to the zero configuration as the joint allows
    assert_allclose(data.joint_positions, jnp.array([expected_position]))
