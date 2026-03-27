import jax.numpy as jnp
import pytest
from jax.errors import JaxRuntimeError
from numpy.testing import assert_array_equal

import jaxsim.api as js


def test_joint_index(
    jaxsim_models_types: js.model.JaxSimModel,
):

    model = jaxsim_models_types

    # =====
    # Tests
    # =====

    for idx, joint_name in enumerate(model.joint_names()):
        assert js.joint.name_to_idx(model=model, joint_name=joint_name) == idx
        assert js.joint.idx_to_name(model=model, joint_index=idx) == joint_name

    # See discussion in https://github.com/gbionics/jaxsim/pull/280
    assert_array_equal(
        js.joint.names_to_idxs(model=model, joint_names=model.joint_names()),
        jnp.arange(model.number_of_joints()),
    )

    assert (
        js.joint.idxs_to_names(
            model=model,
            joint_indices=tuple(
                js.joint.names_to_idxs(
                    model=model, joint_names=model.joint_names()
                ).tolist()
            ),
        )
        == model.joint_names()
    )

    with pytest.raises(ValueError):
        _ = js.joint.name_to_idx(model=model, joint_name="non_existent_joint")

    with pytest.raises(JaxRuntimeError):
        _ = js.joint.idx_to_name(model=model, joint_index=-1)

    with pytest.raises(IndexError):
        _ = js.joint.idx_to_name(model=model, joint_index=model.number_of_joints())
