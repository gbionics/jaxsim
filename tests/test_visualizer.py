import jax
import numpy as np
import pytest
import rod

import jaxsim.api as js
from jaxsim.mujoco import ModelToMjcf, MujocoModelHelper, mujoco_data_from_jaxsim
from jaxsim.mujoco.loaders import MujocoCamera


@pytest.fixture
def mujoco_camera():

    return MujocoCamera.build_from_target_view(
        camera_name="test_camera",
        lookat=(0, 0, 0),
        distance=1,
        azimuth=0,
        elevation=0,
        fovy=45,
        degrees=True,
    )


def test_urdf_loading(jaxsim_model_single_pendulum, mujoco_camera):
    model = jaxsim_model_single_pendulum.built_from

    _ = ModelToMjcf.convert(model=model, cameras=mujoco_camera)


def test_sdf_loading(jaxsim_model_single_pendulum, mujoco_camera):

    model = rod.Sdf.load(sdf=jaxsim_model_single_pendulum.built_from).serialize(
        pretty=True
    )

    _ = ModelToMjcf.convert(model=model, cameras=mujoco_camera)


def test_rod_loading(jaxsim_model_single_pendulum, mujoco_camera):

    model = rod.Sdf.load(sdf=jaxsim_model_single_pendulum.built_from).models()[0]

    _ = ModelToMjcf.convert(model=model, cameras=mujoco_camera)


def test_heightmap(jaxsim_model_single_pendulum, mujoco_camera):

    model = rod.Sdf.load(sdf=jaxsim_model_single_pendulum.built_from).models()[0]

    _ = ModelToMjcf.convert(
        model=model,
        cameras=mujoco_camera,
        heightmap=True,
        heightmap_samples_xy=(51, 51),
    )


def test_inclined_plane(jaxsim_model_single_pendulum, mujoco_camera):

    model = rod.Sdf.load(sdf=jaxsim_model_single_pendulum.built_from).models()[0]

    _ = ModelToMjcf.convert(
        model=model,
        cameras=mujoco_camera,
        plane_normal=(0.3, 0.3, 0.3),
    )


def test_mujoco_model_helper_and_data_from_jaxsim(
    jaxsim_model_box: js.model.JaxSimModel, prng_key: jax.Array
):

    model = jaxsim_model_box
    data = js.data.random_model_data(model=model, key=prng_key)

    mjcf, assets = ModelToMjcf.convert(model=model.built_from)
    mj_helper = MujocoModelHelper.build_from_xml(mjcf_description=mjcf, assets=assets)

    mj_data = mujoco_data_from_jaxsim(
        mujoco_model=mj_helper.model, jaxsim_model=model, jaxsim_data=data
    )
    mj_helper = MujocoModelHelper(model=mj_helper.model, data=mj_data)

    W_H_B = np.array(data.base_transform)
    np.testing.assert_allclose(mj_helper.base_position(), W_H_B[:3, 3], atol=1e-6)
    np.testing.assert_allclose(
        mj_helper.base_orientation(dcm=True), W_H_B[:3, :3], atol=1e-6
    )
    np.testing.assert_allclose(
        mj_helper.body_orientation(body_name=model.base_link(), dcm=True),
        W_H_B[:3, :3],
        atol=1e-6,
    )

    mj_helper.set_base_orientation(orientation=W_H_B[:3, :3], dcm=True)
    np.testing.assert_allclose(
        mj_helper.base_orientation(dcm=True), W_H_B[:3, :3], atol=1e-6
    )
