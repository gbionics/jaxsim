import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import rod

import jaxsim.api as js
import jaxsim.math
from jaxsim import VelRepr

from . import utils
from .utils import assert_allclose


def test_model_creation_and_reduction(
    jaxsim_model_ergocub: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model_full = jaxsim_model_ergocub

    _, subkey = jax.random.split(prng_key, num=2)
    data_full = js.data.random_model_data(
        model=model_full,
        key=subkey,
        velocity_representation=VelRepr.Inertial,
        base_pos_bounds=((0, 0, 0.8), (0, 0, 0.8)),
    )

    # =====
    # Tests
    # =====

    # Check that the data of the full model is valid.
    assert data_full.valid(model=model_full)

    # Build the ROD model from the original description.
    assert isinstance(model_full.built_from, str | pathlib.Path)
    rod_sdf = rod.Sdf.load(sdf=model_full.built_from)
    assert len(rod_sdf.models()) == 1

    # Get all non-fixed joint names from the description.
    joint_names_in_description = [
        j.name for j in rod_sdf.models()[0].joints() if j.type != "fixed"
    ]

    # Check that all non-fixed joints are in the full model.
    assert set(joint_names_in_description) == set(model_full.joint_names())

    # ================
    # Reduce the model
    # ================

    # Get the names of the joints to keep in the reduced model.
    reduced_joints = tuple(
        j
        for j in model_full.joint_names()
        if "camera" not in j
        and "neck" not in j
        and "wrist" not in j
        and "thumb" not in j
        and "index" not in j
        and "middle" not in j
        and "ring" not in j
        and "pinkie" not in j
        #
        and "elbow" not in j
        and "shoulder" not in j
        and "torso" not in j
        and "r_knee" not in j
    )

    # Reduce the model.
    # Note: here we also specify a non-zero position of the removed joints.
    # The process should take into account the corresponding joint transforms
    # when the link-joint-link chains are lumped together.
    model_reduced = js.model.reduce(
        model=model_full,
        considered_joints=reduced_joints,
        locked_joint_positions=dict(
            zip(
                model_full.joint_names(),
                data_full.joint_positions.tolist(),
                strict=True,
            )
        ),
    )

    # Check DoFs.
    assert model_full.dofs() != model_reduced.dofs()

    # Check that all non-fixed joints are in the reduced model.
    assert set(reduced_joints) == set(model_reduced.joint_names())

    # Check that the reduced model maintains the same terrain of the full model.
    assert model_full.terrain == model_reduced.terrain

    # Check that the reduced model maintains the same contact model of the full model.
    assert model_full.contact_model == model_reduced.contact_model

    # Check that the reduced model maintains the same integration step of the full model.
    assert model_full.time_step == model_reduced.time_step

    joint_idxs = js.joint.names_to_idxs(
        model=model_full, joint_names=model_reduced.joint_names()
    )

    # Build the data of the reduced model.
    data_reduced = js.data.JaxSimModelData.build(
        model=model_reduced,
        base_position=data_full.base_position,
        base_quaternion=data_full.base_orientation,
        joint_positions=data_full.joint_positions[joint_idxs],
        base_linear_velocity=data_full.base_velocity()[0:3],
        base_angular_velocity=data_full.base_velocity()[3:6],
        joint_velocities=data_full.joint_velocities[joint_idxs],
        velocity_representation=data_full.velocity_representation,
    )

    # Check that the reduced model data is valid.
    assert not data_reduced.valid(model=model_full)
    assert data_reduced.valid(model=model_reduced)

    # Check that the total mass is preserved.
    assert_allclose(
        js.model.total_mass(model=model_full), js.model.total_mass(model=model_reduced)
    )

    # Check that the CoM position is preserved.
    assert_allclose(
        js.com.com_position(model=model_full, data=data_full),
        js.com.com_position(model=model_reduced, data=data_reduced),
        atol=1e-6,
    )

    # Check that joint serialization works.
    assert_allclose(data_full.joint_positions[joint_idxs], data_reduced.joint_positions)
    assert_allclose(
        data_full.joint_velocities[joint_idxs], data_reduced.joint_velocities
    )

    # Check that link transforms are preserved.
    for link_name in model_reduced.link_names():
        W_H_L_full = js.link.transform(
            model=model_full,
            data=data_full,
            link_index=js.link.name_to_idx(model=model_full, link_name=link_name),
        )
        W_H_L_reduced = js.link.transform(
            model=model_reduced,
            data=data_reduced,
            link_index=js.link.name_to_idx(model=model_reduced, link_name=link_name),
        )
        assert_allclose(W_H_L_full, W_H_L_reduced)

    # Check that collidable point positions are preserved.
    assert_allclose(
        js.contact.collidable_point_positions(model=model_full, data=data_full),
        js.contact.collidable_point_positions(model=model_reduced, data=data_reduced),
    )

    # =====================
    # Test against iDynTree
    # =====================

    kin_dyn_full = utils.build_kindyncomputations_from_jaxsim_model(
        model=model_full, data=data_full
    )

    kin_dyn_reduced = utils.build_kindyncomputations_from_jaxsim_model(
        model=model_reduced, data=data_reduced
    )

    # Check that the total mass is preserved.
    assert_allclose(kin_dyn_full.total_mass(), kin_dyn_reduced.total_mass())

    # Check that the CoM position match.
    assert_allclose(kin_dyn_full.com_position(), kin_dyn_reduced.com_position())
    assert_allclose(
        kin_dyn_full.com_position(),
        js.com.com_position(model=model_reduced, data=data_reduced),
    )
    # Check that link transforms match.
    for link_name in model_reduced.link_names():

        assert_allclose(
            kin_dyn_reduced.frame_transform(frame_name=link_name),
            kin_dyn_full.frame_transform(frame_name=link_name),
            err_msg=f"Mismatch in link {link_name}",
        )

        assert_allclose(
            kin_dyn_reduced.frame_transform(frame_name=link_name),
            js.link.transform(
                model=model_reduced,
                data=data_reduced,
                link_index=js.link.name_to_idx(
                    model=model_reduced, link_name=link_name
                ),
            ),
            err_msg=f"Mismatch in link {link_name}",
        )

    # Check that frame transforms match.
    for frame_name in model_reduced.frame_names():

        if frame_name not in kin_dyn_reduced.frame_names():
            continue

        # Skip some entry of models with many frames.
        if "skin" in frame_name or "laser" in frame_name or "depth" in frame_name:
            continue

        assert_allclose(
            kin_dyn_reduced.frame_transform(frame_name=frame_name),
            kin_dyn_full.frame_transform(frame_name=frame_name),
            err_msg=f"Mismatch in frame {frame_name}",
        )

        assert_allclose(
            kin_dyn_reduced.frame_transform(frame_name=frame_name),
            js.frame.transform(
                model=model_reduced,
                data=data_reduced,
                frame_index=js.frame.name_to_idx(
                    model=model_reduced, frame_name=frame_name
                ),
            ),
            err_msg=f"Mismatch in frame {frame_name}",
        )


def test_model_properties(
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

    m_idt = kin_dyn.total_mass()
    m_js = js.model.total_mass(model=model)
    assert_allclose(m_idt, m_js)

    J_Bh_idt = kin_dyn.total_momentum_jacobian()
    J_Bh_js = js.model.total_momentum_jacobian(model=model, data=data)
    assert_allclose(J_Bh_idt, J_Bh_js)

    h_tot_idt = kin_dyn.total_momentum()
    h_tot_js = js.model.total_momentum(model=model, data=data)
    assert_allclose(h_tot_idt, h_tot_js)

    M_locked_idt = kin_dyn.locked_spatial_inertia()
    M_locked_js = js.model.locked_spatial_inertia(model=model, data=data)
    assert_allclose(M_locked_idt, M_locked_js)

    J_avg_idt = kin_dyn.average_velocity_jacobian()
    J_avg_js = js.model.average_velocity_jacobian(model=model, data=data)
    assert_allclose(J_avg_idt, J_avg_js)

    v_avg_idt = kin_dyn.average_velocity()
    v_avg_js = js.model.average_velocity(model=model, data=data)
    assert_allclose(v_avg_idt, v_avg_js)


def test_model_rbda(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
    velocity_representation: VelRepr,
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

    # Support both fixed-base and floating-base models by slicing the first six rows.
    sl = np.s_[0:] if model.floating_base() else np.s_[6:]

    # Mass matrix
    M_idt = kin_dyn.mass_matrix()
    M_js = js.model.free_floating_mass_matrix(model=model, data=data)
    assert_allclose(M_idt[sl, sl], M_js[sl, sl])

    # Gravity forces
    g_idt = kin_dyn.gravity_forces()
    g_js = js.model.free_floating_gravity_forces(model=model, data=data)
    assert_allclose(g_idt[sl], g_js[sl])

    # Bias forces
    h_idt = kin_dyn.bias_forces()
    h_js = js.model.free_floating_bias_forces(model=model, data=data)
    assert_allclose(h_idt[sl], h_js[sl])

    # Forward kinematics
    HH_js = data._link_transforms
    HH_idt = jnp.stack(
        [kin_dyn.frame_transform(frame_name=name) for name in model.link_names()]
    )
    assert_allclose(HH_idt, HH_js)

    # Bias accelerations
    Jν_js = js.model.link_bias_accelerations(model=model, data=data)
    Jν_idt = jnp.stack(
        [kin_dyn.frame_bias_acc(frame_name=name) for name in model.link_names()]
    )
    assert_allclose(Jν_idt, Jν_js)

    # Mass matrix inverse via RBDA
    M_inv_js = js.model.free_floating_mass_matrix_inverse(model=model, data=data)
    M_inv_idt = jnp.linalg.inv(M_idt)
    assert_allclose(M_inv_idt[sl, sl], M_inv_js[sl, sl])


def test_model_jacobian(
    jaxsim_models_types: js.model.JaxSimModel,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=VelRepr.Inertial
    )

    # =====
    # Tests
    # =====

    # Create random references (joint torques and link forces)
    _, subkey1, subkey2 = jax.random.split(key, num=3)
    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=10 * jax.random.uniform(subkey1, shape=(model.dofs(),)),
        link_forces=jax.random.uniform(subkey2, shape=(model.number_of_links(), 6)),
        data=data,
        velocity_representation=data.velocity_representation,
    )

    # Remove the force applied to the base link if the model is fixed-base
    if not model.floating_base():
        references = references.apply_link_forces(
            forces=jnp.atleast_2d(jnp.zeros(6)),
            model=model,
            data=data,
            link_names=(model.base_link(),),
            additive=False,
        )

    # Get the J.T @ f product in inertial-fixed input/output representation.
    # We use doubly right-trivialized jacobian with inertial-fixed 6D forces.
    f = references.link_forces(
        model=model, data=data, output_representation=VelRepr.Inertial
    )
    assert_allclose(f, references._link_forces)

    J = js.model.generalized_free_floating_jacobian(
        model=model, data=data, input_representation=VelRepr.Inertial
    )
    JTf_inertial = jnp.einsum("l6g,l6->g", J, f)

    for vel_repr in [VelRepr.Body, VelRepr.Mixed]:
        # Get the jacobian having an inertial-fixed input representation (so that
        # it computes the same quantity computed above) and an output representation
        # compatible with the frame in which the external forces are expressed.
        J = js.model.generalized_free_floating_jacobian(
            model=model,
            data=data,
            input_representation=VelRepr.Inertial,
            output_vel_repr=vel_repr,
        )

        # Get the forces in the tested representation and compute the product
        # O_J_WL_W.T @ O_f, producing a generalized acceleration in W.
        # The resulting acceleration can be tested again the one computed before.
        f = references.link_forces(
            model=model, data=data, output_representation=vel_repr
        )
        JTf_other = jnp.einsum("l6g,l6->g", J, f)
        assert_allclose(JTf_inertial, JTf_other, err_msg=vel_repr.name)


def test_coriolis_matrix(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    _, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    # =====
    # Tests
    # =====

    I_ν = data.generalized_velocity()
    C = js.model.free_floating_coriolis_matrix(model=model, data=data)

    h = js.model.free_floating_bias_forces(model=model, data=data)
    g = js.model.free_floating_gravity_forces(model=model, data=data)
    Cν = h - g

    assert_allclose(C @ I_ν, Cν)

    # Compute the free-floating mass matrix.
    # This function will be used to compute the Ṁ with AD.
    # Given q, computing Ṁ by AD-ing this function should work out-of-the-box with
    # all velocity representations, that are handled internally when computing M.
    def M(q) -> jax.Array:

        data_ad = js.data.JaxSimModelData.build(
            model=model,
            velocity_representation=data.velocity_representation,
            base_position=q[:3],
            base_quaternion=q[3:7],
            joint_positions=q[7:],
        )

        M = js.model.free_floating_mass_matrix(model=model, data=data_ad)

        return M

    def compute_q(data: js.data.JaxSimModelData) -> jax.Array:

        q = jnp.hstack(
            [data.base_position, data.base_orientation, data.joint_positions]
        )

        return q

    def compute_q̇(data: js.data.JaxSimModelData) -> jax.Array:

        B_ω_WB = data.base_velocity(VelRepr.Body)[3:6]
        W_ṗ_B = data.base_velocity(VelRepr.Mixed)[0:3]

        W_Q̇_B = jaxsim.math.Quaternion.derivative(
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

    # Compute Ṁ with AD.
    dM_dq = jax.jacfwd(M, argnums=0)(q)
    Ṁ = jnp.einsum("ijq,q->ij", dM_dq, q̇)

    # We need to zero the blocks projecting joint variables to the base configuration
    # for fixed-base models.
    if not model.floating_base():
        Ṁ = Ṁ.at[0:6, 6:].set(0)
        Ṁ = Ṁ.at[6:, 0:6].set(0)

    # Ensure that (Ṁ - 2C) is skew symmetric.
    assert_allclose(Ṁ - C - C.T, 0.0)


def test_model_fd_id_consistency(
    jaxsim_models_types: js.model.JaxSimModel,
    velocity_representation: VelRepr,
    prng_key: jax.Array,
):

    model = jaxsim_models_types

    key, subkey = jax.random.split(prng_key, num=2)
    data = js.data.random_model_data(
        model=model, key=subkey, velocity_representation=velocity_representation
    )

    # =====
    # Tests
    # =====

    # Create random references (joint torques and link forces).
    _, subkey1, subkey2 = jax.random.split(key, num=3)
    references = js.references.JaxSimModelReferences.build(
        model=model,
        joint_force_references=10 * jax.random.uniform(subkey1, shape=(model.dofs(),)),
        link_forces=jax.random.uniform(subkey2, shape=(model.number_of_links(), 6)),
        data=data,
        velocity_representation=data.velocity_representation,
    )

    # Remove the force applied to the base link if the model is fixed-base.
    if not model.floating_base():
        references = references.apply_link_forces(
            forces=jnp.atleast_2d(jnp.zeros(6)),
            model=model,
            data=data,
            link_names=(model.base_link(),),
            additive=False,
        )

    # Compute forward dynamics with ABA.
    v̇_WB_aba, s̈_aba = js.model.forward_dynamics_aba(
        model=model,
        data=data,
        joint_forces=references.joint_force_references(),
        link_forces=references.link_forces(model=model, data=data),
    )

    # Compute forward dynamics with CRB.
    v̇_WB_crb, s̈_crb = js.model.forward_dynamics_crb(
        model=model,
        data=data,
        joint_forces=references.joint_force_references(),
        link_forces=references.link_forces(model=model, data=data),
    )

    assert_allclose(s̈_aba, s̈_crb)
    assert_allclose(v̇_WB_aba, v̇_WB_crb)

    # Compute inverse dynamics with the quantities computed by forward dynamics
    fB_id, τ_id = js.model.inverse_dynamics(
        model=model,
        data=data,
        joint_accelerations=s̈_aba,
        base_acceleration=v̇_WB_aba,
        link_forces=references.link_forces(model=model, data=data),
    )

    # Check consistency between FD and ID
    assert_allclose(τ_id, references.joint_force_references(model=model))
    assert_allclose(fB_id, 0.0)

    if model.floating_base():
        # If we remove the base 6D force from the inputs, we should find it as output.
        fB_id, τ_id = js.model.inverse_dynamics(
            model=model,
            data=data,
            joint_accelerations=s̈_aba,
            base_acceleration=v̇_WB_aba,
            link_forces=references.link_forces(model=model, data=data)
            .at[0]
            .set(jnp.zeros(6)),
        )

        assert_allclose(τ_id, references.joint_force_references(model=model))
        assert_allclose(fB_id, references.link_forces(model=model, data=data)[0])
