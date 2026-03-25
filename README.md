# JaxSim

**JaxSim** is a **differentiable physics engine** built with JAX, tailored for co-design and robotic learning applications.

<div align="center">
<br/>
<table>
  <tr>
    <th><img src="https://github.com/user-attachments/assets/89d0b4ca-7e0c-4f58-bf3e-9540e35b9a01" style="height:300px; width:400px; object-fit:cover;"></th>
    <th><img src="https://github.com/user-attachments/assets/a909e388-d7b4-4b58-89f1-035da8636d94" style="height:300px; width:400px; object-fit:cover;"></th>
  </tr>
  <tr>
    <th><img src="https://github.com/user-attachments/assets/3692bc06-18ed-406d-80bd-480780346224" style="height:300px; width:400px; object-fit:cover;"></th>
    <th><img src="https://github.com/user-attachments/assets/3356f332-4710-4946-9a82-a8c2305dab88" style="height:300px; width:400px; object-fit:cover;"></th>
  </tr>
</table>
<br/>
</div>

## Features

- Physically consistent differentiability w.r.t. hardware parameters.
- Closed chain dynamics support.
- Reduced-coordinate physics engine for **fixed-base** and **floating-base** robots.
- Fully Python-based, leveraging [JAX][jax] following a functional programming paradigm.
- Seamless execution on CPUs, GPUs, and TPUs.
- Supports JIT compilation and automatic vectorization for high performance.
- Compatible with SDF models and URDF (via [sdformat][sdformat] conversion).

> [!WARNING]
> This project is still experimental. APIs may change between releases without notice.

> [!NOTE]
> JaxSim currently focuses on locomotion applications.
> Only contacts between bodies and smooth ground surfaces are supported.

## How to use it

```python
import pathlib

import icub_models
import jax.numpy as jnp

import jaxsim.api as js

# Load the iCub model
model_path = icub_models.get_model_file("iCubGazeboV2_5")

joints = ('torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
          'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
          'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch',
          'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',
          'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch',
          'r_ankle_roll')

# Build and reduce the model
model_description = pathlib.Path(model_path)

full_model = js.model.JaxSimModel.build_from_model_description(
    model_description=model_description,
)

model = js.model.reduce(model=full_model, considered_joints=joints)

# Get the number of degrees of freedom
ndof = model.dofs()

# Initialize data and simulation
# Note that the default data representation is mixed velocity representation
data = js.data.JaxSimModelData.build(
    model=model, base_position=jnp.array([0.0, 0.0, 1.0])
)

T = jnp.arange(start=0, stop=1.0, step=model.time_step)

tau = jnp.zeros(ndof)

# Simulate
for _ in T:
    data = js.model.step(
        model=model, data=data, link_forces=None, joint_force_references=tau
    )
```

Check the example folder for additional use cases!

[jax]: https://github.com/google/jax/
[sdformat]: https://github.com/gazebosim/sdformat
[notation]: https://research.tue.nl/en/publications/multibody-dynamics-notation-version-2
[passive_viewer_mujoco]: https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer

## Installation

<details>
<summary>With <code>conda</code></summary>

You can install the project using [`conda`][conda] as follows:

```bash
conda install jaxsim -c conda-forge
```

GPU support for JAX will be automatically installed if a compatible GPU is detected.

</details>

<details>
<summary>With <code>pixi</code></summary>

> ### Note
> The minimum version of `pixi` required is `0.39.0`.

Since the `pixi.lock` file is stored using Git LFS, make sure you have [Git LFS](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md) installed and properly configured on your system before installation. After cloning the repository, run:

```bash
git lfs install && git lfs pull
```

This ensures all LFS-tracked files are properly downloaded before you proceed with the installation.

You can add the `jaxsim` dependency in your [`pixi`][pixi] project as follows:

```bash
pixi add jaxsim
```

If you are on Linux and you want to use a `cuda`-powered version of `jax`, remember to add the appropriate line in the [`system-requirements`](https://pixi.sh/latest/reference/pixi_manifest/#the-system-requirements-table) table, i.e. adding

~~~toml
[system-requirements]
cuda = "12"
~~~

if you are using a `pixi.toml` file or

~~~toml
[tool.pixi.system-requirements]
cuda = "12"
~~~

if you are using a `pyproject.toml` file.

</details>

<details>
<summary>With <code>pip</code></summary>

You can install the project using [`pypa/pip`][pip], preferably in a [virtual environment][venv], as follows:

```bash
pip install jaxsim
```

Check [`pyproject.toml`](pyproject.toml) for the complete list of optional dependencies.
You can obtain a full installation using `jaxsim[all]`.

If you need URDF support, follow the [official instructions](https://gazebosim.org/docs) to install Gazebo Sim on your operating system,
making sure to obtain `sdformat ≥ 13.0` and `gz-tools ≥ 2.0`.

You don't need to install the entire Gazebo Sim suite.
For example, on Ubuntu, it is sufficient to install the `libsdformat*` and `gz-tools2` packages.

If you need GPU support, follow the official [installation instructions][jax_gpu] of JAX.

</details>

<details>
<summary>Contributors installation (with <code>conda</code>)</summary>

If you want to contribute to the project, we recommend creating the following `jaxsim` conda environment first:

```bash
conda env create -f environment.yml
```

Then, activate the environment and install the project in editable mode:

```bash
conda activate jaxsim
pip install --no-deps -e .
```

</details>

<details>
<summary>Contributors installation (with <code>pixi</code>)</summary>

> ### Note
> The minimum version of `pixi` required is `0.39.0`.

Since the `pixi.lock` file is stored using Git LFS, make sure you have [Git LFS](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md) installed and properly configured on your system before installation. After cloning the repository, run:

```bash
git lfs install && git lfs pull
```

This ensures all LFS-tracked files are properly downloaded before you proceed with the installation.

You can install the default dependencies of the project using [`pixi`][pixi] as follows:

```bash
pixi install
```

See `pixi task list` for a list of available tasks.

</details>

[conda]: https://anaconda.org/
[pip]: https://github.com/pypa/pip/
[pixi]: https://pixi.sh/
[venv]: https://docs.python.org/3/tutorial/venv.html
[jax_gpu]: https://github.com/google/jax/#installation

## Documentation

The JaxSim API documentation is available at [jaxsim.readthedocs.io][readthedocs].

[readthedocs]: https://jaxsim.readthedocs.io/

## Additional features

Jaxsim can also be used as a multi-body dynamics library! With full support for automatic differentiation of RBDAs (forwards and reverse mode) and automatic differentiation against both kinematic and dynamic parameters.

### Using JaxSim as a multibody dynamics library

```python
import pathlib

import icub_models
import jax.numpy as jnp

import jaxsim.api as js

# Load the iCub model
model_path = icub_models.get_model_file("iCubGazeboV2_5")

joints = ('torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
          'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
          'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch',
          'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',
          'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch',
          'r_ankle_roll')

# Build and reduce the model
model_description = pathlib.Path(model_path)

full_model = js.model.JaxSimModel.build_from_model_description(
    model_description=model_description,
)

model = js.model.reduce(model=full_model, considered_joints=joints)

# Initialize model data
data = js.data.JaxSimModelData.build(
    model=model,
    base_position=jnp.array([0.0, 0.0, 1.0]),
)

# Frame and dynamics computations
frame_index = js.frame.name_to_idx(model=model, frame_name="l_foot")

# Frame transformation
W_H_F = js.frame.transform(
    model=model, data=data, frame_index=frame_index
)

# Frame Jacobian
W_J_F = js.frame.jacobian(
    model=model, data=data, frame_index=frame_index
)

# Dynamics properties
M = js.model.free_floating_mass_matrix(model=model, data=data)      # Mass matrix
h = js.model.free_floating_bias_forces(model=model, data=data)      # Bias forces
g = js.model.free_floating_gravity_forces(model=model, data=data)   # Gravity forces
C = js.model.free_floating_coriolis_matrix(model=model, data=data)  # Coriolis matrix

# Print dynamics results
print(f"{M.shape=} \n{h.shape=} \n{g.shape=} \n{C.shape=}")
```

## Credits

The RBDAs are based on the theory of the [Rigid Body Dynamics Algorithms][RBDA]
book by Roy Featherstone.
The algorithms and some simulation features were inspired by its accompanying [code][spatial_v2].

[RBDA]: https://link.springer.com/book/10.1007/978-1-4899-7560-7
[spatial_v2]: http://royfeatherstone.org/spatial/index.html#spatial-software

The development of JaxSim started in late 2021, inspired by early versions of [`google/brax`][brax].
At that time, Brax was implemented in maximal coordinates, and we wanted a physics engine in reduced coordinates.
We are grateful to the Brax team for their work and for showing the potential of [JAX][jax] in this field.

Brax v2 was later implemented with reduced coordinates, following an approach comparable to JaxSim.
The development then shifted to [MJX][mjx], which provides a JAX-based implementation of the Mujoco APIs.

The main differences between MJX/Brax and JaxSim are as follows:

- JaxSim supports out-of-the-box all SDF models with [Pose Frame Semantics][PFS].
- JaxSim only supports collisions between points rigidly attached to bodies and a compliant ground surface.

[brax]: https://github.com/google/brax
[mjx]: https://mujoco.readthedocs.io/en/3.0.0/mjx.html
[PFS]: http://sdformat.org/tutorials?tut=pose_frame_semantics

## Contributing

We welcome contributions from the community.
Please read the [contributing guide](./CONTRIBUTING.md) to get started.

## Citing

If you use JaxSim in your work, please cite the following paper:

```bibtex
@article{ferretti_contact_aware_2026,
  author       = {Filippo Luca Ferretti and Diego Ferigo and Alessandro Croci and Carlotta Sartore and Omar G. Younis and Silvio Traversaro and Daniele Pucci},
  title        = {Contact-Aware Morphology Optimization via Physically Consistent Differentiable Simulation},
  journal      = {IEEE Robotics and Automation Letters},
  year         = {2026},
  doi          = {10.1109/LRA.2026.3678125}
}
```

## People

| Authors | Maintainers |
|:------:|:-----------:|
| [<img src="https://avatars.githubusercontent.com/u/469199?v=4" width="40">][df] [<img src="https://avatars.githubusercontent.com/u/102977828?v=4" width="40">][ff] | [<img src="https://avatars.githubusercontent.com/u/102977828?v=4" width="40">][ff] [<img src="https://avatars.githubusercontent.com/u/57228872?v=4" width="40">][ac] |

[df]: https://github.com/diegoferigo
[ff]: https://github.com/flferretti
[ac]: https://github.com/xela-95

## License

[BSD3](https://choosealicense.com/licenses/bsd-3-clause/)
