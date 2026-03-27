# JaxSim Examples

This folder contains Jupyter notebooks that demonstrate the practical usage of JaxSim.

## Featured examples

| Notebook | Google Colab | Description |
| :--- | :---: | :--- |
| [`jaxsim_as_multibody_dynamics_library`](./jaxsim_as_multibody_dynamics_library.ipynb) | [![Open In Colab][colab_badge]][ipynb_jaxsim_as_multibody_dynamics] | An example demonstrating how to use JaxSim as a multibody dynamics library. |
| [`jaxsim_as_physics_engine.ipynb`](./jaxsim_as_physics_engine.ipynb) | [![Open In Colab][colab_badge]][ipynb_jaxsim_as_physics_engine] | An example demonstrating how to simulate vectorized models in parallel. |
| [`jaxsim_as_physics_engine_advanced.ipynb`](./jaxsim_as_physics_engine_advanced.ipynb) | [![Open In Colab][colab_badge]][jaxsim_as_physics_engine_advanced] | An example showcasing advanced JaxSim usage, such as customizing the integrator, contact model, and more. |
| [`jaxsim_for_robot_controllers.ipynb`](./jaxsim_for_robot_controllers.ipynb) | [![Open In Colab][colab_badge]][ipynb_jaxsim_closed_loop] | A basic example showing how to simulate a PD controller with gravity compensation for a 2-DOF cart-pole. |

[colab_badge]: https://colab.research.google.com/assets/colab-badge.svg
[ipynb_jaxsim_closed_loop]: https://colab.research.google.com/github/gbionics/jaxsim/blob/main/examples/jaxsim_for_robot_controllers.ipynb
[ipynb_jaxsim_as_physics_engine]: https://colab.research.google.com/github/gbionics/jaxsim/blob/main/examples/jaxsim_as_physics_engine.ipynb
[jaxsim_as_physics_engine_advanced]: https://colab.research.google.com/github/gbionics/jaxsim/blob/main/examples/jaxsim_as_physics_engine_advanced.ipynb
[ipynb_jaxsim_as_multibody_dynamics]: https://colab.research.google.com/github/gbionics/jaxsim/blob/main/examples/jaxsim_as_multibody_dynamics_library.ipynb

## How to run the examples

You can run the JaxSim examples with hardware acceleration in two ways.

### Option 1: Google Colab (recommended)

The easiest way is to use the provided Google Colab links to run the notebooks in a hosted environment
with no setup required.

### Option 2: Local execution with `pixi`

To run the examples locally, first install `pixi` following the [official documentation][pixi_installation]:

[pixi_installation]: https://pixi.sh/#installation

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then, from the repository's root directory, execute the example notebooks using:

```bash
pixi run examples
```

This command will automatically handle all necessary dependencies and run the examples in a self-contained environment.
