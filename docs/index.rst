JAXsim
#######

A scalable physics engine and multibody dynamics library implemented with JAX. With JIT batteries 🔋

.. note::
    This simulator currently focuses on locomotion applications. Only contacts with ground are supported.

Features
--------

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Performance
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Physics engine in reduced coordinates implemented with JAX_.
            Compatibility with JIT compilation for increased performance and transparent support to execute logic on CPUs, GPUs, and TPUs.
            Parallel multi-body simulations on hardware accelerators for significantly increased throughput

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Model Parsing
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Support for SDF models (and, upon conversion, URDF models). Revolute, prismatic, and fixed joints supported.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Automatic Differentiation
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Support for automatic differentiation of rigid body dynamics algorithms (RBDAs) for model-based robotics research.
            Soft contacts model supporting full friction cone and sticking / slipping transition.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Complex Dynamics
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            JAXsim provides a variety of integrators for the simulation of multibody dynamics, including RK4, Heun, Euler, and more.
            Support of `multiple velocities representations <https://research.tue.nl/en/publications/multibody-dynamics-notation-version-2>`_.


----

.. toctree::
  :hidden:

  guide/install
  guide/configuration

  examples

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: JAXsim API

  modules/api
  modules/math
  modules/mujoco
  modules/parsers
  modules/rbda
  modules/typing
  modules/utils

Examples
--------

Explore and learn how to use the library through practical demonstrations available in the `examples <https://github.com/gbionics/jaxsim/tree/main/examples>`__ folder.

Credits
-------

The physics module of JAXsim is based on the theory of the `Rigid Body Dynamics Algorithms <https://link.springer.com/book/10.1007/978-1-4899-7560-7>`_ book by Roy Featherstone.
We structured part of our logic following its accompanying `code <http://royfeatherstone.org/spatial/index.html#spatial-software>`_.
The physics engine is developed entirely in Python using JAX_.

The inspiration for developing JAXsim originally stemmed from early versions of Brax_.
Here below we summarize the differences between the projects:

- JAXsim simulates multibody dynamics in reduced coordinates, while :code:`brax v1` uses maximal coordinates.
- The new v2 APIs of brax (and the new MJX_) were then implemented in reduced coordinates, following an approach comparable to JAXsim, with major differences in contact handling.
- The rigid-body algorithms used in JAXsim allow to efficiently compute quantities based on the Euler-Poincarè
  formulation of the equations of motion, necessary for model-based robotics research.
- JAXsim supports SDF (and, indirectly, URDF) models, assuming the model is described with the
  recent `Pose Frame Semantics <http://sdformat.org/tutorials?tut=pose_frame_semantics>`_.
- Contrarily to brax, JAXsim only supports collision detection between bodies and a compliant ground surface.
- The RBDAs of JAXsim support automatic differentiation, but this functionality has not been thoroughly tested.


People
------

Authors
'''''''

`Diego Ferigo <https://github.com/diegoferigo>`_
`Filippo Luca Ferretti <https://github.com/flferretti>`_

Maintainers
'''''''''''

`Filippo Luca Ferretti <https://github.com/flferretti>`_
`Alessandro Croci <https://github.com/xela-95>`_

License
-------

`BSD3 <https://choosealicense.com/licenses/bsd-3-clause/>`_

.. _Brax: https://github.com/google/brax
.. _MJX: https://mujoco.readthedocs.io/en/3.0.0/mjx.html
.. _JAX: https://github.com/google/jax
