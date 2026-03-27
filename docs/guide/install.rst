Installation
============

.. _installation:

Prerequisites
-------------

JAXsim requires Python 3.11 or later.

Basic Installation
------------------

You can install the project with using `conda`_:

.. code-block:: bash

   conda install jaxsim -c conda-forge

Alternatively, you can use `pypa/pip`_, preferably in a `virtual environment`_:

.. code-block:: bash

   pip install jaxsim

Have a look to `pyproject.toml`_ for a complete list of optional dependencies.
You can install all by using ``pip install "jaxsim[all]"``.
.. note::

    If you need GPU support, please follow the official `installation instruction`_ of JAX.

.. _conda: https://anaconda.org/
.. _pyproject.toml: https://github.com/gbionics/jaxsim/blob/main/pyproject.toml
.. _pypa/pip: https://github.com/pypa/pip/
.. _virtual environment: https://docs.python.org/3.8/tutorial/venv.html
.. _installation instruction: https://github.com/google/jax/#installation
