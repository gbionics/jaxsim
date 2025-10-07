from . import logging
from ._version import __version__


# Follow upstream development in https://github.com/google/jax/pull/13304
def _jnp_options() -> None:
    import os

    import jax

    # Check if running on TPU.
    is_tpu = jax.devices()[0].platform == "tpu"

    # Check if running on Metal.
    is_metal = jax.devices()[0].platform == "METAL"

    # Enable by default 64-bit precision to get accurate physics.
    # Users can enforce 32-bit precision by setting the following variable to 0.
    use_x64 = os.environ.get("JAX_ENABLE_X64", "0") != "0"

    # Notify the user if unsupported 64-bit precision was enforced on TPU.
    if (is_tpu or is_metal) and use_x64:
        msg = f"64-bit precision is not allowed on {jax.devices()[0].platform.upper}. Enforcing 32bit precision."
        logging.warning(msg)
        use_x64 = False

        if is_metal:
            logging.warning(
                "JAX Metal backend is experimental. Some functionalities may not be available."
            )

    # Enable 64-bit precision in JAX.
    if use_x64:
        logging.info("Enabling JAX to use 64-bit precision")
        jax.config.update("jax_enable_x64", True)


def _np_options() -> None:
    import numpy as np

    np.set_printoptions(precision=5, suppress=True, linewidth=150, threshold=10_000)


def _is_editable() -> bool:

    import importlib.util
    import pathlib
    import site

    # Get the ModuleSpec of jaxsim.
    jaxsim_spec = importlib.util.find_spec(name="jaxsim")

    # This can be None. If it's None, assume non-editable installation.
    if jaxsim_spec.origin is None:
        return False

    # Get the folder containing the jaxsim package.
    jaxsim_package_dir = str(pathlib.Path(jaxsim_spec.origin).parent.parent)

    # The installation is editable if the package dir is not in any {site|dist}-packages.
    return jaxsim_package_dir not in site.getsitepackages()


def _get_default_logging_level() -> logging.LoggingLevel:
    """
    Get the default logging level.

    Returns:
        The logging level to set.
    """

    import os
    import sys

    # Allow to override the default logging level with an environment variable.
    if overriden_logging_level := os.environ.get("JAXSIM_LOGGING_LEVEL"):
        try:
            return logging.LoggingLevel[overriden_logging_level.upper()]

        except KeyError as exc:
            msg = "Invalid logging level defined in JAXSIM_LOGGING_LEVEL"
            raise RuntimeError(msg) from exc

    # If running under a debugger, set the logging level to DEBUG.
    if getattr(sys, "gettrace", lambda: None)():
        return logging.LoggingLevel.DEBUG

    # If not running under a debugger, set the logging level to INFO or WARNING.
    # INFO for editable installations, WARNING for non-editable installations.
    # This is to avoid too verbose logging in non-editable installations.
    return (
        logging.LoggingLevel.INFO
        if _is_editable()  # noqa: F821
        else logging.LoggingLevel.WARNING
    )


# Configure the logger with the default logging level.
logging.configure(level=_get_default_logging_level())


# Configure JAX.
_jnp_options()

# Initialize the numpy print options.
_np_options()

del _jnp_options
del _np_options
del _get_default_logging_level
del _is_editable

from . import terrain  # isort:skip
from . import api, logging, math, rbda
from .api.common import VelRepr
