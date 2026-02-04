import argparse
import pathlib
import sys
import time

import numpy as np

from . import ModelToMjcf, MujocoModelHelper, MujocoVisualizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="jaxsim.mujoco",
        description="Process URDF and SDF files for Mujoco usage.",
    )

    parser.add_argument(
        "-d",
        "--description",
        required=True,
        metavar="INPUT_FILE",
        type=pathlib.Path,
        help="Path to the URDF or SDF file.",
    )

    parser.add_argument(
        "-m",
        "--model-name",
        metavar="NAME",
        type=str,
        default=None,
        help="The target model of a SDF description if multiple models exists.",
    )

    parser.add_argument(
        "-e",
        "--export",
        metavar="MJCF_FILE",
        type=pathlib.Path,
        default=None,
        help="Path to the exported MJCF file.",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Override the output MJCF file if it already exists (default: %(default)s).",
    )

    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        default=False,
        help="Print in the stdout the exported MJCF string (default: %(default)s).",
    )

    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize the description in the Mujoco viewer (default: %(default)s).",
    )

    parser.add_argument(
        "-b",
        "--base-position",
        metavar=("x", "y", "z"),
        nargs=3,
        type=float,
        default=None,
        help="Override the base position (supports only floating-base models).",
    )

    parser.add_argument(
        "-q",
        "--base-quaternion",
        metavar=("w", "x", "y", "z"),
        nargs=4,
        type=float,
        default=None,
        help="Override the base quaternion (supports only floating-base models).",
    )

    args = parser.parse_args()

    # ==================
    # Validate arguments
    # ==================

    # Expand the path of the URDF/SDF file if not absolute.
    if args.description is not None:
        args.description = (
            (
                args.description
                if args.description.is_absolute()
                else pathlib.Path.cwd() / args.description
            )
            .expanduser()
            .absolute()
        )

        if not pathlib.Path(args.description).is_file():
            msg = f"The URDF/SDF file '{args.description}' does not exist."
            parser.error(msg)
            sys.exit(1)

    # Expand the path of the output MJCF file if not absolute.
    if args.export is not None:
        args.export = (
            (
                args.export
                if args.export.is_absolute()
                else pathlib.Path.cwd() / args.export
            )
            .expanduser()
            .absolute()
        )

        if pathlib.Path(args.export).is_file() and not args.force:
            msg = "The output file '{}' already exists, use '--force' to override."
            parser.error(msg.format(args.export))
            sys.exit(1)

    # ================================================
    # Load the URDF/SDF file and produce a MJCF string
    # ================================================

    mjcf_string, assets = ModelToMjcf.convert(args.description)

    if args.print:
        print(mjcf_string, flush=True)

    # ========================================
    # Write the MJCF string to the output file
    # ========================================

    if args.export is not None:
        with open(args.export, "w+", encoding="utf-8") as file:
            file.write(mjcf_string)

    # =======================================
    # Visualize the MJCF in the Mujoco viewer
    # =======================================

    if args.visualize:

        mj_model_helper = MujocoModelHelper.build_from_xml(
            mjcf_description=mjcf_string, assets=assets
        )

        viz = MujocoVisualizer(model=mj_model_helper.model, data=mj_model_helper.data)

        with viz.open() as viewer:

            with viewer.lock():
                if args.base_position is not None:
                    mj_model_helper.set_base_position(
                        position=np.array(args.base_position)
                    )

                if args.base_quaternion is not None:
                    mj_model_helper.set_base_orientation(
                        orientation=np.array(args.base_quaternion)
                    )

            viz.sync(viewer=viewer)

            while viewer.is_running():
                time.sleep(0.500)

    # =============================
    # Exit the program with success
    # =============================

    sys.exit(0)
