#!/usr/bin/python3
# PYTHON_ARGCOMPLETE_OK

# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

from sys import argv
from shutil import copyfile
from argparse import ArgumentParser
from camino.runner import runner, batch_runner
from camino.utils import setup_logger, logging
from camino.settings import Settings
import argcomplete


def main(args):
    """Process the data."""
    if Settings.WITH_DEBUG:
        setup_logger(logging.DEBUG)
    else:
        setup_logger(logging.INFO)

    parser = ArgumentParser(
        description="Benders solver"
    )
    subparser = parser.add_subparsers(
        title="Commands",
        dest="command",
        description="",
        help="""for argument information use:
                                      <command-name> -h""",
    )
    parser_run = subparser.add_parser("run", help="Run a pre-defined problem")
    parser_run.add_argument("solver")
    parser_run.add_argument("problem_name")
    parser_run.add_argument("--save", default=None, type=str)
    parser_run.add_argument("--args", default=None, nargs="*", type=str)
    parser_copy = subparser.add_parser("copy", help="Copy NL files")
    parser_copy.add_argument("target")
    parser_copy.add_argument("solution")
    parser_copy.add_argument("nlfile")
    parser_batch = subparser.add_parser(
        "batch", help="Run a batch of NL files")
    parser_batch.add_argument("algorithm")
    parser_batch.add_argument("target")
    parser_batch.add_argument("nlfiles", type=str, nargs="+")

    argcomplete.autocomplete(parser)
    parsed = parser.parse_args(args)

    if parsed.command == "copy":
        copyfile(parsed.nlfile, parsed.target)
        print(f"File copied to {parsed.target}")
    elif parsed.command == "batch":
        batch_runner(parsed.algorithm, parsed.target, parsed.nlfiles)
    elif parsed.command == "run":
        runner(parsed.solver, parsed.problem_name, parsed.save, parsed.args)
    else:
        parser.print_help()

    exit(0)


if __name__ == "__main__":
    main(argv[1:])
