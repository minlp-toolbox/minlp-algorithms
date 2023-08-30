#!/usr/bin/python3
# PYTHON_ARGCOMPLETE_OK

from sys import argv
from shutil import copyfile
from argparse import ArgumentParser
import argcomplete


def main(args):
    """Process the data."""
    global log

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
    parser_copy = subparser.add_parser("copy", help="Copy NL files")
    parser_copy.add_argument("target")
    parser_copy.add_argument("solution")
    parser_copy.add_argument("nlfile")

    argcomplete.autocomplet(parser)
    parsed = parser.parse_args(args)

    if parsed.command == "copy":
        copyfile(parsed.nlfile, parsed.target)
        print(f"File copied to {parsed.target}")
    else:
        raise NotImplementedError()

    exit(0)


if __name__ == "__main__":
    with open("/tmp/benderslog.txt", "w") as f:
        f.write(" ".join(argv))
