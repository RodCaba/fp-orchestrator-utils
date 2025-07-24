from dotenv import load_dotenv
import argparse
import sys

from .proto_commands import setup_proto_commands

# Load environment variables
load_dotenv()

def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(
        description="FP Proto Manager CLI Tool - Manage Protocol Buffers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')
    setup_proto_commands(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Call the appropriate command function
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
