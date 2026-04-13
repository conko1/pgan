import sys
import os

from cli import cli

# pridá project root do sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    cli()