#!/usr/bin/python2.7
# coding=windows-1251

__author__ = 'Alexander Egorov'

import sys
import subprocess
import argparse


# Use a shell for subcommands on Windows to get a PATH search.
useShell = sys.platform.startswith("win")

def RunShellCommand(command, universalNewlines=True):
    """Executes a command and returns the output from stdout and the return code.

    Args:
      command: Command to execute.
      universalNewlines: Use universal_newlines flag (default: True).

    Returns:
      Tuple (output, return code)
    """
    p = subprocess.Popen(command, shell=useShell, universal_newlines=universalNewlines)
    p.wait()
    return p.returncode

def main():
    parser = argparse.ArgumentParser(description="TRiD signatures converting tool. Copyright (C) 2012 Alexander Egorov.")


if __name__ == '__main__':
    sys.exit(main())