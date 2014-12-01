__author__ = 'Alexander Egorov'

import sys
import argparse
import subprocess

TOOLS = frozenset([
    'md4',
    'md5',
    'sha1',
    'sha256',
    'sha384',
    'sha512',
    'crc32',
    'whirlpool'
])

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
    parser = argparse.ArgumentParser(description="Brute force crack hash tool. Copyright (C) 2009-2012 Alexander Egorov.")
    parser.add_argument('-t', '--algorithm', dest='algorithm', required=True,
                        help='Hash algorithm. Valid values crc32, md5, md4, sha1, sha256, sha384, sha512, whirlpool')
    parser.add_argument('-s', '--hash', dest='hash', required=True, help='Hash string to crack')
    parser.add_argument('-n', '--min', dest='min', help='set minimum length of the string to restore. 1 by default')
    parser.add_argument('-x', '--max', dest='max', help='set maximum length of the string to restore. 10 by default')
    parser.add_argument('-a', '--dict', dest='dict',
                        help='initial string\'s dictionary by default all digits, upper and lower case latin symbols')

    args = parser.parse_args()

    if not len(TOOLS & {args.algorithm}):
        parser.error('Incorrect algorithm')
    cmd = ['{0}.exe'.format(args.algorithm), "-c", '-m', args.hash]
    if args.min:
        cmd.append('-n')
        cmd.append(args.min)
    if args.max:
        cmd.append('-x')
        cmd.append(args.max)
    if args.dict:
        cmd.append('-a')
        cmd.append(args.dict)

    return RunShellCommand(cmd)

if __name__ == '__main__':
    sys.exit(main())