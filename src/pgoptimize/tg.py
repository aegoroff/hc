import argparse
import os
import subprocess
import sys

__author__ = 'egr'

_ALGORITHMS = (
    'haval-128-5',
	'haval-160-3',
    'haval-160-4',
    'haval-160-5',
	'haval-192-3',
    'haval-192-4',
    'haval-192-5',
	'haval-224-3',
    'haval-224-4',
    'haval-224-5',
	'haval-256-3',
    'haval-256-4',
    'haval-256-5',
)

t = """
    public class %s : Hash
    {
        public override string HashString
        {
            get { return "%s"; }
        }

        public override string EmptyStringHash
        {
            get { return "%s"; }
        }

        public override string StartPartStringHash
        {
            get { return "%s"; }
        }

        public override string MiddlePartStringHash
        {
            get { return "%s"; }
        }

        public override string TrailPartStringHash
        {
            get { return "%s"; }
        }

        public override string Algorithm
        {
            get { return "%s"; }
        }
    }
"""

def run(params):
    return subprocess.Popen(params, stdout=subprocess.PIPE)


def test(algorithm, path):
    exe = 'hq.exe'
    if path:
        exe = os.path.join(path, exe)
    f123 = run([exe, algorithm, "-s", "123"])
    with f123.stdout:
        s123 = f123.stdout.readline().strip()

    fe = run([exe, algorithm, "-s", '""'])
    with fe.stdout:
        se = fe.stdout.readline().strip()

    f12 = run([exe, algorithm, "-s", '12'])
    with f12.stdout:
        s12 = f12.stdout.readline().strip()

    f2 = run([exe, algorithm, "-s", '2'])
    with f2.stdout:
        s2 = f2.stdout.readline().strip()

    f23 = run([exe, algorithm, "-s", '23'])
    with f23.stdout:
        s23 = f23.stdout.readline().strip()

    c = t % (algorithm, s123, se, s12, s2, s23, algorithm)
    print c


def main():
    parser = argparse.ArgumentParser(description="Hash calculators testing tool. Copyright (C) 2013 Alexander Egorov.")
    parser.add_argument('-p', '--path', dest='path', help='Path to executables folder', default=None)

    args = parser.parse_args()

    map(lambda a: test(a, args.path), _ALGORITHMS)

    return 0

if __name__ == '__main__':
    sys.exit(main())
