import argparse
import os
import subprocess
import sys
from string import maketrans

__author__ = 'egr'

_ALGORITHMS = (
    'crc32',
    'md4',
    'md5',
    'sha1',
    'sha256',
    'sha384',
    'sha512',
    'whirlpool',
    'md2',
    'sha224',
    'tiger',
    'tiger2',
    'ripemd128',
    'ripemd160',
    'ripemd256',
    'ripemd320',
    'gost',
    'snefru256',
    'snefru128',
    'tth',
    'haval-128-3',
    'haval-128-4',
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
    'edonr256',
    'edonr512',
	'ntlm',
    'sha-3-224',
    'sha-3-256',
    'sha-3-384',
    'sha-3-512',
    'sha-3k-224',
    'sha-3k-256',
    'sha-3k-384',
    'sha-3k-512',
)

t = """
    public class %s : Hash
    {
        public override string HashString => "%s";

        public override string EmptyStringHash => "%s";

        public override string StartPartStringHash => "%s";

        public override string MiddlePartStringHash => "%s";

        public override string TrailPartStringHash => "%s";

        public override string Algorithm => "%s";
    }
"""

def run(params):
    return subprocess.Popen(params, stdout=subprocess.PIPE)


def test(algorithm, path):
    exe = 'hc.exe'
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

    intab = "-"
    outtab = "_"
    trantab = maketrans(intab, outtab)
    className = algorithm.title().translate(trantab)
    c = t % (className, s123, se, s12, s2, s23, algorithm)
    print c


def main():
    parser = argparse.ArgumentParser(description="Hash calculators testing tool. Copyright (C) 2013 Alexander Egorov.")
    parser.add_argument('-p', '--path', dest='path', help='Path to executables folder', default=None)

    args = parser.parse_args()

    map(lambda a: test(a, args.path), _ALGORITHMS)

    return 0

if __name__ == '__main__':
    sys.exit(main())
