import argparse
import os
import subprocess
import sys

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
)

def run(params):
    return subprocess.Popen(params, stdout=subprocess.PIPE)


def test_method(exe, params):
    cmd = [exe]
    map(cmd.append, params)
    output = run([c for c in cmd])
    for line in output.stdout:
        print line


def print_head(algorithm):
    separator = "-" * 80
    print
    print separator
    print algorithm
    print


def test(algorithm, path):
    print_head(algorithm)
    exe = 'hq.exe'
    if path:
        exe = os.path.join(path, exe)
    f = run([exe, algorithm, "-s", "1234"])
    with f.stdout:
        s_to_crack = f.stdout.readline().strip()

    cases = [
        (algorithm, '-c', '-m', s_to_crack),
        (algorithm, '-c', '-m', s_to_crack, '-a', '0-9', '-x' '6', '-n', '3'),
        (algorithm, '-d', '.'),
        (algorithm, '-d', '.', '-i', "*.exe"),
        (algorithm, '-d', '.', '-e', "*.exe"),
        (algorithm, '-d', '.', '-H', s_to_crack, '-r'),
        ('-C', "for string s from hash '{0}' let s.dict='0-9', s.max = 4 do crack {1};".format(s_to_crack, algorithm)),
        ('-C', "let filemask = '.*exe$'; for file f from dir '.'  where f.{1} == '{0}' and f.size > 20 and f.name ~ filemask do find;".format(s_to_crack, algorithm)),
    ]

    map(lambda case: test_method(exe, case), cases)


def main():
    parser = argparse.ArgumentParser(description="Hash calculators testing tool. Copyright (C) 2013 Alexander Egorov.")
    parser.add_argument('-p', '--path', dest='path', help='Path to executables folder', default=None)

    args = parser.parse_args()

    map(lambda a: test(a, args.path), _ALGORITHMS)

    exe = 'hq.exe'
    if args.path:
        exe = os.path.join(args.path, exe)
    d = os.path.realpath(__file__)
    dd = os.path.dirname(d)
    queries = os.path.join(dd, '..', 'pgo.hlq')
    count = 0
    test_method(exe, ('-F', queries))
    while count < 100:
        test_method(exe, ('-S', '-F', queries))
        count += 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
