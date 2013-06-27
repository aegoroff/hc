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
    'whirlpool'
)

_ALGORITHMS_HQ = (
    'md2',
    'sha224',
    'tiger',
    'tiger2',
    'ripemd128',
    'ripemd160',
    'ripemd256',
    'ripemd320'
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


def test_hq(algorithm, path):
    print_head(algorithm)

    stting_hash = "for string '1234' do {0};".format(algorithm)
    exe = 'hq.exe'
    if path:
        exe = os.path.join(path, exe)
    f = run([exe, "-c", stting_hash])
    with f.stdout:
        s_to_crack = f.stdout.readline().strip()

    cases = (
        "for string s from hash '{0}' do crack {1};",
        "for string s from hash '{0}' let s.dict='0-9' do crack {1};",
        "for string s from hash '{0}' let s.dict='0-9', s.min = 4 do crack {1};",
        "for string s from hash '{0}' let s.dict='0-9', s.max = 4 do crack {1};",
        "for string s from hash '{0}' let s.dict='0-9', s.max = 4 do crack {1};",
    )

    f = lambda c: test_method(exe, ('-c', c.format(s_to_crack, algorithm)))
    map(f, cases)


def test(algorithm, path):
    print_head(algorithm)
    exe = '{0}.exe'.format(algorithm)
    if path:
        exe = os.path.join(path, exe)
    f = run([exe, "-s", "1234"])
    with f.stdout:
        s_to_crack = f.stdout.readline().strip()

    cases = [
        ('-c', '-m', s_to_crack),
        ('-c', '-m', s_to_crack, '-a', '0-9'),
        ('-c', '-m', s_to_crack, '-a', '0-9', '-x' '6'),
        ('-c', '-m', s_to_crack, '-a', '0-9', '-x' '6', '-n', '3'),
        ('-d', '.'),
        ('-d', '.', '-i', "*.exe"),
        ('-d', '.', '-e', "*.exe"),
        ('-d', '.', '-h', s_to_crack),
        ('-d', '.', '-h', s_to_crack, '-r')
    ]

    map(lambda case: test_method(exe, case), cases)


def main():
    parser = argparse.ArgumentParser(description="Hash calculators testing tool. Copyright (C) 2013 Alexander Egorov.")
    parser.add_argument('-p', '--path', dest='path', help='Path to executables folder', default=None)

    args = parser.parse_args()

    map(lambda a: test(a, args.path), _ALGORITHMS)
    map(lambda a: test_hq(a, args.path), _ALGORITHMS)
    map(lambda a: test_hq(a, args.path), _ALGORITHMS_HQ)

    exe = 'hq.exe'
    if args.path:
        exe = os.path.join(args.path, exe)
    d = os.path.realpath(__file__)
    dd = os.path.dirname(d)
    queries = os.path.join(dd, '..', 'pgo.hlq')
    test_method(exe, queries,)

    return 0

if __name__ == '__main__':
    sys.exit(main())
