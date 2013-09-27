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
    'edonr256',
    'edonr512',
)

LET = 'let filemask = ".*exe$";\n'

QUERIES = (
    "# Comment\n",
    "# Comment\nfor file f from dir 'c:' where f.size == 0 do find;",
    "# Comment\nfor file f from dir 'c:' where f.size == 0 do find;",
    "for file f from dir 'c:' where f.name == 'test' do find;",
    "for string '123' do md5;",
    "for string s from hash '202CB962AC59075B964B07152D234B70' let s.max = 5 do crack md5;",
    "for string s from hash '202CB962AC59075B964B07152D234B70' let s.min = 3 do crack md5;",
    "for string s from hash '202CB962AC59075B964B07152D234B70' let s.dict = '0-9' do crack md5;",
    "for string s from hash '202CB962AC59075B964B07152D234B70' let s.max = 5, s.dict = '0-9', s.min = 3 do crack md5;",
    "for string s from hash 'D41D8CD98F00B204E9800998ECF8427E' let s.min = 4 do crack md5;",
    "for file f from dir 'c:' where f.size == 0 do find;",
    "for file f from dir 'c:' where f.size == 0 and f.name ~ filemask do find;",
    "for file f from dir 'c:' where f.size == 0 or f.name ~ '*.exe' do find;",
    "for file f from dir 'c:' where f.size == 0 and (f.name !~ '*.exe' or f.path ~ 'c:\\temp\\*') do find;",
    "for file f from '1' do md5;",
    "for file f from '1' let f.limit = 10 do md5;",
    "for file f from '1' let f.md5 = 'D41D8CD98F00B204E9800998ECF8427E' do validate;",
    "for file f from parameter where f.md5 == 'D41D8CD98F00B204E9800998ECF8427E' do validate;",
    "for file f from dir 'c:' where f.md5 == 'D41D8CD98F00B204E9800998ECF8427E' do find;",
    "for file f from dir '.' where f.md5 == 'D41D8CD98F00B204E9800998ECF84271' and f.limit == 100 and f.offset == 10 do find;",
    "for file f from dir '.' where f.size < 0 and f.md5 == 'D41D8CD98F00B204E9800998ECF84271' do find;",
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
    exe = 'hc.exe'
    if path:
        exe = os.path.join(path, exe)
    f = run([exe, algorithm, "-s", "12345"])
    with f.stdout:
        s_to_crack = f.stdout.readline().strip()

    cases = [
        (algorithm, '-c', '-m', s_to_crack, '-n', '5', '-a', '0-9a-z', '--noprobe'),
        (algorithm, '-d', '.', '-i', "*.exe", '--noprobe'),
        ('-C', "let filemask = '.*exe$'; for file f from dir '.'  where f.{1} == '{0}' and f.size > 20 and f.name ~ filemask do find;".format(s_to_crack, algorithm)),
    ]

    map(lambda case: test_method(exe, case), cases)


def main():
    parser = argparse.ArgumentParser(description="Hash calculators testing tool. Copyright (C) 2013 Alexander Egorov.")
    parser.add_argument('-p', '--path', dest='path', help='Path to executables folder', default=None)

    args = parser.parse_args()

    map(lambda a: test(a, args.path), _ALGORITHMS)

    exe = 'hc.exe'
    if args.path:
        exe = os.path.join(args.path, exe)

    temp = 't.hlq'

    q = "\n".join(QUERIES)
    with open(temp, 'w') as f:
        f.write(LET)
        count = 0
        while count < 1000:
            f.write(q)
            count += 1


    count = 0
    while count < 20:
        test_method(exe, ('-S', '--noprobe', '-F', temp))
        count += 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
