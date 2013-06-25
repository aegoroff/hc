import subprocess
import sys

__author__ = 'egr'


def run(params):
    return subprocess.Popen(params, stdout=subprocess.PIPE)


def test_method(exe, params):
    cmd = [exe]
    map(cmd.append, params)
    output = run([c for c in cmd])
    for line in output.stdout:
        print line,


def test(algorithm):
    separator = "-" * 80
    print
    print separator
    print algorithm
    print
    exe = '{0}.exe'.format(algorithm)
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
    algorithms = [
        'crc32',
        'md4',
        'md5',
        'sha1',
        'sha256',
        'sha384',
        'sha512',
        'whirlpool'
    ]

    map(test, algorithms)

    return 0

if __name__ == '__main__':
    sys.exit(main())
