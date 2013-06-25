import subprocess
import sys

__author__ = 'egr'


def run(params):
    return subprocess.Popen(params, stdout=subprocess.PIPE)


def test_method(exe, *params):
    cmd = [exe]
    map(cmd.append, params)
    output = run([c for c in cmd])
    for line in output.stdout:
        print line,


def test(algorithm):
    print algorithm
    exe = '{0}.exe'.format(algorithm)
    f = run([exe, "-s", "1234"])
    with f.stdout:
        s_to_crack = f.stdout.readline().strip()
    test_method(exe, '-c', '-m', s_to_crack)
    test_method(exe, '-c', '-m', s_to_crack, '-a', '0-9')
    test_method(exe, '-c', '-m', s_to_crack, '-a', '0-9', '-x' '6')
    test_method(exe, '-d', '.')
    test_method(exe, '-d', '.', '-i', "*.exe")
    test_method(exe, '-d', '.', '-e', "*.exe")
    test_method(exe, '-d', '.', '-h', s_to_crack)
    test_method(exe, '-d', '.', '-h', s_to_crack, '-r')


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
