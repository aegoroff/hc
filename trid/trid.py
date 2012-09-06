#!/usr/bin/python2.7
# coding=windows-1251
import logging
import os
import shutil
import sys
import subprocess
import argparse
from lxml import etree
import binascii

__author__ = 'Alexander Egorov'

# Use a shell for subcommands on Windows to get a PATH search.
useShell = sys.platform.startswith("win")

result_dir = 'hql'

def RunShellCommand(command, universalNewlines=True):
    """Executes a command and returns the output from stdout and the return code.

    Args:
      command: Command to execute.
      universalNewlines: Use universal_newlines flag (default: True).

    Returns:
      Tuple (output, return code)
    """
    p = subprocess.Popen(command, shell=useShell, stdout=subprocess.PIPE, universal_newlines=universalNewlines)
    p.wait()
    return p


def CreateQueryFromTridXml(path):
    logging.info("processing %s", path)
    title = ''
    signature_ext = ''
    descr = ''
    bytes = ''
    offset = 0
    with open(path, 'r') as f:
        list = []
        for event, element in etree.iterparse(f, events=("start", "end")):
            if event == 'start':
                if element.tag == 'FileType':
                    title = element.text
                if element.tag == 'Ext':
                    signature_ext = element.text
                if element.tag == 'Rem':
                    descr = element.text
                if element.tag == 'Bytes':
                    bytes = element.text
                if element.tag == 'Pos':
                    offset = int(element.text)
            if event == 'end':
                if element.tag == 'Pattern':
                    binary = binascii.unhexlify(bytes)
                    tmp_file = "test.bin"
                    try:
                        with open(tmp_file, "wb") as tmp:
                            tmp.write(binary)
                        p = RunShellCommand('md5 -f %s' % tmp_file)
                        s = p.stdout.read()
                        pieces = s.split('|')
                        hash = pieces[2].strip()
                        list.append(
                            "(f.offset == %i and f.limit == %i and f.md5 == '%s')" % (offset, len(bytes) / 2, hash))
                    finally:
                        os.remove(tmp_file)
        where = ' and\n'.join(list)
        if descr is None:
            descr = ''
        q = "# %s (%s)\n# %s\n\nfor file f from dir '.' where\n%s\ndo find;" % (title, signature_ext, descr, where)

        dir, name = os.path.split(path)
        root, ext = os.path.splitext(name)
        file_path = os.path.join(result_dir, "%s.hql" % root)
        with open(file_path, "w") as qf:
            qf.write(q)


def main():
    logging.basicConfig(format=("%(asctime).19s %(levelname)s %(message)s "))

    parser = argparse.ArgumentParser(description="TRiD signatures converting tool. Copyright (C) 2012 Alexander Egorov.")
    parser.add_argument('-p', '--path', dest='path', help='Path to TRiD signature files', required=True)
    parser.add_argument('-v', '--verbose', dest='verbose', help='Verbose output', action='store_true', default=False)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    files = os.listdir(args.path)

    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir, True)
    os.mkdir(result_dir)
    for filename in files:
        if filename.rfind(".trid.xml") == -1:
            continue
        CreateQueryFromTridXml(os.path.join(args.path, filename))

if __name__ == '__main__':
    sys.exit(main())