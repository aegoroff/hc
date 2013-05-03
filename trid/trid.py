#!/usr/bin/python2.7
# coding=windows-1251
import hashlib
import logging
import os
import shutil
import sys
import argparse
from lxml import etree
import binascii

__author__ = 'Alexander Egorov'

_RESULT_DIR = 'hql'

def CreateQueryFromTridXml(path):
    logging.info("processing %s", path)
    title = ''
    signature_ext = ''
    descr = ''
    signatureBytes = ''
    offset = 0
    with open(path, 'r') as f:
        whereList = []
        for event, element in etree.iterparse(f, events=("start", "end")):
            if event == 'start':
                if element.tag == 'FileType':
                    title = element.text
                if element.tag == 'Ext':
                    signature_ext = element.text
                if element.tag == 'Rem':
                    descr = element.text
                if element.tag == 'Bytes':
                    signatureBytes = element.text
                if element.tag == 'Pos':
                    offset = int(element.text)
            if event == 'end':
                if element.tag == 'FrontBlock':
                    break
                if element.tag == 'Pattern':
                    binary = binascii.unhexlify(signatureBytes)
                    m = hashlib.md5(binary)
                    d = m.hexdigest()
                    whereList.append(
                        "(f.offset == {0:d} and f.limit == {1:d} and f.md5 == '{2}')".format(offset, len(signatureBytes) / 2, d))

        where = ' and\n'.join(whereList)
        if descr is None:
            descr = ''
        q = "# {0} ({1})\n# {2}\n\nfor file f from parameter where\n{3}\ndo validate;".format(title, signature_ext, descr, where)

        directory, name = os.path.split(path)
        root, ext = os.path.splitext(name)
        file_path = os.path.join(_RESULT_DIR, "{0}.hql".format(root))
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

    if os.path.isdir(_RESULT_DIR):
        shutil.rmtree(_RESULT_DIR, True)
    os.mkdir(_RESULT_DIR)
    for filename in files:
        if filename.rfind(".trid.xml") == -1:
            continue
        CreateQueryFromTridXml(os.path.join(args.path, filename))

if __name__ == '__main__':
    sys.exit(main())