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
import datetime

__author__ = 'Alexander Egorov'

_RESULT_DIR = 'hql'
_QUERY_TEMPLATE = "# {0} ({1})\n# {2}\n\nfor file f from parameter where\n{3}\ndo validate;"


def CreateQueryFromTridXml(path):
    if path.rfind(".trid.xml") == -1:
            return
    logging.debug("processing %s", path)
    title = ''
    sign_ext = ''
    descr = ''
    sign_bytes = ''
    offset = 0
    with open(path, 'r') as f:
        where_parts = []
        for event, element in etree.iterparse(f, events=("start", "end")):
            if event == 'start':
                if element.tag == 'FileType':
                    title = element.text
                if element.tag == 'Ext':
                    sign_ext = element.text
                if element.tag == 'Rem':
                    descr = element.text
                if element.tag == 'Bytes':
                    sign_bytes = element.text
                if element.tag == 'Pos':
                    offset = int(element.text)
            if event == 'end':
                if element.tag == 'FrontBlock':
                    break
                if element.tag == 'Pattern':
                    binary = binascii.unhexlify(sign_bytes)
                    m = hashlib.md5(binary)
                    d = m.hexdigest()
                    item = "(f.offset == {0:d} and f.limit == {1:d} and f.md5 == '{2}')".format(offset, len(binary), d)
                    where_parts.append(item)

        where = ' and\n'.join(where_parts)
        if descr is None:
            descr = ''
        q = _QUERY_TEMPLATE.format(title, sign_ext, descr, where)

        directory, name = os.path.split(path)
        root, ext = os.path.splitext(name)
        file_path = os.path.join(_RESULT_DIR, "{0}.hql".format(root))
        with open(file_path, "w") as qf:
            qf.write(q)


def main():
    logging.basicConfig(format=("%(asctime).19s %(levelname)s %(message)s "))

    parser = argparse.ArgumentParser(description="TRiD signatures converting tool. Copyright (C) 2012 Alexander Egorov.")
    parser.add_argument('-p', '--path', dest='path', help='Path to TRiD signature files', required=True)
    parser.add_argument('-d', '--destination', dest='destination', help='Path to destination dir', default=_RESULT_DIR)
    parser.add_argument('-v', '--verbose', dest='verbose', help='Verbose output', action='store_true', default=False)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    start = datetime.datetime.now()
    files = os.listdir(args.path)

    if os.path.isdir(args.destination):
        shutil.rmtree(args.destination, True)
    os.mkdir(args.destination)

    map(lambda filename: CreateQueryFromTridXml(os.path.join(args.path, filename)), files)
    finish = datetime.datetime.now()
    logging.info('Completed. Time: %s', finish - start)

if __name__ == '__main__':
    sys.exit(main())