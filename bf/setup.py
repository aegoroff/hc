__author__ = 'Alexander Egorov'

from distutils.core import setup
import py2exe, sys

sys.argv.append('py2exe')

setup(
    name='Brute force tool',
    description='Hash algorithm brute force cracker',
    version='1.0.0.0',

    console=['bf.py'],
    options={'py2exe': {'bundle_files': 1, "optimize":2}},
    zipfile=None
)