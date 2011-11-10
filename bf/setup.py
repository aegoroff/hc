__author__ = 'Alexander Egorov'

from distutils.core import setup
import py2exe, sys
from glob import glob

sys.argv.append('py2exe')


data_files = [("Microsoft.VC90.CRT", glob(r'C:\Program Files\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\*.*'))]

setup(
    name='Brute force tool',
    description='Hash algorithm brute force cracker',
    version='1.0.0.0',
    data_files=data_files,

    console=['bf.py'],
    options={'py2exe': {'bundle_files': 1, "optimize":2}},
    zipfile=None
)