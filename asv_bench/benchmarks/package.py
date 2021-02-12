
'\nBenchmarks for pandas at the package-level.\n'
import subprocess
import sys

class TimeImport():

    def time_import(self):
        cmd = [sys.executable, '-X', 'importtime', '-c', 'import pandas as pd']
        p = subprocess.run(cmd, stderr=subprocess.PIPE)
        line = p.stderr.splitlines()[(- 1)]
        field = line.split(b'|')[(- 2)].strip()
        total = int(field)
        return total
