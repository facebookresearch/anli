import os
from pathlib import Path

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PRO_ROOT = SRC_ROOT.parent

if __name__ == '__main__':
    a = 3

    for i in range(10):
        print(a + i)

    print("Hello Remote!")