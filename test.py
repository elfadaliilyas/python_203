import sys
from streamlit import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "launch.py"]
    sys.exit(stcli.main())
