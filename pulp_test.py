"""
Check the pulp library to be installed correctly
"""
from pulp import *

if __name__ == "__main__":
    # Run tests
    pulpTestAll()

    # Show list of available solvers
    print(f"Available solvers: {listSolvers(onlyAvailable=True)}")
