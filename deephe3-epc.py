#!/usr/bin/env python
# ===================================================================== #
# Electron-phonon coupling from a trained model via finite differences  #
# ===================================================================== #

# Usage: python <path-to-this-file>/deephe3-epc.py <your_config>.ini [-n NUM_THREADS] [--debug]
# Default config file is maceh/default_configs/epc_default.ini

import os
import argparse

parser = argparse.ArgumentParser(
    description='Compute the Cartesian AO electron-phonon coupling g_ij,ka(k,q) from '
                'finite-difference derivatives of the predicted Hamiltonian')
parser.add_argument('config', type=str, metavar='CONFIG', help='Config file for EPC calculation')
parser.add_argument('-n', type=int, default=None, help='Maximum number of threads')
parser.add_argument('--debug', action='store_true',
                    help='Fill unpredicted matrix elements with 0 instead of throwing error.')
args = parser.parse_args()

if args.n is not None:
    os.environ["OMP_NUM_THREADS"] = f"{args.n}"
    os.environ["MKL_NUM_THREADS"] = f"{args.n}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{args.n}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{args.n}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{args.n}"
    import torch
    torch.set_num_threads(args.n)

from maceh.epc.run import run_epc
run_epc(args.config, debug=args.debug)
