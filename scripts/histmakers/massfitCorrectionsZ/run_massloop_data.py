# Runs all the steps
# Once for Iter0: ./massscales_data ./massfit ./resolfit
# For niter times: ./massscales_data(command updated to use previous fit values) ./massfit ./resolfit 
# Authors: Cristina Alexe, Lorenzo Bianchini

import argparse
import os
import sys
import copy
import math
import ROOT
import time
   
parser = argparse.ArgumentParser(description='run')

parser.add_argument('--none', action='store_true'  , help = 'none')
parser.add_argument('--dryrun', action='store_true'  , help = 'dry run')
parser.add_argument('--tag',   default='PostVFP' , help = 'type of data used')
parser.add_argument('--niter', dest = 'niter'  , type = int,  default=1, help='number of iterations after the 0th')
parser.add_argument('--forceIter', dest = 'forceIter'  , type = int,  default=-1, help='will only do a specific iteration and skip the rest')

args = parser.parse_args()

def loop_one():    

    assert args.forceIter <= args.niter 
    tag = args.tag
    cmd_histo_iter0 = './massscales_data --firstIter=-1 --lastIter=2 '+\
        ' --tag='+tag+' '+\
        ' --run=Iter0 '+\
        ' --nRMSforGausFit=-1 '+\
        ' --minNumEvents=100 --minNumEventsPerBin=30 '+\
        ' --minNumMassBins=4 '+\
        ' --rebin=2 '+\
        ' --fitNorm --fitWidth '+\
        '  --y2016 --scaleToData '
    # --lumi
    if not args.forceIter>0:
        print(cmd_histo_iter0)
    if not (args.dryrun or args.forceIter>0):
        os.system(cmd_histo_iter0)
    cmd_fit_iter0 = './massfit --ntoys=1 --bias=-1 '+\
        '--tag='+tag+' '+\
        '--run=Iter0 '
    if not args.forceIter>0:
        print(cmd_fit_iter0)
    if not (args.dryrun or args.forceIter>0):
        os.system(cmd_fit_iter0)
    cmd_resol_iter0 = './resolfit --ntoys=1 --bias=-1 '+\
        ' --tag='+tag+' '+\
        ' --run=Iter0 '+\
        ' --maxSigmaErr=0.1 '
    if not args.forceIter>0:
        print(cmd_resol_iter0)
    if not (args.dryrun or args.forceIter>0):
        os.system(cmd_resol_iter0)

    for iter in range(1, args.niter+1):
        if (args.forceIter>0 and iter!=args.forceIter) or args.forceIter==0 :
            continue
        cmd_histo_iteri = cmd_histo_iter0.replace('--run=Iter0', '--run=Iter'+str(iter))
        cmd_histo_iteri += ' --usePrevMassFit '+\
            ' --tagPrevMassFit='+tag+' '+\
            ' --runPrevMassFit=Iter'+str(iter-1)+' '
        cmd_histo_iteri += ' --usePrevResolFit '+\
            ' --tagPrevResolFit='+tag+' '+\
            ' --runPrevResolFit=Iter'+str(iter-1)+' '
        print(cmd_histo_iteri)
        if not args.dryrun:
            os.system(cmd_histo_iteri)
        cmd_fit_iteri = cmd_fit_iter0.replace('--run=Iter0', '--run=Iter'+str(iter))
        print(cmd_fit_iteri)
        if not args.dryrun:
            os.system(cmd_fit_iteri)
        cmd_resol_iteri = cmd_resol_iter0.replace('--run=Iter0', '--run=Iter'+str(iter))
        print(cmd_resol_iteri)
        if not args.dryrun:
            os.system(cmd_resol_iteri)
    return


if __name__ == '__main__':
    start = time.time()
    print('Running on data and MC')
    loop_one()
    end = time.time()
    print('Done', args.niter, 'iterations in', (end - start)/60., 'min.')
