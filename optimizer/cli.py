from __future__ import annotations
import argparse, sys
from optimizer import router

def main(argv=None):
    p = argparse.ArgumentParser(description="Unified optimizer runner (problems 2-5)")
    p.add_argument('--problem','-p', type=int, default=router.PROBLEM, choices=[2,3,4,5], help='Problem id (2-5)')
    p.add_argument('--algo','-a', type=str, default=router.ALGO, choices=['sa','ga','pso','hybrid'], help='Algorithm')
    p.add_argument('--bombs','-b', type=int, default=router.BOMBS_COUNT, help='Bombs per drone (Q3/Q5 only)')
    p.add_argument('--dt', type=float, default=router.DT, help='Time step')
    p.add_argument('--method','-m', type=str, default=router.OCCLUSION_METHOD, choices=['sampling','judge_caps','vectorized_sampling'], help='Occlusion method')
    p.add_argument('--backend', type=str, default=router.BACKEND, help='judge_caps fast backend (rough|vectorized* etc)')
    p.add_argument('--iters', type=int, default=None, help='Iterations / generations override')
    p.add_argument('--gens', type=int, default=None, help='GA generations override')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--verbose','-v', action='store_true')
    args = p.parse_args(argv)
    # apply to router module globals
    router.PROBLEM = args.problem
    router.ALGO = args.algo
    router.BOMBS_COUNT = args.bombs
    router.DT = args.dt
    router.OCCLUSION_METHOD = args.method
    router.BACKEND = args.backend
    router.ITERS = args.iters
    router.GA_GENERATIONS = args.gens
    router.SEED = args.seed
    router.VERBOSE = args.verbose
    router.run()

if __name__ == '__main__':
    main()
