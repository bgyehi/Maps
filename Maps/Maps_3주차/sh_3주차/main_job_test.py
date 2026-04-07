import json

import pandas as pd
from PIL.Image import module
from numpy.ma.extras import average

import config
from milp import milp_scheduling
from module import *

if __name__ == '__main__':

    for i in range(10, 16):

        num_job = 5+i
        num_mch = 2
        instance = generate_prob(numJob=num_job, numMch=num_mch, setup=True, family=False, method='Schutten', identical_mch=False)



        result = milp_scheduling(instance, 300)

        print(f'job: {num_job} mch: {num_mch}')
        print("Objective (wT):", result.objective)
        print("Status:", result.status)
        print("Solve Time:", result.comp_time)
    print('Done')
