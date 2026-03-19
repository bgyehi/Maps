import json

import pandas as pd
from PIL.Image import module
from numpy.ma.extras import average

import config
from exact import milp_scheduling
from module import *
from exact import cp, milp
from heuristic import dispatching
from llm import PLAID, ATONCE, test, train
import llm.ATONCE_NOLIMIT
from exact.milpcg import solve_with_column_generation
import exact.milpcg2
Rules = ['SPT', 'EDD', 'MST', 'COVERT_nosetup', 'CR', 'SLACK']

if __name__ == '__main__':
    # test.finetuning()
    # train.upload_jsonl_openai('dataset/enumerate/train/train.jsonl', 'train')
    # train.upload_jsonl_openai('dataset/train2/validation/validate.jsonl', 'validate')
    # train.fine_tuning_openai()
    # config.OPENAI_MODEL = 'gpt-4.1-mini-2025-04-14'
    # config.OPENAI_MODEL = 'gpt-5-2025-08-07'
    # train.fine_tuning_openai()
    # train.fine_tuning_gemini()
    # train.fine_tuning_claude()
    # train.convert_jsonl_to_gemini_with_integer_output('dataset/train2/train.jsonl', 'dataset/train2/train_gemini2.jsonl')
    # train.fine_tuning_gemini()
    # summary = pd.DataFrame(columns=['Num', 'MILP', 'CP', 'ATCS', 'MST', 'FT_OpenAI', 'noFT_OpenAI', 'FT_Gemini'])
    # summary = pd.DataFrame(columns=['Num', 'FT_OpenAI', 'noFT_OpenAI', 'FT_Gemini'])
    # summary = pd.DataFrame(columns=['Num', 'OpenAI_wT', 'Claude_wT', 'Gemini_wT', 'Perplexity_wT', 'xAI_wT', 'OpenAI_Time', 'Claude_Time', 'Gemini_Time', 'Perplexity_Time', 'xAI_Time'])
    # summary = pd.DataFrame(
    #     columns=['Num', 'Base_wT', 'Mini_wT', 'Nano_wT', 'MiniF_wT', 'NanoF_wT',
    #              'Base_Time', 'Mini_Time', 'Nano_Time', 'MiniF_Time', 'NanoF_Time'])

    #
    # # Convert to DataFrame
    # df_results = pd.DataFrame(results)
    # df_results.sort_values(by='avg_gap', inplace=True)
    # df_results.to_csv('sensitivity.csv', index=False)
    summary = pd.DataFrame(columns=['Num', 'wT', 'Time', 'Status'])
    for i in range(1, 31):
        inst = load_instance_from_json("EO_data\Training_Small\instance_{}.json".format(i))
        solver = exact.milpcg2.solve_instance_auto(inst, 3600, 3600)
        new_row = {}
        new_row['Num'] = i
        new_row['wT'] = solver['objective']
        new_row['Time'] = solver['total_time_spent_sec']
        new_row['Status'] = solver['status']
        summary = summary._append(new_row, ignore_index=True)
        summary.to_excel("Lag_Test.xlsx", sheet_name='test')

    result = milp_scheduling(inst, 3600)

    summary = pd.DataFrame(columns=['Num', 'wT', 'Time', 'h'])

    # cnt = 1
    # tau_set = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # rho_set = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for tau in tau_set:
    #     for rho in rho_set:
    #         LB = (1 - rho - tau / 2)
    #         UB = (1 - rho + tau / 2)
    #         if LB >= -0.01:
    #             KIM_TAU = rho
    #             KIM_RHO = tau
    #             num_job = 5
    #             num_mch = 2
    #             instance = generate_prob(numJob=num_job, numMch=num_mch, setup=True, family=False, method='Schutten',
    #                                      identical_mch=False)
    #             # for job in instance.job_list:
    #             #     job.weight = 1
    #             # instance.objective = 'T'
    #
    #             new_row = {}
    #             cp_solution = cp.cp_scheduling(instance, 60)
    #             dr_solution = dispatching.scheduling(instance, 'ATCS')  # Fluctuate Performance
    #             openai = PLAID.scheduling_openai_keep(instance, with_ft=False)
    #
    #             new_row['wT'] = cp_solution.objective
    #             new_row['Time'] = openai.objective
    #             new_row['h'] = dr_solution.objective
    #
    #             instance.saveFile('dataset/example/instance_{0}.prob'.format(cnt))
    #             summary = summary._append(new_row, ignore_index=True)
    #             summary.to_excel("test_exam.xlsx", sheet_name='test')
    #             cnt+=1
    instance = generate_prob(numJob=5, numMch=2, setup=True, family=False, method='Schutten',
                             identical_mch=False)
    instance.loadFile('dataset/example/instance_{0}.prob'.format(16))

    for i in range(0, 36):
        # num_job = random.randint(100, 300)
        # num_mch = random.randint(4, 7)
        num_job = 5
        num_mch = 2
        instance = generate_prob(numJob=num_job, numMch=num_mch, setup=True, family=False, method='Schutten', identical_mch=False)
        instance.loadFile('dataset/noweight/instance_{0}.prob'.format(i + 1))

        # instance.saveFile('dataset/train2/instance_{0}.prob'.format(i + 1))
        # openai = llm.ATONCE_NOLIMIT.scheduling_openai(instance)
        # claude = llm.ATONCE_NOLIMIT.scheduling_claude(instance)
        # gemini = llm.ATONCE_NOLIMIT.scheduling_gemini(instance)
        # perplexity = llm.ATONCE_NOLIMIT.scheduling_perplexity(instance)
        # xai = llm.ATONCE_NOLIMIT.scheduling_xai(instance)
        # a, b = PLAID.get_total_token(instance)
        # cost = 2 * a/1000000 + 8 * b/1000000
        new_row = {}
        new_row['Num'] = "Train_Small_{}".format(i + 1)
        # new_row['OpenAI_wT'] = openai.objective
        # new_row['Claude_wT'] = claude.objective
        # new_row['Gemini_wT'] = gemini.objective
        # new_row['Perplexity_wT'] = perplexity.objective
        # new_row['xAI_wT'] = xai.objective
        # new_row['OpenAI_Time'] = openai.comp_time
        # new_row['Claude_Time'] = claude.comp_time
        # new_row['Gemini_Time'] = gemini.comp_time
        # new_row['Perplexity_Time'] = perplexity.comp_time
        # new_row['xAI_Time'] = xai.comp_time

        # cp_solution = train.retrieve_decisions_atonce(instance, 3600, 'dataset/noweight/train.jsonl')
        # instance.saveFile('dataset/validate/instance_{0}.prob'.format(i+1))
        # instance.loadFile('dataset/train/instance_{0}.prob'.format(i+1))
        # cp_solution = train.retrieve_decisions_atonce(instance, 3600, 'llm/training_full.jsonl')
        # openai = ATONCE.scheduling_openai(instance, 60)  # Fluctuate Performance
        # openai = PLAID.scheduling_openai(instance, with_ft=False)  # Fluctuate Performance
        # config.OPENAI_MODEL = 'gpt-4.1-mini-2025-04-14'
        # config.OPENAI_FT_MODEL_ID = 'ft:gpt-4.1-mini-2025-04-14:personal::BxXuMlUj'
        # openai_mini_1 = PLAID.scheduling_openai(instance, with_ft=False)  # Fluctuate Performance
        # openai_mini_2 = PLAID.scheduling_openai(instance, with_ft=True)  # Fluctuate Performance
        # config.OPENAI_MODEL = 'gpt-4.1-nano-2025-04-14'
        # config.OPENAI_FT_MODEL_ID = 'ft:gpt-4.1-nano-2025-04-14:personal::BxXpJ5KC'
        # openai_nano_1 = PLAID.scheduling_openai(instance, with_ft=False)  # Fluctuate Performance
        # openai_nano_2 = PLAID.scheduling_openai(instance, with_ft=False)  # Fluctuate Performance
        # claude = ATONCE.scheduling_claude(instance, 60)  # Fluctuate Performance
        # claude = PLAID.scheduling_claude(instance, with_ft=False)
        # gemini = ATONCE.scheduling_gemini(instance, 60)  # Fluctuate Performance
        # gemini = PLAID.scheduling_gemini(instance, with_ft=False)
        # perplexity = ATONCE.scheduling_perplexity(instance, 60)
        # xai = ATONCE.scheduling_xai(instance, 60)
        # cp_solution = cp.cp_scheduling(instance, 3600)
        # milp_solution = milp.milp_scheduling(instance, 3600)
        # cp_solution = cp.cp_scheduling(instance, 3600)
        # atcs = dispatching.scheduling(instance, rule='COVERT_new')
        # mst = dispatching.scheduling(instance, rule='MST')
        # test1 = PLAID.scheduling_openai(instance, with_ft=True)
        # test2 = PLAID.scheduling_openai(instance, with_ft=False)
        # test = PLAID.scheduling_gemini(instance, with_ft=False)
        # test = PLAID.scheduling_xai(instance)
        # test = dispatching.scheduling(instance, 'MDD')
        # PLAID.WITH_REASONING = True
        # config.OPENAI_MODEL = 'gpt-5-2025-08-07'
        # config.OPENAI_FT_MODEL_ID = 'ft:gpt-4.1-mini-2025-04-14:personal::BxXuMlUj'
        atc_solution = dispatching.scheduling(instance, 'ATCS')  # Fluctuate Performance
        # openai = PLAID.scheduling_openai(instance, with_ft=False)
        # config.OPENAI_FT_MODEL_ID = 'ft:gpt-4.1-2025-04-14:personal::C6BN22In'
        # openai = PLAID.scheduling_openai(instance, with_ft=False)
        atc_solution2 = PLAID.scheduling_openai_keep(instance, with_ft=False)  # Fluctuate Performance

        new_row['wT'] = atc_solution.objective
        new_row['Time'] = atc_solution.comp_time
        new_row['h'] = atc_solution.objective
        # best = None
        # for rule in Rules:
        #     test = dispatching.scheduling(instance, rule=rule)
        #     if best is None:
        #         best = test
        #     elif best.objective > test.objective:
        #         best = test
        #
        # new_row['Base_wT'] = best.objective
        # new_row['Base_Time'] = best.algorithm

        # new_row['Base_wT'] = openai.objective
        # new_row['Base_Time'] = openai.comp_time
        # new_row['Mini_wT'] = openai_mini_1.objective
        # new_row['Mini_Time'] = openai_mini_1.comp_time
        # new_row['Nano_wT'] = openai_nano_1.objective
        # new_row['Nano_Time'] = openai_nano_1.comp_time
        # new_row['MiniF_wT'] = openai_mini_2.objective
        # new_row['MiniF_Time'] = openai_mini_2.comp_time
        # new_row['NanoF_wT'] = openai_nano_2.objective
        # new_row['NanoF_Time'] = openai_nano_2.comp_time
        # summary = summary._append(new_row, ignore_index=True)

        # new_row['MILP'] = milp_solution.objective
        # new_row['CP'] = cp_solution.objective
        # # new_row['ATCS'] = atcs.objective
        # # new_row['MST'] = mst.objective
        # new_row['FT_OpenAI'] = test1.objective
        # new_row['noFT_OpenAI'] = test2.objective
        # new_row['FT_Gemini'] = test3.objective
        summary = summary._append(new_row, ignore_index=True)
        summary.to_excel("test_1.xlsx", sheet_name='test')
        print(i+1)

    edd_solution = dispatching.scheduling(instance, rule='MST')

    # milp_solution = milp.milp_scheduling(instance, 60)
    # milp_solution_edd = milp.milp_scheduling(instance, 60, init_sol=edd_solution)
    # cp_solution = cp.cp_scheduling(instance, 60)
    # cp_solution_edd = cp.cp_scheduling(instance, 60, init_sol=edd_solution)
    # cp_solution_ortools = cp.cp_scheduling_ortools(instance, 60)
    # openai_solution = ATONCE.scheduling(instance, model='openai')
    openai_solution_1 = PLAID.scheduling(instance, model='openai')
    print('Done')
