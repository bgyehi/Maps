import pickle
import copy
import numpy as np
import pandas as pd
import json
import os
from typing import List

OBJECTIVE_FUNCTION = 'wT'


class Job:
    def __init__(self, _id: int):
        self.ID = _id
        self.due = -1
        self.weight = 0
        self.family = None

        self.complete = False
        self.start = -1
        self.end = -1
        self.assignedMch = -1
        self.priority = 0

    def __repr__(self):
        return 'Job {0}'.format(str(self.ID)) + \
               " {0}".format("- Family " + str(self.family) if self.family is not None else "")

    def __eq__(self, other):
        if isinstance(other, Job):
            return (other.ID == self.ID) and (other.due == self.due)
        return False

    def get_setups(self, mch_list):
        setup_times = [mch.get_setup(self) for mch in mch_list]
        return {
            'Min': min(setup_times),
            'Max': max(setup_times),
            'Avg': sum(setup_times) / len(setup_times)
        }

    def get_ptimes(self, mch_list):
        ptimes = [mch.get_ptime(self) for mch in mch_list]
        return {
            'Min': min(ptimes),
            'Max': max(ptimes),
            'Avg': sum(ptimes) / len(ptimes)
        }

    def get_min_comp(self, mch_list):
        min_comp = float("inf")
        for mch in mch_list:
            exp_comp = mch.available + mch.get_setup(self) + mch.get_ptime(self)
            if exp_comp < min_comp:
                min_comp = exp_comp
        return min_comp

    def to_dict(self):
        return {
            "ID": self.ID,
            "due": self.due,
            "weight": self.weight,
            "family": self.family,
            "complete": self.complete,
            "start": self.start,
            "end": self.end,
            "assignedMch": self.assignedMch if isinstance(self.assignedMch, int) else (
                self.assignedMch.ID if self.assignedMch else None
            ),
            "priority": self.priority
        }


class Machine:
    def __init__(self, _id: int):
        self.ID = _id
        self.available = 0
        self.assigned = []
        self.setup = None
        self.ptime = None
        self.schedules = []
        self.priority = 0

    def __repr__(self):
        return 'Machine ' + str(self.ID)

    def get_setup(self, job: Job):
        if len(self.assigned) == 0:
            return 0
        return self.setup[self.assigned[-1].ID][job.ID]

    def get_ptime(self, job: Job):
        return self.ptime[job.ID]

    def process(self, job: Job):
        job.assignedMch = self
        ptime = self.ptime[job.ID]
        setup = self.get_setup(job)

        job.start = self.available + setup
        self.available += (ptime + setup)
        job.end = self.available
        job.complete = True

        self.assigned.append(job)
        self.schedules.append(Bar(job, setup))

    def get_min_comp(self, job_list: List[Job]):
        min_comp = float("inf")
        for job in job_list:
            exp_comp = self.available + self.get_setup(job) + self.ptime[job.ID]
            if exp_comp < min_comp:
                min_comp = exp_comp
        return min_comp

    def to_dict(self):
        return {
            "ID": self.ID,
            "available": self.available,
            "assigned": [job.ID for job in self.assigned],
            "setup": self.setup,
            "ptime": self.ptime,
            "schedules": [str(s) for s in self.schedules],
            "priority": self.priority
        }


class Instance:
    type = 'PMSP'

    def __init__(self, jobs: list, mchs: list, ptime, setups):
        self.numJob = len(jobs)
        self.numMch = len(mchs)
        self.job_list = jobs
        self.machine_list = mchs
        self.ptime = ptime
        self.setup = setups
        self.with_setup = True
        self.family_setup = False
        self.objective = OBJECTIVE_FUNCTION
        self.identical_mch = False

    def deepcopy(self):
        job_list = [copy.deepcopy(job) for job in self.job_list]
        mch_list = [copy.deepcopy(mch) for mch in self.machine_list]
        ptime = copy.deepcopy(self.ptime)
        setups = copy.deepcopy(self.setup)
        return Instance(job_list, mch_list, ptime, setups)

    def make_subprob(self, mch_id: int, job_id: int):
        job_list = [copy.deepcopy(job) for job in self.job_list]
        mch_list = [copy.deepcopy(mch) for mch in self.machine_list]
        ptime = copy.deepcopy(self.ptime)
        setups = copy.deepcopy(self.setup)
        result = Instance(job_list, mch_list, ptime, setups)
        result.findMch(mch_id).process(result.findJob(job_id))
        return result

    def __repr__(self):
        return 'Instance with {0} jobs and {1} machines'.format(self.numJob, self.numMch)

    def getPTime(self, job: Job, machine: Machine):
        return self.ptime[machine.ID][job.ID]

    def getSetup(self, job_i: Job, job_j: Job, machine: Machine):
        return self.setup[machine.ID][job_i.ID][job_j.ID]

    def findJob(self, id: int) -> Job:
        result = [i for i in self.job_list if i.ID == id][0]
        return result

    def findMch(self, id: int) -> Machine:
        result = [i for i in self.machine_list if i.ID == id][0]
        return result

    def saveFile(self, path: str):
        with open(path, mode='wb') as fw:
            pickle.dump(self, fw)

    def loadFile(self, path: str):
        with open(path, mode='rb') as fr:
            instance = pickle.load(fr)
            self.numJob = instance.numJob
            self.numMch = instance.numMch
            self.job_list = instance.job_list
            self.machine_list = instance.machine_list
            self.ptime = instance.ptime
            self.setup = instance.setup
            self.with_setup = instance.with_setup
            self.family_setup = instance.family_setup
            self.objective = instance.objective
            self.identical_mch = instance.identical_mch

    def to_dict(self):
        return {
            "numJob": self.numJob,
            "numMch": self.numMch,
            "jobs": [job.to_dict() for job in self.job_list],
            "machines": [mch.to_dict() for mch in self.machine_list],
            "ptime": self.ptime,
            "setup": self.setup,
            "with_setup": self.with_setup,
            "family_setup": self.family_setup,
            "objective": str(self.objective),
            "identical_mch": self.identical_mch
        }


class Schedule:
    def __init__(self, _alg: str, instance, obj: float):
        self.algorithm = _alg
        self.instance = instance
        self.objective = obj
        self.bars = []
        self.comp_time = 'None'
        self.status = 'None'

        for m in self.instance.machine_list:
            for bar in m.schedules:
                self.bars.append(bar)

    def __repr__(self):
        return 'Schedule obtained by {0} - Objective: {1} (Total Setup Times: {2})'.format(
            self.algorithm, self.objective, get_total_setups(self.instance)
        )

    def print_schedule(self):
        for m in self.instance.machine_list:
            for bar in m.schedules:
                print(bar)

    def save_schedule_html(self):
        pass


class Bar:
    def __init__(self, job, setup: int):
        self.seq = job.ID
        self.job = job
        self.machine = job.assignedMch.ID
        self.start = job.start
        self.end = job.end
        self.setup = setup

    def __repr__(self):
        return 'Job {0} at Machine {1} : Setup ({2} - {3}, {4}) Processing {5} - {6}'.format(
            self.job.ID, self.machine, self.start - self.setup, self.start,
            self.setup, self.start, self.end
        )


def build_instance_from_csv(jobs_csv: str, setup_csv: str) -> Instance:
    jobs_df = pd.read_csv(jobs_csv)
    setup_df = pd.read_csv(setup_csv)

    jobs_df = jobs_df.sort_values("id").reset_index(drop=True)

    if "weight" not in jobs_df.columns:
        raise ValueError("jobs.csv에 weight 컬럼이 없습니다.")

    pt_cols = [c for c in jobs_df.columns if c.startswith("pt_")]
    num_jobs = len(jobs_df)
    num_mchs = len(pt_cols)

    job_list: List[Job] = []
    for _, row in jobs_df.iterrows():
        job = Job(int(row["id"]) - 1)
        job.due = int(row["due_date"])
        job.weight = int(row["weight"])
        job_list.append(job)

    machine_list: List[Machine] = [Machine(i) for i in range(num_mchs)]

    ptime = []
    for m, col in enumerate(pt_cols):
        p_list = [int(v) for v in jobs_df[col].tolist()]
        ptime.append(p_list)
        machine_list[m].ptime = p_list

    setup = [[[0 for _ in range(num_jobs)] for _ in range(num_jobs)] for _ in range(num_mchs)]

    for m in range(num_mchs):
        machine_name = f"Machine {m}"
        sub_df = setup_df[setup_df["machine"] == machine_name].copy()
        sub_df = sub_df.sort_values("id").reset_index(drop=True)

        if len(sub_df) != num_jobs:
            raise ValueError(f"{machine_name}의 setup row 수가 jobs 수와 다릅니다.")

        for i in range(num_jobs):
            for j in range(num_jobs):
                col_name = f"job{j+1}"
                setup[m][i][j] = int(sub_df.loc[i, col_name])

        machine_list[m].setup = setup[m]

    instance = Instance(job_list, machine_list, ptime, setup)
    instance.with_setup = True
    instance.objective = "wT"
    return instance


def save_instance_to_json(instance: Instance, json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(instance.to_dict(), f, ensure_ascii=False, indent=2)


def load_instance_from_json(path: str) -> Instance:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    job_list: List[Job] = []
    for j in data["jobs"]:
        job = Job(j["ID"])
        job.due = j["due"]
        job.weight = j["weight"]
        job.family = j["family"]
        job.complete = j["complete"]
        job.start = j["start"]
        job.end = j["end"]
        job.priority = j.get("priority", 0)
        job_list.append(job)

    job_dict = {job.ID: job for job in job_list}

    machine_list: List[Machine] = []
    for m in data["machines"]:
        mch = Machine(m["ID"])
        mch.available = m["available"]
        mch.priority = m.get("priority", 0)
        mch.ptime = m["ptime"]
        mch.setup = m["setup"]
        machine_list.append(mch)

    mch_dict = {mch.ID: mch for mch in machine_list}

    for m in data["machines"]:
        mch = mch_dict[m["ID"]]
        for job_id in m["assigned"]:
            job = job_dict[job_id]
            job.assignedMch = mch
            mch.assigned.append(job)

    instance = Instance(
        jobs=job_list,
        mchs=machine_list,
        ptime=data["ptime"],
        setups=data["setup"]
    )

    instance.with_setup = data.get("with_setup", True)
    instance.family_setup = data.get("family_setup", False)
    instance.objective = data.get("objective", OBJECTIVE_FUNCTION)
    instance.identical_mch = data.get("identical_mch", False)

    return instance


def load_or_create_instance(json_path: str, jobs_csv: str, setup_csv: str) -> Instance:
    if os.path.exists(json_path):
        return load_instance_from_json(json_path)

    instance = build_instance_from_csv(jobs_csv, setup_csv)
    save_instance_to_json(instance, json_path)
    return instance


def get_obj(prob: Instance, objective=OBJECTIVE_FUNCTION):
    result = 0

    if objective == 'C':
        for m in prob.machine_list:
            for job in m.assigned:
                result += job.end

    elif objective == 'Cmax':
        cmax = 0
        for m in prob.machine_list:
            for job in m.assigned:
                if cmax < job.end:
                    cmax = job.end
        result = cmax

    elif objective == 'T':
        for m in prob.machine_list:
            for job in m.assigned:
                result += max(job.end - job.due, 0)

    elif objective == 'wT':
        for m in prob.machine_list:
            for job in m.assigned:
                result += job.weight * max(job.end - job.due, 0)

    return result


def get_total_setups(prob: Instance):
    result = 0
    for m in prob.machine_list:
        for bar in m.schedules:
            result += bar.setup
    return result