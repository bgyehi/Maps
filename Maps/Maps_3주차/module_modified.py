import pickle
import copy
import json
import re
from typing import List

import pandas as pd

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
        return 'Job {0}'.format(str(self.ID)) + " {0}".format(
            "- Family " + str(self.family) if self.family is not None else ""
        )

    def __eq__(self, other):
        return isinstance(other, Job) and (other.ID == self.ID) and (other.due == self.due)

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
        return Instance(
            [copy.deepcopy(job) for job in self.job_list],
            [copy.deepcopy(mch) for mch in self.machine_list],
            copy.deepcopy(self.ptime),
            copy.deepcopy(self.setup),
        )

    def __repr__(self):
        return 'Instance with {0} jobs and {1} machines'.format(self.numJob, self.numMch)

    def getPTime(self, job: Job, machine: Machine):
        return self.ptime[machine.ID][job.ID]

    def getSetup(self, job_i: Job, job_j: Job, machine: Machine):
        return self.setup[machine.ID][job_i.ID][job_j.ID]

    def findJob(self, id: int) -> Job:
        return [i for i in self.job_list if i.ID == id][0]

    def findMch(self, id: int) -> Machine:
        return [i for i in self.machine_list if i.ID == id][0]

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
            self.job.ID, self.machine, self.start - self.setup, self.start, self.setup, self.start, self.end
        )


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
                cmax = max(cmax, job.end)
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


def _parse_machine_idx_csv(value):
    s = str(value).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        raise ValueError(f"machine 값을 해석할 수 없습니다: {value}")
    return int(m.group(1))


def load_instance_from_csv(job_csv_path: str, setup_csv_path: str) -> Instance:
    jobs_df = pd.read_csv(job_csv_path)
    setup_df = pd.read_csv(setup_csv_path)

    jobs_df.columns = [str(c).strip() for c in jobs_df.columns]
    setup_df.columns = [str(c).strip() for c in setup_df.columns]

    required_job_cols = {"id", "due_date", "weight"}
    missing = required_job_cols - set(jobs_df.columns)
    if missing:
        raise ValueError(f"jobs.csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    jobs_df = jobs_df.sort_values("id").reset_index(drop=True)

    pt_info = []
    for col in jobs_df.columns:
        m = re.fullmatch(r"pt_Machine\s*(\d+)", str(col).strip())
        if m:
            pt_info.append((int(m.group(1)), col))
    pt_info.sort(key=lambda x: x[0])

    if not pt_info:
        raise ValueError("jobs.csv에서 pt_Machine 0, pt_Machine 1 같은 컬럼을 찾지 못했습니다.")

    csv_job_ids = jobs_df["id"].astype(int).tolist()
    job_id_to_idx = {job_id: idx for idx, job_id in enumerate(csv_job_ids)}

    num_jobs = len(csv_job_ids)
    num_machines = len(pt_info)

    jobs = []
    for _, row in jobs_df.iterrows():
        csv_id = int(row["id"])
        i = job_id_to_idx[csv_id]
        job = Job(i)
        job.due = int(row["due_date"])
        job.weight = int(row["weight"])
        jobs.append(job)

    machines = [Machine(k) for k in range(num_machines)]

    ptime = [[0 for _ in range(num_jobs)] for _ in range(num_machines)]
    for k, col in pt_info:
        for _, row in jobs_df.iterrows():
            csv_id = int(row["id"])
            i = job_id_to_idx[csv_id]
            ptime[k][i] = int(row[col])
        machines[k].ptime = ptime[k]

    setup = [[[0 for _ in range(num_jobs)] for _ in range(num_jobs)] for _ in range(num_machines)]

    setup_job_cols = []
    for col in setup_df.columns:
        m = re.fullmatch(r"job(\d+)", str(col).strip(), re.IGNORECASE)
        if m:
            setup_job_cols.append((int(m.group(1)), col))
    setup_job_cols.sort(key=lambda x: x[0])

    if not setup_job_cols:
        raise ValueError("setup_times.csv에서 job1, job2, ... 컬럼을 찾지 못했습니다.")

    for _, row in setup_df.iterrows():
        k = _parse_machine_idx_csv(row["machine"])
        from_job_csv_id = int(row["id"])
        i = job_id_to_idx[from_job_csv_id]

        for to_job_csv_id, col in setup_job_cols:
            j = job_id_to_idx[to_job_csv_id]
            setup[k][i][j] = int(row[col])

    for k in range(num_machines):
        machines[k].setup = setup[k]

    inst = Instance(jobs, machines, ptime, setup)
    inst.with_setup = True
    inst.objective = "wT"
    return inst
