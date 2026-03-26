import pickle # Instance 객체를 파일로 저장/불러오기 위한 모듈
import random # 랜덤 인스턴스 생성을 위한 난수 모듈
from typing import List
import copy # 원본 객체를 건드리지 않으려고 복사본 만들 때 사용
import math
import numpy as np
import json
from typing import Dict, Any, List

OBJECTIVE_FUNCTION = 'wT'
GEN_METHOD = ['Kim', 'Lin', 'Schutten']
LIN_DENOMINATOR = 7
LIN_TAU = 0.8
LIN_R = 0.4
KIM_TAU = 0.4
KIM_RHO = 0.8
SCHUTTEN_SETUP = 0.6
SCHUTTEN_TIGHT = 0.6

def CompareTwoValues(A, B) -> str:
    if A == B:
        return '='
    elif A > B:
        return '>'
    else:
        return '<'

# 하나의 job을 표현하는 클래
class Job():

    def __init__(self, _id: int):
        # Parameters given by User
        self.ID = _id  # instance variable unique to each instance # 작업번호
        self.due = -1 # 납기일
        self.weight = 0 # 가중치
        self.family = None
        # Variables changed over time during scheduling
        self.complete = False # 작업 완료 여부
        self.start = -1  # 작업 시작 시간
        self.end = -1  # 작업이 끝나는 시간
        self.assignedMch = -1 # 배정된 기계
        self.priority = 0 #우선순위 정보

    def __repr__(self):
        return 'Job {0}'.format(str(self.ID)) + " {0}".format("- Family " + str(self.family) if self.family is not None else "")

    def __eq__(self, other):
        if isinstance(other, Job):
            if (other.ID == self.ID) and (other.due == self.due):
                return True
        return False

    def get_setups(self, mch_list):
        setup_times = [mch.get_setup(self) for mch in mch_list]
        result = {}
        result['Min'] = min(setup_times)
        result['Max'] = max(setup_times)
        result['Avg'] = sum(setup_times) / len(setup_times)
        return result

    def get_ptimes(self, mch_list):
        ptimes = [mch.get_ptime(self) for mch in mch_list]
        result = {}
        result['Min'] = min(ptimes)
        result['Max'] = max(ptimes)
        result['Avg'] = sum(ptimes) / len(ptimes)
        return result

    def get_min_comp(self, mch_list):
        min_comp = float("inf")
        for mch in mch_list:
            exp_comp = mch.available + mch.get_setup(self) + mch.get_ptime(self)
            if exp_comp < min_comp:
                min_comp = exp_comp
        return min_comp

    # Job 객체를 JSON 저장용 딕셔너리 형태로 변환
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


# 하나의 machine을 표현하는 클래스
class Machine:

    def __init__(self, _id: int):
        self.ID = _id  # instance variable unique to each instance
        self.available = 0  # 작업이 가능한 시점
        self.assigned = []  # machine에 할당된 job list
        self.setup = None
        self.ptime = None
        self.schedules = []
        self.priority = 0

    def __repr__(self):
        return 'Machine ' + str(self.ID)

    # 현재 machine의 마지막 작업 이후에 새 job을 처리할 때 필요한 setup time 반환
    def get_setup(self, job: Job):
        if len(self.assigned) == 0:
            return 0
        else:
            return self.setup[self.assigned[-1].ID][job.ID]

    # 해당 machine에서 job의 processing time 반환
    def get_ptime(self, job: Job):
        return self.ptime[job.ID]

    # 작업 1개를 실제로 machine 위에 올려서 스케줄 상태를 갱신하는 함수
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
            "schedules": [str(s) for s in self.schedules],  # Bar 객체라면 별도 처리 필요
            "priority": self.priority
        }


# 스케줄링 문제 전체를 나타내는 클래스
class Instance:
    type = 'PMSP'

    def __init__(self, jobs: list, mchs: list, ptime, setups):
        self.numJob = len(jobs)
        self.numMch = len(mchs)
        self.job_list = jobs
        self.machine_list = mchs
        self.ptime = ptime  # Process Times as 2-Dim Array
        self.setup = setups  # Setup Times as 2-Dim Array
        self.with_setup = True
        self.family_setup = False
        self.objective = OBJECTIVE_FUNCTION
        self.identical_mch = False

    def deepcopy(self):
        job_list = [copy.deepcopy(job) for job in self.job_list]
        mch_list = [copy.deepcopy(mch) for mch in self.machine_list]
        ptime = copy.deepcopy(self.ptime)
        setups = copy.deepcopy(self.setup)
        result = Instance(job_list, mch_list, ptime, setups)
        return result

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
        return self.ptime[machine.ID][job.ID]  # add time return code

    def getSetup(self, job_i: Job, job_j: Job, machine: Machine):
        return self.setup[machine.ID][job_i.ID][job_j.ID]

    def findJob(self, id: int) -> Job:
        try:
            result = [i for i in self.job_list if i.ID == id][0]
        except ValueError:
            print("No Matching Job in List")
        return result

    def findMch(self, id: int) -> Machine:
        try:
            result = [i for i in self.machine_list if i.ID == id][0]
        except ValueError:
            print("No Matching Machine in List")
        return result

    # Instance 객체를 pickle(.prob) 형식으로 저장 / 불러오기
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


#스케줄 결과를 저장하는 클래스
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
        return 'Schedule obtained by {0} - Objective: {1} (Total Setup Times: {2})'.format(self.algorithm, self.objective, get_total_setups(self.instance))

    def print_schedule(self):
        for m in self.instance.machine_list:
            for bar in m.schedules:
                print(bar)

    def save_schedule_html(self):
        pass #TODO: To be implemented


#스케줄 상에서 하나의 작업 구간을 표현하는 클래스
class Bar:
    def __init__(self, job, setup: int):
        self.seq = job.ID
        self.job = job
        self.machine = job.assignedMch.ID
        self.start = job.start
        self.end = job.end
        self.setup = setup

    def __repr__(self):
        return 'Job {0} at Machine {1} : Setup ({2} - {3}, {4}) Processing {5} - {6}'.format(self.job.ID, self.machine, self.start - self.setup, self.start, self.setup, self.start, self.end)

#랜덤 스케줄링 인스턴스를 생성하는 함수
def generate_prob(numJob: int, numMch: int, setup: bool=True, family: bool=False, identical_mch: bool=False, method: str='Kim') -> Instance:

    job_list = []
    machine_list = []
    jobs = [*range(0, numJob)]
    machines = [*range(0, numMch)]
    if family:
        if method == 'Kim':
            # batch and batch group
            num_family = random.randint(2, math.trunc(numJob / 5))
        elif method == 'Lin':
            num_family = math.trunc(numJob / LIN_DENOMINATOR)
        elif method == 'Schutten':
            num_family = random.randint(2, max(2, math.trunc(numJob / 5)))
    else:
        num_family = numJob
    families = [*range(0, num_family)]

    job_list += [Job(i) for i in jobs]
    machine_list += [Machine(i) for i in machines]
    for j in jobs:
        if not family:
            job_list[j].family = families[j]
        else:
            if method == 'Kim':
                job_list[j].family = random.randint(2, math.trunc(numJob / 5))
            elif method == 'Lin':
                job_list[j].family = math.trunc(numJob / LIN_DENOMINATOR)
            elif method == 'Schutten':
                job_list[j].family = random.choice(families)

    if identical_mch:
        identical_ptimes = [random.randint(1, 100) for j in jobs]
        ptimes = [identical_ptimes for m in machines]
        ptimes_avg = np.mean(np.array(ptimes))

        for m in machines:
            machine_list[m].ptime = ptimes[m]
            machine_list[m].available = 0
    else:
        # ptimes = [[random.randint(1, 100) for j in jobs] for m in machines]
        ptimes = [[random.randint(30, 60) for j in jobs] for m in machines]
        ptimes_avg = np.mean(np.array(ptimes))

        for m in machines:
            machine_list[m].ptime = ptimes[m]
            machine_list[m].available = 0

    if setup:
        setup_matrix = [[[0 for j2 in jobs] for j1 in jobs] for m in machines]
        # fam_setups = [[[random.randint(1, math.floor(SCHUTTEN_SETUP*ptimes_avg)) for f2 in families] for f1 in families] for m in machines]
        fam_setups = [[[random.randint(10, 90) for f2 in families] for f1 in families] for m in machines]
        for m in machines:
            for j1 in jobs:
                for j2 in jobs:
                    if job_list[j1].family != job_list[j2].family:
                        setup_matrix[m][j1][j2] = fam_setups[m][job_list[j1].family][job_list[j2].family]
                    else:
                        setup_matrix[m][j1][j2] = 0
            machine_list[m].setup = setup_matrix[m]
            machine_list[m].family_setup_times = fam_setups[m]

        for j in jobs:
            P = (np.min(np.array(ptimes))+np.min(np.array(fam_setups)))*numJob/numMch
            lb = round(P*(1-KIM_RHO-KIM_TAU/2))
            ub = round(P*(1-KIM_RHO+KIM_TAU/2))
            # lb = round(job_list[j].get_ptimes(machine_list)['Max'])
            # ub = round(job_list[j].get_ptimes(machine_list)['Max'] + SCHUTTEN_TIGHT*ptimes_avg)
            job_list[j].due = random.randint(lb, ub)
            job_list[j].weight = random.randint(1, 10)
    else:
        setup_matrix = [[[0 for j2 in jobs] for j1 in jobs] for m in machines]
        for m in machines:
            machine_list[m].setup = setup_matrix[m]

        for j in jobs:
            P = (np.min(np.array(ptimes))) * numJob / numMch
            lb = round((P - KIM_TAU - KIM_RHO) / 2)
            ub = round((P - KIM_TAU + KIM_RHO) / 2)
            # lb = round(job_list[j].get_ptimes(machine_list)['Avg'])
            # ub = round(job_list[j].get_ptimes(machine_list)['Avg'] + SCHUTTEN_TIGHT * ptimes_avg)
            job_list[j].due = random.randint(lb, ub)
            job_list[j].weight = random.randint(1, 10)

    result = Instance(job_list, machine_list, ptimes, setup_matrix)
    result.with_setup = setup
    result.family_setup = family
    result.identical_mch = identical_mch

    return result


# 현재 스케줄 상태를 기준으로 목적함수값을 계산
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
                result += max(job.end-job.due, 0)
    elif objective == 'wT':
        for m in prob.machine_list:
            for job in m.assigned:
                result += job.weight*max(job.end-job.due, 0)
    return result


def get_total_setups(prob: Instance):
    result = 0
    for m in prob.machine_list:
        for bar in m.schedules:
            result += bar.setup
    return result

def get_obj_name(_obj: str):
    if _obj == 'wT':
        objective = 'the total weighted tardiness'
    elif _obj == 'T':
        objective = 'the total tardiness'
    elif _obj == 'C':
        objective = 'the total completion time'
    else:
        objective = 'the makespan'
    return objective

def load_instance_from_json(path: str) -> Instance:
    """
    Instance.to_dict()로 저장된 JSON 파일을 읽어서
    Instance / Job / Machine 객체로 복원
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------------------------
    # 1) Job 객체 생성
    # -------------------------
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

    # -------------------------
    # 2) Machine 객체 생성
    # -------------------------
    machine_list: List[Machine] = []
    for m in data["machines"]:
        mch = Machine(m["ID"])
        mch.available = m["available"]
        mch.priority = m.get("priority", 0)
        mch.ptime = m["ptime"]
        mch.setup = m["setup"]
        machine_list.append(mch)

    mch_dict = {mch.ID: mch for mch in machine_list}

    # -------------------------
    # 3) Machine에 Job 할당 복원
    # -------------------------
    for m in data["machines"]:
        mch = mch_dict[m["ID"]]
        for job_id in m["assigned"]:
            job = job_dict[job_id]
            job.assignedMch = mch
            mch.assigned.append(job)

    # -------------------------
    # 4) Instance 생성
    # -------------------------
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