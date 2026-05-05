import copy
import random


def create_random_instance_from_template(template_inst, n_jobs, seed=0,
                                         p_low=1, p_high=20,
                                         s_low=0, s_high=10,
                                         w_low=1, w_high=10,
                                         due_tightness=0.6):
    rnd = random.Random(seed)

    if template_inst.numMch < 2:
        raise ValueError("템플릿 인스턴스의 machine 수가 2보다 작습니다.")

    inst = copy.deepcopy(template_inst)

    # machine 2대 고정
    inst.numMch = 2
    inst.machine_list = copy.deepcopy(template_inst.machine_list[:2])

    # processing time 생성
    ptime = [[rnd.randint(p_low, p_high) for _ in range(n_jobs)] for _ in range(2)]

    # total work 기준 due date 생성
    total_work = sum(min(ptime[0][j], ptime[1][j]) for j in range(n_jobs))
    due_low = max(1, int(total_work * due_tightness * 0.4))
    due_high = max(due_low + 1, int(total_work * due_tightness * 1.0))

    # job 생성
    if len(template_inst.job_list) == 0:
        raise ValueError("템플릿 인스턴스에 job이 없습니다.")

    job_proto = template_inst.job_list[0]
    inst.job_list = []

    for j in range(n_jobs):
        job = copy.deepcopy(job_proto)
        job.ID = j
        job.weight = rnd.randint(w_low, w_high)
        job.due = rnd.randint(due_low, due_high)
        job.start = 0
        job.end = 0
        inst.job_list.append(job)

    # setup 생성
    setup = []
    for m in range(2):
        sm = []
        for i in range(n_jobs):
            row = []
            for j in range(n_jobs):
                row.append(0 if i == j else rnd.randint(s_low, s_high))
            sm.append(row)
        setup.append(sm)

    inst.numJob = n_jobs
    inst.ptime = ptime
    inst.setup = setup

    if not hasattr(inst, "objective") or inst.objective is None:
        inst.objective = "wT"

    # 중요: machine 내부 데이터도 새 길이에 맞게 갱신
    for k, mch in enumerate(inst.machine_list):
        mch.ID = k

        if hasattr(mch, "assigned"):
            mch.assigned = []

        if hasattr(mch, "available"):
            mch.available = 0

        # machine.process()가 참조하는 내부 배열 갱신
        if hasattr(mch, "ptime"):
            mch.ptime = inst.ptime[k]

        if hasattr(mch, "setup"):
            mch.setup = inst.setup[k]

    return inst