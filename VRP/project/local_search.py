# local_search.py (수정판)
import copy
import random
import math
import module

def _get_depot_loc(instance):
    """instance에서 depot 위치를 안전하게 추출 (fallback (0,0))."""
    # 여러 possible names: instance.depot.loc, instance.depot_loc, instance.depot_location 등을 시도
    if hasattr(instance, "depot"):
        dep = getattr(instance, "depot")
        if dep is None:
            return (0.0, 0.0)
        # depot may be an object with .loc or .location
        if hasattr(dep, "loc"):
            return tuple(dep.loc)
        if hasattr(dep, "location"):
            return tuple(dep.location)
    if hasattr(instance, "depot_loc"):
        return tuple(getattr(instance, "depot_loc"))
    if hasattr(instance, "depot_location"):
        return tuple(getattr(instance, "depot_location"))
    # fallback default
    return (0.0, 0.0)

def route_metrics_from_sequence(start_loc, start_available, speed, capacity, cust_seq):
    """
    주어진 차량의 시작 위치(start_loc), 시작 가능 시간(start_available),
    speed, capacity, 그리고 고객 seq(list of Customer objs)에 대해
    route total tardiness and distance, and whether capacity feasible.
    반환: (tardiness, distance, feasible)
    """
    prev_loc = tuple(start_loc)
    available = float(start_available or 0.0)
    cum_load = 0.0
    total_tard = 0.0
    total_dist = 0.0

    for c in cust_seq:
        travel_km = module.get_dist(prev_loc, tuple(c.loc))
        travel_h = travel_km / max(1e-9, float(speed or 1.0))
        start = max(float(c.tw[0]), available + travel_h)
        end = start + float(c.serv_time or 0.0)
        tard = max(0.0, end - float(c.tw[1]))

        total_tard += tard
        total_dist += travel_km
        available = end
        prev_loc = tuple(c.loc)
        cum_load += float(c.weight or 0.0)
        # capacity check
        if cum_load > float(capacity or 0.0) + 1e-9:
            return (1e9, 1e9, False)  # infeasible indicator

    return (total_tard, total_dist, True)

def evaluate_solution(instance):
    """
    instance.vehicles에 할당된 .schedules 가 이미 세팅되어 있을 때
    전체 tardiness, total_distance, and unserved count을 계산
    (단, schedules의 고객 순서대로 시뮬레이션)
    """
    total_tard = 0.0
    total_dist = 0.0
    unserved = []

    # compute a reliable depot start location fallback
    depot_loc = _get_depot_loc(instance)

    for v in instance.vehicles:
        # use stored initial values if present; otherwise fallback to depot/0.0
        start_loc = getattr(v, "now_loc_init", depot_loc)
        start_avail = getattr(v, "available_init", 0.0)

        seq = v.schedules
        t, d, feasible = route_metrics_from_sequence(start_loc, start_avail, v.speed, v.capacity, seq)
        total_tard += t
        total_dist += d
        # if route infeasible (capacity), t will be large from route_metrics

    # any customers with .complete != True are considered unserved
    for c in instance.customers:
        if not getattr(c, "complete", False):
            unserved.append(c.ID)
    return total_tard, total_dist, len(unserved)

def deep_copy_solution(instance):
    """instance는 참조형이므로 solution-level 복사본을 만들 때 사용"""
    return copy.deepcopy(instance)

def flatten_schedules(instance):
    """vehicle별 schedules -> global list of (vhc_index, customer)"""
    flat = []
    for vi, v in enumerate(instance.vehicles):
        for c in v.schedules:
            flat.append((vi, c))
    return flat

def _ensure_vehicle_initials(instance):
    """
    로컬 서치 시작 전에 모든 차량에 대해 now_loc_init, available_init을 설정.
    - 가능한 경우 instance의 depot 위치를 사용.
    - 기존에 이미 올바르게 설정돼 있다면 덮어쓰지 않음.
    """
    depot_loc = _get_depot_loc(instance)
    for v in instance.vehicles:
        # only set if not present or seems invalid (tuple of last assigned loc)
        if not hasattr(v, "now_loc_init") or v.now_loc_init is None:
            # set to depot
            v.now_loc_init = tuple(depot_loc)
        if not hasattr(v, "available_init") or v.available_init is None:
            v.available_init = 0.0

def local_improve_by_relocate(instance, max_iters=200):
    """
    Greedy best-improvement relocate: 다른 차량에 고객을 옮기거나
    같은 차량 안에서 위치를 바꿔서 tardiness를 줄이면 적용.
    """
    # ensure we have correct start locs for evaluating routes
    _ensure_vehicle_initials(instance)

    improved = True
    it = 0
    while improved and it < max_iters:
        it += 1
        improved = False
        best_gain = 0.0
        best_move = None  # (from_vi, from_pos, to_vi, to_pos, seq_from, seq_to, gain)

        # current total tardiness
        cur_tard = 0.0
        for v in instance.vehicles:
            t, d, feasible = route_metrics_from_sequence(v.now_loc_init, v.available_init, v.speed, v.capacity, v.schedules)
            cur_tard += t

        V = len(instance.vehicles)
        # try all relocates (including intra-route reposition)
        for vi in range(V):
            v_from = instance.vehicles[vi]
            n_from = len(v_from.schedules)
            for pos in range(n_from):
                c = v_from.schedules[pos]
                # try move to every vehicle and every insertion position
                for vj in range(V):
                    v_to = instance.vehicles[vj]
                    # range of insertion positions for v_to
                    for ins_pos in range(len(v_to.schedules) + (0 if vi != vj else 1)):
                        # skip no-op cases for same-vehicle insertion
                        if vi == vj and (ins_pos == pos or ins_pos == pos + 1):
                            continue
                        # shallow copies of sequences (list of customer objs)
                        seq_from = v_from.schedules.copy()
                        seq_to = v_to.schedules.copy()
                        # remove from seq_from
                        seq_from.pop(pos)
                        # insert into seq_to
                        seq_to.insert(ins_pos, c)
                        # evaluate both routes with correct starts
                        t_from, d_from, f_from = route_metrics_from_sequence(v_from.now_loc_init, v_from.available_init, v_from.speed, v_from.capacity, seq_from)
                        t_to, d_to, f_to = route_metrics_from_sequence(v_to.now_loc_init, v_to.available_init, v_to.speed, v_to.capacity, seq_to)
                        if not (f_from and f_to):
                            continue
                        # compute new total tardiness by summing t for all vehicles with modified ones replaced
                        new_total = 0.0
                        for vk in range(V):
                            if vk == vi:
                                new_total += t_from
                            elif vk == vj:
                                new_total += t_to
                            else:
                                tt, dd, ff = route_metrics_from_sequence(instance.vehicles[vk].now_loc_init, instance.vehicles[vk].available_init, instance.vehicles[vk].speed, instance.vehicles[vk].capacity, instance.vehicles[vk].schedules)
                                new_total += tt
                        gain = cur_tard - new_total
                        # only accept positive gain (reduce tardiness)
                        if gain > best_gain + 1e-9:
                            best_gain = gain
                            best_move = (vi, pos, vj, ins_pos, seq_from, seq_to, best_gain)

        if best_move:
            vi, pos, vj, ins_pos, seq_from, seq_to, gain = best_move
            # commit move (modify schedules)
            instance.vehicles[vi].schedules = seq_from
            instance.vehicles[vj].schedules = seq_to
            improved = True

    return instance

def intra_route_swap_improve(instance, max_iters=200):
    """각 차량 내에서 두 고객의 위치를 교환해 tardiness 개선하면 적용"""
    _ensure_vehicle_initials(instance)

    improved = True
    it = 0
    while improved and it < max_iters:
        it += 1
        improved = False
        best_gain = 0.0
        best_move = None
        # current total tardiness
        cur_tard = 0.0
        for v in instance.vehicles:
            t, d, f = route_metrics_from_sequence(v.now_loc_init, v.available_init, v.speed, v.capacity, v.schedules)
            cur_tard += t

        for vi, v in enumerate(instance.vehicles):
            n = len(v.schedules)
            for i in range(n):
                for j in range(i+1, n):
                    seq = v.schedules.copy()
                    seq[i], seq[j] = seq[j], seq[i]
                    t_new, d_new, f_new = route_metrics_from_sequence(v.now_loc_init, v.available_init, v.speed, v.capacity, seq)
                    if not f_new:
                        continue
                    # compute new total tardiness where vi route replaced by t_new
                    new_total = t_new
                    for vk, vv in enumerate(instance.vehicles):
                        if vk == vi:
                            continue
                        tt, dd, ff = route_metrics_from_sequence(vv.now_loc_init, vv.available_init, vv.speed, vv.capacity, vv.schedules)
                        new_total += tt
                    gain = cur_tard - new_total
                    if gain > best_gain + 1e-9:
                        best_gain = gain
                        best_move = (vi, i, j, seq, best_gain)
        if best_move:
            vi, i, j, seq, gain = best_move
            instance.vehicles[vi].schedules = seq
            improved = True
    return instance

def local_search_improve(instance, max_iters=500):
    """
    간단한 로컬 서치: relocate greedy + intra-route swap 반복
    항상 tardiness(총 지연)를 줄이는 방향으로만 개선을 적용한다.
    """
    # ensure initial start info exists
    _ensure_vehicle_initials(instance)

    improved = True
    it = 0
    while improved and it < max_iters:
        it += 1
        prev_tard, _, _ = evaluate_solution(instance)
        instance = local_improve_by_relocate(instance, max_iters=50)
        instance = intra_route_swap_improve(instance, max_iters=50)
        new_tard, _, _ = evaluate_solution(instance)
        # only continue if new tardiness improved (reduced)
        if new_tard < prev_tard - 1e-9:
            improved = True
        else:
            improved = False
    return instance

def simulated_annealing_on_solution(instance, initial_temp=1.0, final_temp=1e-3, alpha=0.95, iter_per_temp=50):
    """
    간단한 SA: 랜덤한 relocate 또는 swap을 적용하고, tardiness 기준으로 accept/reject.
    instance는 deepcopy하여 수정 후 반환
    """
    _ensure_vehicle_initials(instance)

    cur_inst = deep_copy_solution(instance)
    best_inst = deep_copy_solution(instance)
    best_tard, _, _ = evaluate_solution(best_inst)
    cur_tard = best_tard

    T = initial_temp
    while T > final_temp:
        for _ in range(iter_per_temp):
            cand = deep_copy_solution(cur_inst)
            # 랜덤 move: relocate or swap or intra-route swap
            move_type = random.choice(['relocate','swap','intra_swap'])
            V = len(cand.vehicles)
            if move_type == 'relocate':
                vi = random.randrange(V)
                v_from = cand.vehicles[vi]
                if not v_from.schedules:
                    continue
                pos = random.randrange(len(v_from.schedules))
                c = v_from.schedules.pop(pos)
                vj = random.randrange(V)
                ins_pos = random.randrange(len(cand.vehicles[vj].schedules)+1)
                cand.vehicles[vj].schedules.insert(ins_pos, c)
            elif move_type == 'swap':
                vi = random.randrange(V); vj = random.randrange(V)
                if vi==vj:
                    continue
                if not cand.vehicles[vi].schedules or not cand.vehicles[vj].schedules:
                    continue
                pi = random.randrange(len(cand.vehicles[vi].schedules))
                pj = random.randrange(len(cand.vehicles[vj].schedules))
                cand.vehicles[vi].schedules[pi], cand.vehicles[vj].schedules[pj] = cand.vehicles[vj].schedules[pj], cand.vehicles[vi].schedules[pi]
            else:
                vi = random.randrange(V)
                v = cand.vehicles[vi]
                if len(v.schedules) < 2:
                    continue
                i = random.randrange(len(v.schedules))
                j = random.randrange(len(v.schedules))
                if i==j:
                    continue
                v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

            cand_tard, _, _ = evaluate_solution(cand)
            delta = cand_tard - cur_tard
            # accept criterion: lower tardiness OR probabilistic acceptance
            if delta < 0 or random.random() < math.exp(-delta / max(1e-9, T)):
                cur_inst = cand
                cur_tard = cand_tard
                if cur_tard < best_tard:
                    best_inst = deep_copy_solution(cur_inst)
                    best_tard = cur_tard
        T *= alpha

    return best_inst
