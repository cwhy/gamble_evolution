import numpy as np
from numpy import random
from collections import Counter
import time


n_options = 5
options = [1, 3, 5, 10, 20]
area = [12, 6, 4, 2, 1]
values = [2, 4, 6, 12, 24]


i_lookup_list = []
lookup_list = []
for num in range(sum(area)):
    cumsum_ = 0
    for i, (o, a) in enumerate(zip(options, area)):
        cumsum_ += a
        if num < cumsum_:
            i_lookup_list.append(i)
            lookup_list.append(o)
            break

lookup_array = np.array(lookup_list)
i_lookup_array = np.array(i_lookup_list)





init_val = 1000
# alpha = 1
alpha = 1 / 10
n_agents = 1000
n_keep = int(round(n_agents * 0.1))
n_kill = int(round(n_agents * 0.5))
n_cross = n_agents - n_keep - n_kill

def cross(a, b):
    s1, n1, o1 = a
    s2, n2, o2 = b
    return ((s1 + s2) / 2, (n1 + n2) / 2, (o1 + o2)/2)

def mutate(a):
    s1, n1, o1 = a
    s2 = s1 + np.random.rand() * 0.1 - 0.05
    if not 1 < s2 < 3:
        s2 = 1 + 2 * np.random.rand()
    n2 = n1 + np.random.rand() * 0.1 - 0.05
    if not 0 < n2 < 1:
        n2 = np.random.rand()

    o2 = o1 + np.random.dirichlet(np.ones(n_options) * alpha) * 0.1
    o2 /= o2.sum()
    return (s2, n2, o2)

def evolve(ags, u):
    ids = np.argsort(-u)
    keep_ids = ids[:n_keep]
    new_ags_list = ags[keep_ids].tolist()
    cross_ids = ids[:-n_kill]
    for _ in range(n_agents - n_keep):
        a, b = np.random.choice(cross_ids, 2)
        cross_ag = cross(ags[a], ags[b])
        new_ags_list.append(mutate(cross_ag))
    return np.array(new_ags_list, dtype=object)


agents = np.array([
        (1 + 2 * np.random.rand(), np.random.rand(), np.random.dirichlet(np.ones(n_options) * alpha))
        for _ in range(n_agents)
], dtype=object)
for t in range(100):
    utils = []
    rolls = random.randint(0, 24, 1000000)
    roll_names = lookup_array[rolls]
    roll_ids = i_lookup_array[rolls]
    for agent in agents:
        roll_n_ = 0
        satisfy_ratio, next_ratio, odds = agent
        totals = []
        for _ in range(10):
            total_ = init_val
            for _ in range(10):
                hold_ = int(round(total_ * next_ratio))
                total_ -= hold_
                out_thres = satisfy_ratio * hold_
                n_rounds = 0
                while hold_ < out_thres:
                    # odds_n = np.random.dirichlet(odds * 1000)
                    odds_n = odds
                    bets = (odds_n * hold_).astype(int)
                    hold_ -= bets.sum()
                    old_hold = hold_
                    try:
                        roll_id = roll_ids[roll_n_]
                    except IndexError:
                        print("err")
                        roll_id = i_lookup_array[random.randint(0, 24)]
                    hold_ += values[roll_id] * bets[roll_id]
                    roll_n_ += 1
                    n_rounds += 1
                    if hold_ <= 10 or n_rounds > 1000:
                        break
                total_ += hold_
            totals.append(total_)
        # print("totals:", [int(round(t)) for t in totals])
        util = sum(totals) - init_val * len(totals)
        utils.append(util)
    print(np.array(utils).max(), np.median(utils), np.array(utils).min(), np.mean(utils))
    if np.median(utils).mean() > -1000:
        print(agents[np.random.randint(10)])
    agents = evolve(agents, np.array(utils))
