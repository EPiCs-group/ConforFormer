from collections import defaultdict
import lmdb

def balance_dict_keys(d: dict[str, list], n: int) -> list[list[str]]:
    # Step 1: Convert to list of (key, weight)
    key_weights = [(k, len(v)) for k, v in d.items()]

    # Step 2: Sort keys by weight descending (greedy step)
    key_weights.sort(key=lambda x: -x[1])

    # Step 3: Initialize groups
    groups = [[] for _ in range(n)]
    group_sums = [0] * n

    # Step 4: Greedily assign each key to the group with the current smallest sum
    for key, weight in key_weights:
        min_index = group_sums.index(min(group_sums))
        groups[min_index].append(key)
        group_sums[min_index] += weight

    return groups


def init_lmdb_env(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=256,
    )
    return env


def merge_lowercase_with_prev(chars):
    result = []
    for ch in chars:
        if ch.islower() and result:
            # append lowercase letter to the last element in result
            result[-1] += ch
        else:
            # start a new element
            result.append(ch)
    return result
