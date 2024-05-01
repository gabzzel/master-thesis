import numpy as np


def format_number(number, digits=1):
    if number >= 1000000000:
        return str(round(number / 1000000000, digits)) + "B"

    if number >= 1000000:
        return str(round(number / 1000000, digits)) + "M"

    if number >= 1000:
        return str(round(number / 1000, digits)) + "K"

    return str(number)


def get_stats(a: np.array, name: str = "", print_results=True, round_digits=3, return_results=False):
    if (a is None or len(a) == 0) and return_results:
        print(f"No stats for {name}.")
        return 0, 0, 0, 0, 0

    try:
        _max = round(np.max(a), round_digits) if round_digits >= 0 else np.max(a)
        _min = round(np.min(a), round_digits) if round_digits >= 0 else np.min(a)
        avg = round(np.average(a), round_digits) if round_digits >= 0 else np.average(a)
        med = round(float(np.median(a=a)), round_digits) if round_digits >= 0 else np.median(a)
        std = round(float(np.std(a=a)), round_digits) if round_digits >= 0 else np.std(a)
        count = len(a)
    except Exception as e:
        _max = _min = avg = med = std = count = 0
        print("Getting stats failed due to " + str(e))

    if print_results:
        print(f"{name} stats: Count={count}, Max={_max}, Min={_min}, Avg/Mean={avg}, Med={med}, Std={std}")
    if return_results:
        return _max, _min, avg, med, std
