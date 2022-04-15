import platform
import smtplib
from collections import deque
from collections.abc import Set, Mapping
from datetime import datetime
from math import log, exp
from numbers import Number
from sys import getsizeof
from typing import List, Union, Sequence, Any

import psutil
from numpy import ndarray, argsort
from scipy.stats import kendalltau


def send_email(subject=None, body=None):
    subject = subject or 'Running experiment'

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = body or f'Experiment result at {current_time}'

    s_from = 'pymail4exp@gmail.com'
    s_to = 'timoping@gmail.com'

    email_text = f'From: {s_from}\nTo: {s_to}\nSubject: {subject}\n\n{body}'

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.ehlo()
        server.login(s_from, 'pYmail--4--exp')
        server.sendmail(s_from, s_to, email_text)
        server.close()

        print('Email sent!')


def normalize_weights(weights: Sequence[float]) -> Union[List[float], ndarray]:
    """These weights should add up to 1, so that they are a distribution."""
    weight_sum = sum(weights)
    assert weight_sum > 0

    if isinstance(weights, ndarray):
        return weights / weight_sum
    else:
        return [weight / weight_sum for weight in weights]


def calculate_kendall_tau_distance(x: Sequence, y: Sequence):
    assert len(x) == len(y)

    tau_score, _ = kendalltau(argsort(x), argsort(y))
    return round((1 - tau_score) * len(x) * (len(x) - 1) / 4)


def get_computer_info():
    os_info = platform.uname()
    python_version = platform.python_version()
    return f'Python {python_version} on {os_info}'


def display_memory_usage_of_current_process():
    process = psutil.Process()
    memory_gb = process.memory_info()[0] / (1024 ** 3)  # Bytes -> GB
    memory_pct = process.memory_percent()
    print(f'[INFO] Current process uses {memory_gb:.2f} GB ({memory_pct:.2f}%) RAM.')


def is_running_out_of_memory(verbose=False):
    stats = psutil.virtual_memory()
    available_memory_ratio = stats.available / stats.total
    is_running_out = available_memory_ratio < 0.2

    if verbose:
        memory_gb = stats.used / (1024 ** 3)  # Bytes -> GB
        memory_avail_gb = stats.available / (1024 ** 3)  # Bytes -> GB
        if is_running_out:
            print(f'[WARNING] Out of memory! {memory_gb:.2f} GB RAM in usage. Only {memory_avail_gb:.2f} GB RAM available.')
        else:
            print(f'[INFO] {memory_gb:.2f} GB RAM in usages. {memory_avail_gb:.2f} GB RAM available.')

    return is_running_out


def get_ram_gb_size():
    stats = psutil.virtual_memory()
    return round(stats.total / (1024 ** 3))  # Bytes -> GB


def get_object_size(obj_0: Any):
    """
    Recursively iterate to sum size of object & members.

    From https://github.com/mCodingLLC/VideosSampleCode/blob/master/videos/080_python_slots/slots.py
    """
    ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)
    _seen_ids = set()

    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = getsizeof(obj)
        if isinstance(obj, ZERO_DEPTH_BASES):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size

    return inner(obj_0)


def log_plus(log_a, log_b):
    """Compute log(a + b) given log(a) and log(b)."""
    return log_a + log(1 + exp(log_b - log_a))


def log_minus(log_a, log_b):
    """Compute log(a - b) given log(a) and log(b)."""
    return log_a + log(1 - exp(log_b - log_a))


if __name__ == '__main__':
    is_running_out_of_memory(verbose=True)
