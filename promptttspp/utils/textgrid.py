#!/usr/bin/python
# Downloaded from https://github.com/feelins/Praat_Scripts/blob/master/00-Python/textgrid.py
##############################################################################
# The MIT License (MIT)
#
# Copyright (c) 2016 Kyler Brown
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################


from collections import namedtuple

# from language_util import _SILENCE

Entry = namedtuple("Entry", ["start", "stop", "name", "tier"])


def read_textgrid(filename, tierName="phones"):
    """
    Reads a TextGrid file into a dictionary object
    each dictionary has the following keys:
    "start"
    "stop"
    "name"
    "tier"

    Points and intervals use the same format,
    but the value for "start" and "stop" are the same
    """
    if isinstance(filename, str):
        with open(filename, "r", encoding="utf-8") as f:
            content = _read(f)
    elif hasattr(filename, "readlines"):
        content = _read(filename)
    else:
        raise TypeError("filename must be a string or a readable buffer")

    interval_lines = [
        i
        for i, line in enumerate(content)
        if line.startswith("intervals [") or line.startswith("points [")
    ]
    tier_lines = []
    tiers = []
    for i, line in enumerate(content):
        if line.startswith("name ="):
            tier_lines.append(i)
            tiers.append(line.split('"')[-2])

    for _, line in enumerate(content):
        if line.startswith("xmax = "):
            time_array = line.split()
            time_array = [
                item for item in filter(lambda x: x.strip() != "", time_array)
            ]
            float(time_array[-1])
            break

    interval_tiers = _find_tiers(interval_lines, tier_lines, tiers)
    assert len(interval_lines) == len(interval_tiers)
    adjust_list = [
        _build_entry(i, content, t)
        for i, t in zip(interval_lines, interval_tiers)
        if t == tierName
    ]
    return adjust_list


def _find_tiers(interval_lines, tier_lines, tiers):
    tier_pairs = zip(tier_lines, tiers)
    tier_pairs = iter(tier_pairs)
    _, cur_tier = next(tier_pairs)
    next_tline, next_tier = next(tier_pairs, (None, None))
    tiers = []
    for il in interval_lines:
        if next_tline is not None and il > next_tline:
            _, cur_tier = next_tline, next_tier
            next_tline, next_tier = next(tier_pairs, (None, None))
        tiers.append(cur_tier)
    return tiers


def _read(f):
    return [x.strip() for x in f.readlines()]


def _build_entry(i, content, tier):
    """
    takes the ith line that begin an interval and returns
    a dictionary of values
    """
    start = _get_float_val(content[i + 1])  # addition is cheap typechecking
    if content[i].startswith("intervals ["):
        offset = 1
    else:
        offset = 0  # for "point" objects
    stop = _get_float_val(content[i + 1 + offset])
    label = _get_str_val(content[i + 2 + offset])
    return Entry(start=start, stop=stop, name=label, tier=tier)


def _get_float_val(string):
    """
    returns the last word in a string as a float
    """
    return float(string.split()[-1])


def _get_str_val(string):
    """
    returns the last item in quotes from a string
    """
    return string.split('"')[-2]
