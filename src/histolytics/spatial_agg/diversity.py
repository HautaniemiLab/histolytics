"""Indices adapted from: https://github.com/pysal/inequality.

BSD 3-Clause License

Copyright (c) 2018, pysal-inequality developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Sequence

import numpy as np

__all__ = [
    "simpson_index",
    "shannon_index",
    "gini_index",
    "theil_index",
    "theil_between_group",
    "theil_within_group",
    "DIVERSITY_LOOKUP",
    "GROUP_DIVERSITY_LOOKUP",
]

SMALL = np.finfo("float").tiny


def simpson_index(counts: Sequence) -> float:
    """Compute the Simpson diversity index on a count vector.

    Note:
        Simpson diversity index is a quantitative measure that reflects how many
        different types (such as species) there are in a dataset (a community). It
        is a probability measure, when it is low, the greater the probability that
        two randomly selected individuals will be the same species.
        - [A. Wilson, N. Gownaris](https://bio.libretexts.org/Courses/Gettysburg_College/01%3A_Ecology_for_All/22%3A_Biodiversity/22.02%3A_Diversity_Indices)


    **Simpson index:**
    $$
    D = 1 - \\sum_{i=1}^n \\left(\\frac{n_i}{N}\\right)^2
    $$

    where $n_i$ is the count of species $i$ and $N$ is the total count of species.

    Parameters:
        counts (Sequence):
            A count vector/list of shape (C, ).

    Returns:
        float:
            The computed Simpson diversity index.
    """
    N = np.sum(counts) + SMALL
    return 1 - np.sum([(n / N) ** 2 for n in counts if n != 0])


def shannon_index(counts: Sequence) -> float:
    """Compute the Shannon Weiner index/entropy on a count vector.

    Note:
        "*The Shannon index is related to the concept of uncertainty. If for example,
        a community has very low diversity, we can be fairly certain of the identity of
        an organism we might choose by random (high certainty or low uncertainty). If a
        community is highly diverse and we choose an organism by random, we have a
        greater uncertainty of which species we will choose (low certainty or high
        uncertainty).*"
        - [A. Wilson, N. Gownaris](https://bio.libretexts.org/Courses/Gettysburg_College/01%3A_Ecology_for_All/22%3A_Biodiversity/22.02%3A_Diversity_Indices)

    **Shannon index:**
    $$
    H^{\\prime} = -\\sum_{i=1}^n p_i \\ln(p_i)
    $$

    where $p_i$ is the proportion of species $i$ and $n$ is the total count of species.

    Parameters:
        counts (Sequence):
            A count vector/list of shape (C, ).

    Returns:
        float:
            The computed Shannon diversity index.
    """
    N = np.sum(counts) + SMALL
    probs = [float(n) / N for n in counts]

    entropy = -np.sum([p * np.log(p) for p in probs if p != 0])

    if entropy == 0:
        return 0.0

    return entropy


def gini_index(x: Sequence) -> float:
    """Compute the gini coefficient of inequality for species.

    Note:
        This is based on
        http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm.

    **Gini-index:**
    $$
    G = \\frac{\\sum_{i=1}^n (2i - n - 1)x_i} {n \\sum_{i=1}^n x_i}
    $$

    where $x_i$ is the count of species $i$ and $n$ is the total count of species.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )

    Raises:
        ValueError:
            If there are negative input values.

    Returns:
        float:
            The computed Gini coefficient.
    """
    if np.min(x) < 0:
        raise ValueError("Input values need to be positive for Gini coeff")

    n = len(x)
    s = np.sum(x)
    nx = n * s + SMALL

    rx = (2.0 * np.arange(1, n + 1) * x[np.argsort(x)]).sum()
    return (rx - nx - s) / nx


def theil_index(x: Sequence) -> float:
    """Compute the Theil index of inequality for species.

    **Theil-index:**
    $$
    T = \\sum_{i=1}^n \\left(
        \\frac{y_i}{\\sum_{i=1}^n y_i} \\ln \\left[
            N \\frac{y_i} {\\sum_{i=1}^n y_i}
        \\right]\\right)
    $$

    where $y_i$ is the count of species $i$ and $N$ is the total count of species.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )

    Returns:
        float:
            The computed Theil index.
    """
    n = len(x)
    x = x + SMALL * (x == 0)  # can't have 0 values
    xt = np.sum(x, axis=0) + SMALL
    s = x / (xt * 1.0)
    lns = np.log(n * s)
    slns = s * lns
    t = np.sum(slns)

    return t


def theil_between_group(x: Sequence, partition: Sequence) -> float:
    """Compute the between group Theil index.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )
        partition (Sequence):
            The groups for each x value. Shape (n, ).

    Returns:
        float:
            The computed between group Theil index.
    """
    groups = np.unique(partition)
    x_total = x.sum(0) + SMALL

    # group totals
    g_total = np.array([x[partition == gid].sum(axis=0) for gid in groups])

    if x_total.size == 1:  # y is 1-d
        sg = g_total / (x_total * 1.0)
        sg.shape = (sg.size, 1)
    else:
        sg = np.dot(g_total, np.diag(1.0 / x_total))

    ng = np.array([np.sum(partition == gid) for gid in groups])
    ng.shape = (ng.size,)  # ensure ng is 1-d
    n = x.shape[0]

    # between group inequality
    sg = sg + (sg == 0)  # handle case when a partition has 0 for sum
    bg = np.multiply(sg, np.log(np.dot(np.diag(n * 1.0 / ng), sg))).sum(axis=0)

    return float(bg)


def theil_within_group(x: Sequence, partition: Sequence) -> float:
    """Compute the within group Theil index.

    Parameters:
        x (Sequence):
            The input value-vector. Shape (n, )
        partition (Sequence):
            The groups for each x value. Shape (n, ).

    Returns:
        float:
            The computed within group Theil index.
    """
    theil = theil_index(x)
    theil_bg = theil_between_group(x, partition)

    return float(theil - theil_bg)


DIVERSITY_LOOKUP = {
    "simpson_index": simpson_index,
    "shannon_index": shannon_index,
    "gini_index": gini_index,
    "theil_index": theil_index,
}


GROUP_DIVERSITY_LOOKUP = {
    "theil_between_group": theil_between_group,
    "theil_within_group": theil_within_group,
}
