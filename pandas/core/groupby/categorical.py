
from typing import Optional, Tuple
import numpy as np
from pandas.core.algorithms import unique1d
from pandas.core.arrays.categorical import Categorical, CategoricalDtype, recode_for_categories
from pandas.core.indexes.api import CategoricalIndex

def recode_for_groupby(c, sort, observed):
    '\n    Code the categories to ensure we can groupby for categoricals.\n\n    If observed=True, we return a new Categorical with the observed\n    categories only.\n\n    If sort=False, return a copy of self, coded with categories as\n    returned by .unique(), followed by any categories not appearing in\n    the data. If sort=True, return self.\n\n    This method is needed solely to ensure the categorical index of the\n    GroupBy result has categories in the order of appearance in the data\n    (GH-8868).\n\n    Parameters\n    ----------\n    c : Categorical\n    sort : boolean\n        The value of the sort parameter groupby was called with.\n    observed : boolean\n        Account only for the observed values\n\n    Returns\n    -------\n    New Categorical\n        If sort=False, the new categories are set to the order of\n        appearance in codes (unless ordered=True, in which case the\n        original order is preserved), followed by any unrepresented\n        categories in the original order.\n    Categorical or None\n        If we are observed, return the original categorical, otherwise None\n    '
    if observed:
        unique_codes = unique1d(c.codes)
        take_codes = unique_codes[(unique_codes != (- 1))]
        if c.ordered:
            take_codes = np.sort(take_codes)
        categories = c.categories.take(take_codes)
        codes = recode_for_categories(c.codes, c.categories, categories)
        dtype = CategoricalDtype(categories, ordered=c.ordered)
        return (Categorical(codes, dtype=dtype, fastpath=True), c)
    if sort:
        return (c, None)
    cat = c.unique()
    cat = cat.add_categories(c.categories[(~ c.categories.isin(cat.categories))])
    return (c.reorder_categories(cat.categories), None)

def recode_from_groupby(c, sort, ci):
    '\n    Reverse the codes_to_groupby to account for sort / observed.\n\n    Parameters\n    ----------\n    c : Categorical\n    sort : boolean\n        The value of the sort parameter groupby was called with.\n    ci : CategoricalIndex\n        The codes / categories to recode\n\n    Returns\n    -------\n    CategoricalIndex\n    '
    if sort:
        return ci.set_categories(c.categories)
    new_cats = c.categories[(~ c.categories.isin(ci.categories))]
    return ci.add_categories(new_cats)
