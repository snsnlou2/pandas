
'\nPublic API for Rolling Window Indexers.\n'
from pandas.core.indexers import check_array_indexer
from pandas.core.window.indexers import BaseIndexer, FixedForwardWindowIndexer, VariableOffsetWindowIndexer
__all__ = ['check_array_indexer', 'BaseIndexer', 'FixedForwardWindowIndexer', 'VariableOffsetWindowIndexer']
