REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .updet_agent import UPDeT
REGISTRY['updet'] = UPDeT

from .trans import Trans
REGISTRY['trans'] = Trans

from .transformer_agg_agent import TransformerAggregationAgent
REGISTRY['transformer_aggregation'] = TransformerAggregationAgent