"""The TransFuser SIL core: the policy-specific half of ``PolicyAgentCore``."""

from lead.evaluation.agents.transfuser.transfuser_control import TransfuserControlMixin
from lead.evaluation.inference.agent_core import PolicyAgentCore


class TransfuserCore(TransfuserControlMixin, PolicyAgentCore):
    """TransFuser behaviour on top of the CARLA-free agent core."""
