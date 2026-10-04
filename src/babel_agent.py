"""Simple agents for The Babel Machine."""

from mesa import Agent

DEFAULT_LLM_SYSTEM_ID = "system_a"


class BabelAgent(Agent):
    """An agent with a categorical language, platform convention, and LLM system."""

    def __init__(
        self,
        model,
        language_id,
        platform_id,
        skill_level,
        cooperation_tendency,
        llm_system_id=DEFAULT_LLM_SYSTEM_ID,
    ):
        if not isinstance(llm_system_id, str) or not llm_system_id:
            raise ValueError("llm_system_id must be a non-empty string")
        super().__init__(model)
        self.llm_system_id = llm_system_id
        self.language_id = int(language_id)
        self.platform_id = int(platform_id)
        self.skill_level = float(skill_level)
        self.cooperation_tendency = float(cooperation_tendency)
        self.communication_history = []
        self.accumulated_contribution = 0.0
        self.successful_communications = 0
        self.failed_communications = 0

    @property
    def platform_language(self):
        """The convention associated with this agent's platform."""
        return self.platform_id
