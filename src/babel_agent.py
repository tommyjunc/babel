"""Simple agents for The Babel Machine."""

from mesa import Agent


class BabelAgent(Agent):
    """An agent with a categorical language and a platform convention."""

    def __init__(
        self,
        model,
        language_id,
        platform_id,
        skill_level,
        cooperation_tendency,
    ):
        super().__init__(model)
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
