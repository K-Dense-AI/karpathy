import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from .tools import delegate_task
from .utils import load_instructions

load_dotenv()

# Configuration
MODEL = os.getenv("AGENT_MODEL")

# Main agent
main_agent = LlmAgent(
    name="MainAgent",
    model=LiteLlm(model=MODEL),
    description="The main agent that makes sure the user's machine learning requests are successfully fulfilled",
    instruction=load_instructions("main_agent"),
    tools=[delegate_task],
    output_key="final_output",
)