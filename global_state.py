# global_state.py

import os

# bizim provider
from llm_providers.gemini import GeminiClient

_PROVIDER = os.getenv("PROVIDER", "gemini").lower()
_DEFAULT_MODEL = os.getenv("COMPLETION_MODEL", "gemini-2.5-pro")
_llm_client = None

def get_llm_client():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    if _PROVIDER == "gemini":
        _llm_client = GeminiClient(default_model=_DEFAULT_MODEL)
        return _llm_client
    raise RuntimeError(f"Unknown PROVIDER={_PROVIDER}")

LOG_PATH = ""

START_FLAG = False
FIRST_MAIN = False


EXIT_FLAG = False
INIT_FLAG = False
# WORKFLOW_FLAG = False
# AGENTCREATE_FLAS = False

# AGENT_CREATOR_AGENT_STATE = False
# META_AGENT_LAST_QUERY = ''
# WORKFLOW_CREATOR_AGENT_STATE = False
# WORKFLOW_AGENT_LAST_QUERY = ''

# container_name = 'auto_agent'
# port = 12345
# test_pull_name = 'autoagent_mirror'
# git_clone = True
# local_env = False