import os
import json
import re
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


# Input validation constants
MAX_QUESTION_LENGTH = 1000
MIN_QUESTION_LENGTH = 10
ALLOWED_MODELS = {
    "deepseek-ai/DeepSeek-V3",
    "meta-llama/Llama-3.3-70B-Instruct",
    "moonshotai/Kimi-K2-Thinking",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
}

api_key = os.environ.get("DEEPINFRA_API_KEY")
if not api_key:
    raise ValueError("DEEPINFRA_API_KEY not found in .env file.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepinfra.com/v1/openai",
)

# Defaults
DEFAULT_AGENT_MODEL = "deepseek-ai/DeepSeek-V3"
DEFAULT_MODERATOR_MODEL = "deepseek-ai/DeepSeek-V3"
CLEANER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

ROLES = [
    "Economist", "Sociologist", "Technologist",
    "Ecologist", "Political Scientist", "Legal Ethicist"
]


def clean_with_llm(raw_text):
    """Fallback: Uses Llama 3.3 to fix broken JSON."""
    try:
        response = client.chat.completions.create(
            model=CLEANER_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a JSON repair engine. Extract the JSON object. Fix any syntax errors (missing quotes, commas). Output ONLY the JSON string."},
                {"role": "user", "content": f"Fix this JSON:\n\n{raw_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM JSON cleaner failed: {e}")
        return None


def extract_json(raw_content):
    """Robust JSON extraction pipeline."""
    if not raw_content:
        return None

    # 1. Try fast substring extraction
    try:
        start = raw_content.find('{')
        end = raw_content.rfind('}')
        if start != -1 and end != -1:
            return json.loads(raw_content[start: end + 1])
    except json.JSONDecodeError as e:
        logger.debug(f"Fast JSON extraction failed: {e}")

    # 2. Try LLM Cleanup
    cleaned_text = clean_with_llm(raw_content)
    if cleaned_text is None:
        logger.warning("LLM cleanup returned None")
        return None

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed after LLM cleanup: {e}")
        return None


class Agent:
    def __init__(self, role, model_id):
        self.role = role
        self.model_id = model_id

    def generate_pestle_and_scenarios(self, focal_question):
        # We allow "Chain of Thought" but enforce JSON at the end
        system_prompt = f"You are an expert {self.role}. Deep reasoning is encouraged, but your final output must be a valid JSON object."
        user_prompt = f"""
        Focal Question: "{focal_question}"

        TASK:
        1. Analyze 5 driving forces per PESTLE factor.
        2. Select 3 critical forces (Variables).
        3. Create 8 Scenarios. For each, define the 'Force States' (e.g., 'Economy: Boom' vs 'Economy: Bust').

        Output strictly JSON structure:
        {{
            "pestle": {{ "Political": [], ... }},
            "selected_forces": ["Force A", "Force B", "Force C"],
            "scenarios": [
                {{
                    "title": "Scenario Title", 
                    "force_states": {{ "Force A": "State X", "Force B": "State Y", "Force C": "State Z" }},
                    "description": "Narrative description...",
                    "signposts": ["Indicator 1", "Indicator 2"],
                    "black_swan": "Low probability event"
                }}
            ]
        }}
        """
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.75
            )
            return extract_json(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Agent {self.role} failed to generate scenarios: {e}")
            return None


class Moderator:
    def __init__(self, model_id):
        self.model_id = model_id

    def debate_and_select(self, focal_question, agent_outputs):
        master_scenarios = {}
        scenarios_text_list = []

        # Flatten all scenarios
        for role, data in agent_outputs.items():
            if data and 'scenarios' in data:
                for s in data['scenarios']:
                    s_id = f"{role}: {s['title']}"
                    master_scenarios[s_id] = s
                    states_str = " | ".join([f"{k}: {v}" for k, v in s.get('force_states', {}).items()])
                    scenarios_text_list.append(
                        f"ID: {s_id}\n   Forces: [{states_str}]\n   Desc: {s['description'][:300]}...")

        # Shuffle for fairness
        random.shuffle(scenarios_text_list)
        scenarios_block = "\n\n".join(scenarios_text_list)

        # RESTORED: The "Devil's Advocate" System Prompt
        conversation = [
            {"role": "system", "content": f"""
            You are a Critical Foresight Moderator. 
            Your goal is to select the 4 scenarios that offer the MAXIMUM STRUCTURAL DIVERGENCE.

            RULES:
            1. You must select scenarios from at least 3 DIFFERENT roles.
            2. You must actively critique your own selection in each round.
            3. You must SWAP scenarios if they are too similar.
            """},
            {"role": "user", "content": f"""
            Focal Question: "{focal_question}"

            CANDIDATE SCENARIOS:
            {scenarios_block}

            ROUND 1: INITIAL SELECTION
            Select 4 scenarios that seem most distinct.
            Output JSON: {{ 
                "rationale": "I selected these because...", 
                "selected_ids": ["Role: Title 1", "Role: Title 2", ...] 
            }}
            """}
        ]

        debate_log = []
        final_selection_ids = []

        # Run 3 Rounds of Debate
        for round_num in range(1, 4):
            try:
                logger.info(f"Moderator Round {round_num}...")
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=conversation,
                    response_format={"type": "json_object"},
                    temperature=0.7
                )

                content = extract_json(response.choices[0].message.content)
                if not content:
                    logger.warning("Moderator output empty/invalid.")
                    break

                current_ids = content.get("selected_ids", [])
                rationale = content.get("rationale", "No rationale provided.")

                debate_log.append(f"Round {round_num}: {rationale}")
                final_selection_ids = current_ids

                # Append assistant response to history
                conversation.append(response.choices[0].message)

                # Inject Critique for the next round
                if round_num < 3:
                    critique_prompt = f"""
                    ROUND {round_num + 1}: CRITIQUE & ITERATE

                    Critique your selection ({current_ids}).
                    - Do they share too many similar assumptions?
                    - Are 3+ distinct roles represented?

                    MANDATORY: Swap at least 1 scenario to increase diversity or address a blind spot.
                    Output JSON with 'rationale' and new 'selected_ids'.
                    """
                    conversation.append({"role": "user", "content": critique_prompt})

            except Exception as e:
                logger.error(f"Moderator debate loop error: {e}")
                break

        # Retrieve full scenario objects for the winners (copy to avoid mutating originals)
        final_scenarios_full = []
        for s_id in final_selection_ids:
            if s_id in master_scenarios:
                full_obj = master_scenarios[s_id].copy()
                full_obj['id_key'] = s_id
                final_scenarios_full.append(full_obj)

        return {
            "debate_log": debate_log,
            "final_scenarios": final_scenarios_full
        }


def validate_input(focal_question, model_map):
    """Validate user inputs before processing."""
    # Validate focal question
    if not focal_question or not isinstance(focal_question, str):
        raise ValidationError("Focal question must be a non-empty string.")

    focal_question = focal_question.strip()

    if len(focal_question) < MIN_QUESTION_LENGTH:
        raise ValidationError(f"Focal question must be at least {MIN_QUESTION_LENGTH} characters.")

    if len(focal_question) > MAX_QUESTION_LENGTH:
        raise ValidationError(f"Focal question must not exceed {MAX_QUESTION_LENGTH} characters.")

    # Validate model selections
    if model_map:
        for role, model_id in model_map.items():
            if model_id not in ALLOWED_MODELS:
                raise ValidationError(f"Invalid model '{model_id}' for role '{role}'. Choose from allowed models.")

    return focal_question


def run_foresight_simulation(focal_question, model_map=None, timeout=300):
    """
    Run the foresight simulation with the given focal question.

    Args:
        focal_question: The question to analyze
        model_map: Optional mapping of roles to model IDs
        timeout: Maximum time in seconds for the simulation (default: 300)

    Raises:
        ValidationError: If inputs are invalid
        TimeoutError: If simulation exceeds timeout
    """
    if model_map is None:
        model_map = {}

    # Validate inputs
    focal_question = validate_input(focal_question, model_map)

    logger.info(f"Starting simulation: {focal_question[:100]}...")
    agent_results = {}

    # Use max_workers=6 for paid tier (Restore to 3 if on free tier)
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_role = {}
        for role in ROLES:
            # Get model for this role, or default
            m_id = model_map.get(role, DEFAULT_AGENT_MODEL)
            future_to_role[executor.submit(Agent(role, m_id).generate_pestle_and_scenarios, focal_question)] = role

        for future in as_completed(future_to_role, timeout=timeout):
            role = future_to_role[future]
            try:
                res = future.result(timeout=timeout)
                if res:
                    agent_results[role] = res
                    logger.info(f"Agent {role} finished successfully")
            except FuturesTimeoutError:
                logger.error(f"Agent {role} timed out")
                raise TimeoutError(f"Simulation timed out while waiting for {role}")
            except Exception as e:
                logger.error(f"Agent {role} failed: {e}")

    # Run Moderator
    mod_model = model_map.get("Moderator", DEFAULT_MODERATOR_MODEL)
    logger.info("Moderator debating scenarios...")
    moderator_report = Moderator(mod_model).debate_and_select(focal_question, agent_results)

    return {"agent_data": agent_results, "moderator_report": moderator_report}