import os
import json
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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
        print(f"Cleaner failed: {e}")
        return None


def extract_json(raw_content):
    """Robust JSON extraction pipeline."""
    if not raw_content: return None

    # 1. Try fast substring extraction
    try:
        start = raw_content.find('{')
        end = raw_content.rfind('}')
        if start != -1 and end != -1:
            return json.loads(raw_content[start: end + 1])
    except:
        pass

    # 2. Try LLM Cleanup
    cleaned_text = clean_with_llm(raw_content)
    try:
        return json.loads(cleaned_text)
    except:
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
            print(f"❌ {self.role} Error: {e}")
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
                print(f"Moderator Round {round_num}...")
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=conversation,
                    response_format={"type": "json_object"},
                    temperature=0.7
                )

                content = extract_json(response.choices[0].message.content)
                if not content:
                    print("Moderator output empty/invalid.")
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
                print(f"Debate Loop Error: {e}")
                break

        # Retrieve full scenario objects for the winners
        final_scenarios_full = []
        for s_id in final_selection_ids:
            if s_id in master_scenarios:
                full_obj = master_scenarios[s_id]
                full_obj['id_key'] = s_id
                final_scenarios_full.append(full_obj)

        return {
            "debate_log": debate_log,
            "final_scenarios": final_scenarios_full
        }


def run_foresight_simulation(focal_question, model_map=None):
    if model_map is None: model_map = {}

    print(f"🚀 Starting: {focal_question}")
    agent_results = {}

    # Use max_workers=6 for paid tier (Restore to 3 if on free tier)
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_role = {}
        for role in ROLES:
            # Get model for this role, or default
            m_id = model_map.get(role, DEFAULT_AGENT_MODEL)
            future_to_role[executor.submit(Agent(role, m_id).generate_pestle_and_scenarios, focal_question)] = role

        for future in as_completed(future_to_role):
            role = future_to_role[future]
            try:
                res = future.result()
                if res:
                    agent_results[role] = res
                    print(f"✅ {role} Finished")
            except Exception as e:
                print(f"❌ {role} Failed: {e}")

    # Run Moderator
    mod_model = model_map.get("Moderator", DEFAULT_MODERATOR_MODEL)
    print("🧠 Moderator debating...")
    moderator_report = Moderator(mod_model).debate_and_select(focal_question, agent_results)

    return {"agent_data": agent_results, "moderator_report": moderator_report}