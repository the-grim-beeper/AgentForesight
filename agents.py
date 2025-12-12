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

# Default Fallbacks
DEFAULT_AGENT_MODEL = "deepseek-ai/DeepSeek-V3"
DEFAULT_MODERATOR_MODEL = "deepseek-ai/DeepSeek-V3"
CLEANER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

ROLES = [
    "Economist", "Sociologist", "Technologist",
    "Ecologist", "Political Scientist", "Legal Ethicist"
]


def clean_with_llm(raw_text, error_msg=""):
    try:
        response = client.chat.completions.create(
            model=CLEANER_MODEL,
            messages=[
                {"role": "system", "content": "Extract valid JSON only. Fix syntax errors."},
                {"role": "user", "content": f"Fix and extract JSON:\n\n{raw_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception:
        return None


def extract_json(raw_content):
    if not raw_content: return None
    try:
        start = raw_content.find('{')
        end = raw_content.rfind('}')
        if start != -1 and end != -1:
            return json.loads(raw_content[start: end + 1])
    except:
        pass

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
        system_prompt = f"You are an expert {self.role}. Reasoning allowed, but end with valid JSON."
        user_prompt = f"""
        Focal Question: "{focal_question}"

        TASKS:
        1. List 5 driving forces per PESTLE factor.
        2. Select 3 critical forces.
        3. Create 8 Scenarios (Title, State of Forces, Description, Signposts, Black Swan).

        Output strictly JSON structure:
        {{
            "pestle": {{ ... }},
            "selected_forces": [...],
            "scenarios": [
                {{
                    "title": "...", 
                    "force_states": {{ "Force A": "State X", ... }},
                    "description": "...",
                    "signposts": [...],
                    "black_swan": "..."
                }}
            ]
        }}
        """
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.7
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

        for role, data in agent_outputs.items():
            if data and 'scenarios' in data:
                for s in data['scenarios']:
                    s_id = f"{role}: {s['title']}"
                    master_scenarios[s_id] = s
                    states_str = " | ".join([f"{k}: {v}" for k, v in s.get('force_states', {}).items()])
                    scenarios_text_list.append(
                        f"ID: {s_id}\n   Forces: [{states_str}]\n   Desc: {s['description'][:250]}...")

        random.shuffle(scenarios_text_list)
        scenarios_block = "\n\n".join(scenarios_text_list)

        conversation = [
            {"role": "system", "content": "You are a Moderator. Select 4 diverse scenarios from at least 3 roles."},
            {"role": "user",
             "content": f"Question: {focal_question}\nCandidates:\n{scenarios_block}\n\nROUND 1: Select 4. Output JSON: {{ 'rationale': '...', 'selected_ids': [...] }}"}
        ]

        debate_log = []
        final_selection_ids = []

        # 3 Rounds for speed
        for round_num in range(1, 4):
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=conversation,
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                content = extract_json(response.choices[0].message.content)
                if not content: break

                current_ids = content.get("selected_ids", [])
                debate_log.append(f"Round {round_num}: {content.get('rationale', 'No rationale')}")
                final_selection_ids = current_ids

                conversation.append(response.choices[0].message)
                if round_num < 3:
                    conversation.append({"role": "user",
                                         "content": f"ROUND {round_num + 1}: Critique {current_ids}. Ensure diversity. Swap if needed."})

            except Exception:
                break

        final_scenarios_full = []
        for s_id in final_selection_ids:
            if s_id in master_scenarios:
                full_obj = master_scenarios[s_id]
                full_obj['id_key'] = s_id
                final_scenarios_full.append(full_obj)

        return {"debate_log": debate_log, "final_scenarios": final_scenarios_full}


def run_foresight_simulation(focal_question, model_map=None):
    if model_map is None: model_map = {}

    print(f"🚀 Starting: {focal_question}")
    agent_results = {}

    # Use max_workers=6 for paid tier, 3 for free tier
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_role = {}
        for role in ROLES:
            # Pick model for this specific role, or default
            m_id = model_map.get(role, DEFAULT_AGENT_MODEL)
            future_to_role[executor.submit(Agent(role, m_id).generate_pestle_and_scenarios, focal_question)] = role

        for future in as_completed(future_to_role):
            role = future_to_role[future]
            try:
                res = future.result()
                if res: agent_results[role] = res
            except:
                pass

    # Run Moderator
    mod_model = model_map.get("Moderator", DEFAULT_MODERATOR_MODEL)
    moderator_report = Moderator(mod_model).debate_and_select(focal_question, agent_results)

    return {"agent_data": agent_results, "moderator_report": moderator_report}