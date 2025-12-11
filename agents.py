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

# Kimi K2 Thinking is excellent for the heavy lifting of scenario generation
AGENT_MODEL = "moonshotai/Kimi-K2-Thinking"

# DeepSeek V3 remains the Moderator (Cost-effective & High Logic)
MODERATOR_MODEL = "deepseek-ai/DeepSeek-V3"

ROLES = [
    "Economist", "Sociologist", "Technologist",
    "Ecologist", "Political Scientist", "Legal Ethicist"
]


def clean_json_response(raw_content):
    if not raw_content: return ""

    # 1. Attempt to find the first '{' and the last '}'
    # This ignores any "Thinking Process" text that comes before the JSON
    start_index = raw_content.find('{')
    end_index = raw_content.rfind('}')

    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_str = raw_content[start_index: end_index + 1]
    else:
        # Fallback: just return the raw string (it might be pure JSON)
        json_str = raw_content

    # 2. Remove any leftover Markdown formatting from the extracted string
    cleaned = re.sub(r"```json", "", json_str, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)

    return cleaned.strip()


class Agent:
    def __init__(self, role):
        self.role = role

    def generate_pestle_and_scenarios(self, focal_question):
        # Kimi K2 performs best when you explicitly tell it to use its reasoning capabilities
        system_prompt = f"You are an expert {self.role}. You are a 'Thinking Model' that reasons deeply before answering. Use your internal chain-of-thought to derive unique insights, but ensure your Final Output is strictly valid JSON."

        user_prompt = f"""
        Focal Question: "{focal_question}"

        TASK 1: PESTLE ANALYSIS
        List 5 driving forces per PESTLE factor.

        TASK 2: SELECTION
        Select the 3 most critical driving forces (Variables) for this question.

        TASK 3: SCENARIO GENERATION (Create 8 Scenarios)
        For EACH scenario, you must define the specific STATE of your 3 forces.

        Example of States:
        Force: "Global Trade" -> State: "Fragmented" OR "Unified"
        Force: "AI Regulation" -> State: "Banned" OR "Laissez-faire"

        Output strictly JSON:
        {{
            "pestle": {{ "Political": [], ... }},
            "selected_forces": ["Force A", "Force B", "Force C"],
            "scenarios": [
                {{
                    "title": "Title", 
                    "force_states": {{ "Force A": "State X", "Force B": "State Y", "Force C": "State Z" }},
                    "description": "Detailed narrative...",
                    "signposts": ["Early Indicator 1", "Early Indicator 2"],
                    "black_swan": "Low probability high impact event"
                }}
            ]
        }}
        """
        try:
            response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                # We don't enforce json_object mode strictly for Kimi as it sometimes interferes with its "Thinking" block
                # Instead, we rely on the clean_json_response function
                temperature=0.7
            )
            return json.loads(clean_json_response(response.choices[0].message.content))
        except Exception as e:
            print(f"❌ {self.role} Error: {e}")
            return None


class Moderator:
    def debate_and_select(self, focal_question, agent_outputs):
        # 1. Prepare Master List
        master_scenarios = {}
        scenarios_text_list = []

        for role, data in agent_outputs.items():
            if data and 'scenarios' in data:
                for s in data['scenarios']:
                    # Unique ID: "Economist: The Great Stagnation"
                    s_id = f"{role}: {s['title']}"
                    master_scenarios[s_id] = s

                    # Format for the prompt
                    states_str = " | ".join([f"{k}: {v}" for k, v in s.get('force_states', {}).items()])
                    entry = f"ID: {s_id}\n   Forces: [{states_str}]\n   Desc: {s['description'][:250]}..."
                    scenarios_text_list.append(entry)

        # Shuffle to prevent bias towards the first agent in the list
        random.shuffle(scenarios_text_list)
        scenarios_block = "\n\n".join(scenarios_text_list)

        # 2. Debate Loop
        conversation = [
            {"role": "system", "content": f"""
            You are a Foresight Moderator. 
            Constraint 1: You must select scenarios from at least 3 DIFFERENT ROLES (e.g. 1 Economist, 1 Technologist, 1 Ecologist). 
            Constraint 2: Do not let one role dominate.
            Constraint 3: Ensure maximum structural divergence in the 'Forces'.
            """},
            {"role": "user", "content": f"""
            Focal Question: "{focal_question}"

            CANDIDATE SCENARIOS (shuffled):
            {scenarios_block}

            ROUND 1: INITIAL SELECTION
            Select 4 scenarios that are divergent and plausible.
            Output JSON: {{
                "rationale": "Reasoning...",
                "selected_ids": ["Role: Title 1", "Role: Title 2", "Role: Title 3", "Role: Title 4"]
            }}
            """}
        ]

        debate_log = []
        final_selection_ids = []

        for round_num in range(1, 6):  # 5 Rounds
            try:
                print(f"Moderator Round {round_num}...")
                response = client.chat.completions.create(
                    model=MODERATOR_MODEL,
                    messages=conversation,
                    response_format={"type": "json_object"},
                    temperature=0.7
                )

                content = json.loads(clean_json_response(response.choices[0].message.content))
                current_ids = content.get("selected_ids", [])
                rationale = content.get("rationale", "No rationale")

                debate_log.append(f"Round {round_num}: {rationale}")
                final_selection_ids = current_ids

                conversation.append(response.choices[0].message)

                if round_num < 5:
                    critique_prompt = f"""
                    ROUND {round_num + 1}: CRITIQUE & DIVERSIFY
                    Current Selection: {current_ids}

                    CHECK: Are there at least 3 different roles represented?
                    CHECK: Do the 'Forces' states conflict nicely (e.g. one has High Regulation, one has Low)?

                    ACTION: Swap at least 1 scenario to improve Role Diversity or Structural Divergence.
                    Output JSON with "rationale" and "selected_ids".
                    """
                    conversation.append({"role": "user", "content": critique_prompt})

            except Exception as e:
                print(f"Debate Error: {e}")
                break

        # 3. Retrieve FULL details
        final_scenarios_full = []
        for s_id in final_selection_ids:
            if s_id in master_scenarios:
                full_obj = master_scenarios[s_id]
                full_obj['id_key'] = s_id
                final_scenarios_full.append(full_obj)

        return {
            "debate_log": debate_log,
            "final_scenarios": final_scenarios_full,
        }


def run_foresight_simulation(focal_question):
    print(f"🚀 Starting: {focal_question}")
    agent_results = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_role = {executor.submit(Agent(role).generate_pestle_and_scenarios, focal_question): role for role in
                          ROLES}
        for future in as_completed(future_to_role):
            role = future_to_role[future]
            try:
                res = future.result()
                if res: agent_results[role] = res
                print(f"✅ {role} Done")
            except:
                print(f"❌ {role} Failed")

    if not agent_results:
        return {"error": "All agents failed."}

    print("🧠 Moderator debating...")
    moderator_report = Moderator().debate_and_select(focal_question, agent_results)
    print("🏁 Finished")

    return {"agent_data": agent_results, "moderator_report": moderator_report}