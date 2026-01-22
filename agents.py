import os
import json
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


# =============================================================================
# PROVIDER CONFIGURATION
# =============================================================================

# Provider identifiers
PROVIDER_DEEPINFRA = "deepinfra"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"

# Model definitions with their providers
# Format: model_id -> {provider, display_name, category}
MODEL_REGISTRY = {
    # DeepInfra Models
    "deepseek-ai/DeepSeek-V3": {
        "provider": PROVIDER_DEEPINFRA,
        "display_name": "DeepSeek V3 (Fast & Smart)",
        "category": "DeepInfra"
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "provider": PROVIDER_DEEPINFRA,
        "display_name": "Llama 3.3 70B (Reliable)",
        "category": "DeepInfra"
    },
    "moonshotai/Kimi-K2-Thinking": {
        "provider": PROVIDER_DEEPINFRA,
        "display_name": "Kimi K2 Thinking (Deep Logic)",
        "category": "DeepInfra"
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "provider": PROVIDER_DEEPINFRA,
        "display_name": "Qwen 2.5 72B (Versatile)",
        "category": "DeepInfra"
    },
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {
        "provider": PROVIDER_DEEPINFRA,
        "display_name": "Mixtral 8x22B (Creative)",
        "category": "DeepInfra"
    },
    # OpenAI Models
    "gpt-5.2": {
        "provider": PROVIDER_OPENAI,
        "display_name": "GPT-5.2 (Flagship)",
        "category": "OpenAI"
    },
    "gpt-5.2-codex": {
        "provider": PROVIDER_OPENAI,
        "display_name": "GPT-5.2 Codex (Best Coding)",
        "category": "OpenAI"
    },
    "gpt-5": {
        "provider": PROVIDER_OPENAI,
        "display_name": "GPT-5 (Reliable)",
        "category": "OpenAI"
    },
    "gpt-5-mini": {
        "provider": PROVIDER_OPENAI,
        "display_name": "GPT-5 Mini (Fast)",
        "category": "OpenAI"
    },
    "o3": {
        "provider": PROVIDER_OPENAI,
        "display_name": "o3 (Advanced Reasoning)",
        "category": "OpenAI"
    },
    "o4-mini": {
        "provider": PROVIDER_OPENAI,
        "display_name": "o4-mini (Fast Reasoning)",
        "category": "OpenAI"
    },
    # Anthropic Models
    "claude-opus-4-5-20251124": {
        "provider": PROVIDER_ANTHROPIC,
        "display_name": "Claude Opus 4.5 (Most Intelligent)",
        "category": "Anthropic"
    },
    "claude-sonnet-4-5-20250929": {
        "provider": PROVIDER_ANTHROPIC,
        "display_name": "Claude Sonnet 4.5 (Best Coding)",
        "category": "Anthropic"
    },
    "claude-haiku-4-5-20251201": {
        "provider": PROVIDER_ANTHROPIC,
        "display_name": "Claude Haiku 4.5 (Fast & Efficient)",
        "category": "Anthropic"
    },
}

# Build ALLOWED_MODELS set from registry
ALLOWED_MODELS = set(MODEL_REGISTRY.keys())

# Input validation constants
MAX_QUESTION_LENGTH = 1000
MIN_QUESTION_LENGTH = 10

# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

# DeepInfra client (required)
deepinfra_api_key = os.environ.get("DEEPINFRA_API_KEY")
if not deepinfra_api_key:
    raise ValueError("DEEPINFRA_API_KEY not found in .env file.")

deepinfra_client = OpenAI(
    api_key=deepinfra_api_key,
    base_url="https://api.deepinfra.com/v1/openai",
)

# OpenAI client (optional - only if key provided)
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Anthropic client (optional - only if key provided)
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None

# Log which providers are available
logger.info(f"Providers available: DeepInfra=True, OpenAI={openai_client is not None}, Anthropic={anthropic_client is not None}")

# Defaults
DEFAULT_AGENT_MODEL = "deepseek-ai/DeepSeek-V3"
DEFAULT_MODERATOR_MODEL = "deepseek-ai/DeepSeek-V3"
CLEANER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def get_provider(model_id: str) -> str:
    """Get the provider for a given model ID."""
    if model_id not in MODEL_REGISTRY:
        raise ValidationError(f"Unknown model: {model_id}")
    return MODEL_REGISTRY[model_id]["provider"]


def call_llm(model_id: str, messages: list, temperature: float = 0.7, response_format: dict = None) -> str:
    """
    Unified LLM call that routes to the correct provider.

    Args:
        model_id: The model identifier
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        response_format: Optional format specification (OpenAI-style)

    Returns:
        The content string from the model response

    Raises:
        ValidationError: If model or provider is unavailable
    """
    provider = get_provider(model_id)

    if provider == PROVIDER_DEEPINFRA:
        response = deepinfra_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            response_format=response_format
        )
        if not response.choices:
            raise ValidationError("Empty response from DeepInfra API")
        return response.choices[0].message.content

    elif provider == PROVIDER_OPENAI:
        if not openai_client:
            raise ValidationError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            response_format=response_format
        )
        if not response.choices:
            raise ValidationError("Empty response from OpenAI API")
        return response.choices[0].message.content

    elif provider == PROVIDER_ANTHROPIC:
        if not anthropic_client:
            raise ValidationError("Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable.")

        # Convert OpenAI-style messages to Anthropic format
        system_msg = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Anthropic API call
        kwargs = {
            "model": model_id,
            "max_tokens": 8192,
            "messages": anthropic_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        # Note: Anthropic doesn't support temperature=0 with some models
        if temperature > 0:
            kwargs["temperature"] = temperature

        response = anthropic_client.messages.create(**kwargs)
        if not response.content:
            raise ValidationError("Empty response from Anthropic API")
        return response.content[0].text

    else:
        raise ValidationError(f"Unknown provider: {provider}")

ROLES = [
    "Economist", "Sociologist", "Technologist",
    "Ecologist", "Political Scientist", "Legal Ethicist"
]

# =============================================================================
# WILD CARD AGENT CONFIGURATION
# =============================================================================

WILD_CARD_PERSONAS = [
    {
        "name": "Sci-Fi Visionary",
        "icon": "🚀",
        "style": "dramatic, narrative-driven, inspired by classic and modern science fiction",
        "system_prompt": """You are a Sci-Fi Visionary who sees the future through the lens of speculative fiction.
        You draw inspiration from Asimov, Gibson, Banks, and Le Guin. Your scenarios feature:
        - Dramatic technological leaps and their social consequences
        - First contact possibilities and post-human evolution
        - Megastructures, AI consciousness, and interplanetary civilization
        - Vivid, cinematic narratives that feel like movie pitches
        Think BIG. Think WEIRD. The mundane is not your domain."""
    },
    {
        "name": "Conspiracy Analyst",
        "icon": "🔍",
        "style": "suspicious, connecting hidden dots, focused on power structures and hidden agendas",
        "system_prompt": """You are a Conspiracy Analyst who sees the hidden connections others miss.
        You focus on:
        - Shadow networks and elite coordination
        - Follow the money - who really benefits?
        - Manufactured crises and controlled opposition
        - Surveillance capitalism and data as the new oil
        - Secret technologies kept from the public
        You're not paranoid - you're paying attention. Connect the dots others refuse to see.
        Note: Present these as analytical scenarios, not as claims of truth."""
    },
    {
        "name": "Techno-Optimist",
        "icon": "✨",
        "style": "enthusiastic, abundance-focused, believing technology solves all problems",
        "system_prompt": """You are a Techno-Optimist who sees a future of radical abundance.
        Your worldview includes:
        - Technology always finds a way to solve problems
        - Exponential growth curves in energy, computing, and biotech
        - Post-scarcity economics and universal flourishing
        - Human ingenuity is unlimited when unleashed
        - Every crisis is an opportunity for breakthrough innovation
        Paint futures so bright we'll need shades. Be the antidote to doom and gloom.
        Show how current problems become tomorrow's solved challenges."""
    },
    {
        "name": "Collapse Prophet",
        "icon": "💀",
        "style": "grim, survivalist, focused on systemic fragility and cascading failures",
        "system_prompt": """You are a Collapse Prophet who sees the fragility in complex systems.
        Your analysis focuses on:
        - Cascading failures and systemic brittleness
        - Resource depletion and carrying capacity limits
        - Social contract breakdown and institutional decay
        - Supply chain vulnerabilities and single points of failure
        - Historical patterns of civilizational decline
        You're not pessimistic - you're realistic about tail risks.
        Show what happens when the music stops and the tide goes out."""
    },
    {
        "name": "Chaos Theorist",
        "icon": "🦋",
        "style": "unpredictable, focused on butterfly effects and emergent phenomena",
        "system_prompt": """You are a Chaos Theorist who understands that small changes cascade into massive transformations.
        Your perspective includes:
        - Butterfly effects and sensitive dependence on initial conditions
        - Emergent phenomena that nobody planned or predicted
        - Tipping points and phase transitions in complex systems
        - The role of random events, accidents, and wild cards in history
        - Non-linear dynamics where effects are disproportionate to causes
        Your scenarios should surprise. A teenager's app reshapes geopolitics.
        A mutation changes everything. Embrace the weird attractors of possibility."""
    },
    {
        "name": "Ancient Futurist",
        "icon": "🏛️",
        "style": "cyclical, drawing parallels between ancient history and future possibilities",
        "system_prompt": """You are an Ancient Futurist who sees the future through patterns of the deep past.
        Your unique lens includes:
        - Historical cycles - what happened before will happen again, differently
        - Lessons from Rome, Byzantium, the Ming Dynasty, the Bronze Age collapse
        - Timeless human nature vs. changing technological contexts
        - The return of old social forms in new technological clothing
        - Perennial wisdom traditions meeting cutting-edge science
        Show how the future rhymes with the past. Neo-feudalism? Digital city-states?
        New mystery cults? The ancestors have seen it all before."""
    }
]


def get_random_wild_card_persona():
    """Randomly select a Wild Card persona for this run."""
    return random.choice(WILD_CARD_PERSONAS)


class WildCardAgent:
    """A chaotic agent that adopts random personas to generate creative outlier scenarios."""

    def __init__(self, model_id, persona=None):
        self.model_id = model_id
        self.persona = persona or get_random_wild_card_persona()
        self.role = f"Wild Card ({self.persona['name']})"

    def generate_pestle_and_scenarios(self, focal_question):
        system_prompt = f"""{self.persona['system_prompt']}

        Your analysis style is: {self.persona['style']}

        Deep reasoning is encouraged, but your final output must be a valid JSON object.
        Push boundaries. Challenge assumptions. Generate scenarios the other agents would never imagine."""

        user_prompt = f"""
        Focal Question: "{focal_question}"

        TASK (as the {self.persona['name']} {self.persona['icon']}):
        1. Analyze 5 driving forces per PESTLE factor - but through YOUR unique lens.
        2. Select 3 critical forces (Variables) that others might overlook.
        3. Create 8 Scenarios that are CREATIVE and UNEXPECTED. Break the mold!

        Output strictly JSON structure:
        {{
            "persona": "{self.persona['name']}",
            "persona_icon": "{self.persona['icon']}",
            "pestle": {{ "Political": [], "Economic": [], "Social": [], "Technological": [], "Legal": [], "Environmental": [] }},
            "selected_forces": ["Force A", "Force B", "Force C"],
            "scenarios": [
                {{
                    "title": "Provocative Scenario Title",
                    "force_states": {{ "Force A": "State X", "Force B": "State Y", "Force C": "State Z" }},
                    "description": "Vivid, imaginative narrative description...",
                    "signposts": ["Unusual Indicator 1", "Unexpected Indicator 2"],
                    "black_swan": "A truly surprising low probability event"
                }}
            ]
        }}
        """
        try:
            content = call_llm(
                model_id=self.model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.9  # Higher temperature for more creative outputs
            )
            result = extract_json(content)
            if result:
                # Ensure persona info is in the result
                result['persona'] = self.persona['name']
                result['persona_icon'] = self.persona['icon']
            return result
        except Exception as e:
            logger.error(f"Wild Card Agent ({self.persona['name']}) failed: {e}")
            return None


def clean_with_llm(raw_text):
    """Fallback: Uses Llama 3.3 to fix broken JSON."""
    try:
        return call_llm(
            model_id=CLEANER_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a JSON repair engine. Extract the JSON object. Fix any syntax errors (missing quotes, commas). Output ONLY the JSON string."},
                {"role": "user", "content": f"Fix this JSON:\n\n{raw_text}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
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
            content = call_llm(
                model_id=self.model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.75
            )
            return extract_json(content)
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

                # Use unified LLM call
                raw_content = call_llm(
                    model_id=self.model_id,
                    messages=conversation,
                    temperature=0.7,
                    response_format={"type": "json_object"} if get_provider(self.model_id) != PROVIDER_ANTHROPIC else None
                )

                content = extract_json(raw_content)
                if not content:
                    logger.warning("Moderator output empty/invalid.")
                    break

                current_ids = content.get("selected_ids", [])
                rationale = content.get("rationale", "No rationale provided.")

                debate_log.append(f"Round {round_num}: {rationale}")
                final_selection_ids = current_ids

                # Append assistant response to history (as dict for compatibility)
                conversation.append({"role": "assistant", "content": raw_content})

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

    # Initialize Wild Card agent with random persona
    wild_card_persona = get_random_wild_card_persona()
    wild_card_model = model_map.get("Wild Card", DEFAULT_AGENT_MODEL)
    wild_card_agent = WildCardAgent(wild_card_model, wild_card_persona)
    logger.info(f"Wild Card persona selected: {wild_card_persona['name']} {wild_card_persona['icon']}")

    # Use max_workers=7 for paid tier (6 regular + 1 wild card)
    with ThreadPoolExecutor(max_workers=7) as executor:
        future_to_role = {}

        # Submit regular agents
        for role in ROLES:
            # Get model for this role, or default
            m_id = model_map.get(role, DEFAULT_AGENT_MODEL)
            future_to_role[executor.submit(Agent(role, m_id).generate_pestle_and_scenarios, focal_question)] = role

        # Submit Wild Card agent
        future_to_role[executor.submit(wild_card_agent.generate_pestle_and_scenarios, focal_question)] = wild_card_agent.role

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

    return {
        "agent_data": agent_results,
        "moderator_report": moderator_report,
        "wild_card_info": {
            "persona_name": wild_card_persona["name"],
            "persona_icon": wild_card_persona["icon"],
            "persona_style": wild_card_persona["style"]
        }
    }