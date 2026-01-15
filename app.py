import os
import logging
from flask import Flask, render_template, request, jsonify, Response
from functools import wraps
from agents import run_foresight_simulation, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- PASSWORD PROTECTION CONFIGURATION ---
# Username is 'team'. Password MUST be set via APP_PASSWORD environment variable.
USERNAME = "team"
PASSWORD = os.environ.get("APP_PASSWORD")

if not PASSWORD:
    raise ValueError("APP_PASSWORD environment variable is required. Set it before starting the application.")


def check_auth(username, password):
    """Checks if username/password are valid."""
    return username == USERNAME and password == PASSWORD


def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


# -----------------------------------------

@app.route('/')
@requires_auth  # <--- Locks the dashboard behind a password
def index():
    return render_template('index.html')


@app.route('/visualize')
@requires_auth  # <--- Locks visualization behind a password
def visualize():
    return render_template('visualize.html')


# Request timeout in seconds
REQUEST_TIMEOUT = 300  # 5 minutes max for simulation

@app.route('/run', methods=['POST'])
@requires_auth  # <--- Locks the API behind a password
def run():
    data = request.json
    focal_question = data.get('question')

    # Extract the dynamic model selection from the frontend
    model_config = data.get('model_config', {})

    if not focal_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Run the simulation with the user's specific model choices
        result = run_foresight_simulation(focal_question, model_config, timeout=REQUEST_TIMEOUT)
        return jsonify(result)
    except ValidationError as e:
        # Validation errors are safe to return to the client
        return jsonify({"error": str(e)}), 400
    except TimeoutError:
        logger.warning(f"Simulation timed out for question: {focal_question[:100]}")
        return jsonify({"error": "Simulation timed out. Please try a simpler question or try again later."}), 504
    except Exception as e:
        # Log full error details server-side, return generic message to client
        logger.exception(f"Simulation failed for question: {focal_question[:100]}")
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500


if __name__ == '__main__':
    # Use the port Render assigns, or default to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)