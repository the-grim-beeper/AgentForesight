import os
from flask import Flask, render_template, request, jsonify, Response
from functools import wraps
from agents import run_foresight_simulation

app = Flask(__name__)

# --- PASSWORD PROTECTION CONFIGURATION ---
USERNAME = "team"
# We will set the real password in the online dashboard later
PASSWORD = os.environ.get("APP_PASSWORD", "default_secret")


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
@requires_auth  # <--- This locks the page
def index():
    return render_template('index.html')


@app.route('/run', methods=['POST'])
@requires_auth  # <--- This locks the API
def run():
    data = request.json
    focal_question = data.get('question')

    if not focal_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Run the heavy lifting
        result = run_foresight_simulation(focal_question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Use the port Render assigns, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)