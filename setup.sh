## if .venv is not installed, install it
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
## activate the virtual environment
source .venv/bin/activate
## install the requirements
pip install -r requirements.txt