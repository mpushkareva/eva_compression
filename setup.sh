git clone https://github.com/mpushkareva/eva_compression.git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd /workspace/eva_compression
uv venv compress --python 3.11
source compress/bin/activate
uv pip install -r requirements.txt
export PYTHONPATH=/workspace/eva_compression:$PYTHONPATH