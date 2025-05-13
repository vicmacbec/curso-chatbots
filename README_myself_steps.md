# Installation

- vscode
    - extentions
        - mermaid (optional)
        - git (optional)
        - jupiter (optional)
        - copilot (optional)
        - camouflage (optional)
- git
    - fork https://github.com/educep/curso-chatbots
    - cd path/to/clone
    - git clone https://github.com/vicmacbec/curso-chatbots.git
- make (optional)
    - if dont, install manually packages of Makefile
        - curl -LsSf https://astral.sh/uv/install.sh | sh
        - uv venv .venv --python 3.12.4 # Para instalar torchaudio
        - source .venv/bin/activate
        - uv pip install pip --upgrade
        - uv pip install -r requirements.txt
        - Instalar torch en Ubuntu:
            - uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        - Instalar torch en Mac:
            - uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
- Open AI
    - Modificar API en .env file
- Hugging face

* Por hacer:
    - git
    - API
        - ejecutar scripts
            - tmp.py
            - a_basic.py




- Parámetros:
    - temperatura [0, 2]:
        - alta - es más creativo
        - baja - más estricto
    - top_p [0, 1]:
        - toma hasta el acumulado del porcentaje p de las palabras
    - max_tokens
        tokens máximos a escribir 
    - frequency_penalty
    - presence_penalty
