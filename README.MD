# agent-triage-sast

Agente para realizar el triage y analisis de vulnerabilidades de código fuente usando dos MCP server para conectarse a Slack y GitHub y un RAG con Milvus y Redis para memoria.

## Setup local

Inicializar el ambiente de Python

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

Instalar las dependencias

```bash
pip install -r requirements.txt
```

Ejecutar localmente

```bash
python agent-triage-analysis.py
```

## Generar imagen de contenedor

### Variables
```bash
export USER=lcoronad
export ORGANIZATION_QUAY=lcoronad
export VERSION=1.0.0
```

### Build imagen
```bash
podman build -t agent-triage-sast .
```

### Run local
```bash
podman run -p 7861:7861 -e LLAMA_STACK_BASE_URL=http://192.168.68.122:8321 --name agent-triage-sast agent-triage-sast:latest
```

### Login quay
```bash
podman login -u ${USER} quay.io
```

### tag the local container image and push to quay
```bash
podman tag localhost/agent-triage-sast quay.io/${ORGANIZATION_QUAY}/agent-triage-sast:${VERSION}
```

### Push image to quay
```bash
podman push quay.io/${ORGANIZATION_QUAY}/agent-triage-sast:${VERSION}
```