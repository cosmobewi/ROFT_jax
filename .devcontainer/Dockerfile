FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Évite les invites APT et le cache pip
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Limiter la pré-allocation mémoire GPU par JAX/XLA (optionnel)
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git wget curl build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Crée un venv pour contourner PEP 668 ---
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Met à jour pip/setuptools/wheel dans le venv (OK, pas bloqué par PEP 668)
RUN python -m pip install --upgrade pip setuptools wheel

# --- JAX GPU (CUDA 13) ---
# Le -f est nécessaire pour récupérer les wheels CUDA
RUN pip install -U "jax[cuda13]" 
# --- Dépendances notebook H0LiCOW ---
RUN pip install emcee numpy tqdm scipy matplotlib pandas scikit-learn astropy lenstronomy jupyterlab tabulate numpyro arviz jax jaxlib matplotlib pandas numpy

WORKDIR /workspace

CMD ["/bin/bash"]
