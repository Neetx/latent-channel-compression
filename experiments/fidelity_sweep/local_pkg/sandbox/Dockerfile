# Locked-down image for executing untrusted model-generated Python in the deliberation toolcaller.
# Build:  docker build -t lcc-pysandbox experiments/fidelity_sweep/local_pkg/sandbox
# Used by py_sandbox.run_python_sandboxed (which adds --network none, --read-only, rlimits, etc.).
FROM python:3.12-slim
RUN pip install --no-cache-dir sympy numpy && rm -rf /root/.cache
USER 65534:65534
WORKDIR /work
