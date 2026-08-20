# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
# INVAR agent container: llama.cpp runtime + invar. Mount your models and
# worldline; the endpoint stays inside the container network unless you map it.
#
#   docker build -t anomly/invar .
#   docker run -p 127.0.0.1:8577:8577 -v $PWD/models:/models -v $PWD/data:/data \
#     anomly/invar invar serve --host 0.0.0.0 --model /models/your.gguf \
#     --worldline /data/worldline.jsonl
# (--host 0.0.0.0 is safe here: the docker -p mapping pins it to host loopback.)
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /src/llama.cpp \
 && cmake -S /src/llama.cpp -B /src/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
 && cmake --build /src/llama.cpp/build -t llama-cli -j"$(nproc)"
COPY . /src/invar
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist /src/invar

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* && useradd -m invar
COPY --from=build /src/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
ENV LD_LIBRARY_PATH=/usr/local/lib INVAR_LLAMA_BIN=/usr/local/bin/llama-cli
USER invar
WORKDIR /data
EXPOSE 8577 8579
CMD ["invar", "--help"]
