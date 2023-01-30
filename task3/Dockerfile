FROM ghcr.io/osai-ai/dokai:22.03-pytorch
ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_HOME /workdir/data/.torch
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python3 -m pip install ipykernel -U --user --force-reinstall
RUN python3 -m pip install --upgrade pip
RUN pip install git+https://github.com/huggingface/transformers

COPY *.* /workdir/
COPY speaker_verification /workdir/speaker_verification
COPY notebooks /workdir/notebooks

# RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

RUN mkdir /workdir/results

WORKDIR /workdir

ENTRYPOINT tail -f /dev/null


