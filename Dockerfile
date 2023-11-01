FROM python:3.8-slim

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    # for installing poetry
    curl \
    # for installing git deps
    git \
    # for building python deps
    build-essential

# make poetry install to this location
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

# install poetry
RUN curl -sSL https://install.python-poetry.org | python -

WORKDIR /MISTRAL-GPT

COPY pyproject.toml /MISTRAL-GPT/pyproject.toml
COPY poetry.lock /MISTRAL-GPT/poetry.lock

RUN poetry install --no-root --no-dev

COPY data /MISTRAL-GPT/data 

COPY mistral_pdf /MISTRAL-GPT/mistral_pdf

COPY model /MISTRAL-GPT/model

COPY static /MISTRAL-GPT/static

COPY templates /MISTRAL-GPT/templates

COPY tests /MISTRAL-GPT/tests

COPY app.py /MISTRAL-GPT/app.py

COPY load_model.py /MISTRAL-GPT/load_model.py

COPY README.md /MISTRAL-GPT/README.md


RUN poetry install --no-dev

ENTRYPOINT [ "poetry", "run", "python" ]
CMD [ "mistral_pdf/embeddings.py" ]

ENTRYPOINT [ "poetry", "run", "python" ]
CMD [ "load_model.py" ]

ENTRYPOINT [ "poetry", "run", "python" ]
CMD [ "app.py" ]