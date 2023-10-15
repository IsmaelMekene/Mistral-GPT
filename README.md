# Mistral-PDF

Make sure to first install poetry and docker desktop once the repository has been cloned!

Start by running `docker compose up -d` at the root of this repository.

Thereafter, run `poetry shell` and `poetry install` in order to have the required packages and libraries.

Make sure to have the PDF file in the `data/` directory and update your path in `mistral_pdf/embeddings.py`.

Download the [Mistral 7B model](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) and have it in the `model/` directory.

Generate your embeddings by run `poetry run python mistral_pdf/embeddings.py`
Check if the llm model class load well with `poetry run python load_model.py`
And finally build the app up with `poetry run python app.py` and start prompting!
