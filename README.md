
This branch is for possible use if we choose to translate the LaTeX to natural language before we feed the questions/answers to the language models

If you run the jupyter notebook data_preprocessing it will generate a pandas DataFrame with the questions translated from LaTeX to natural language as well as a .csv file with this and an excel file


LLM powered math solver for the AIMO Kaggle Competition with Langchain.

## Overview of the App

<img src="diagram.jpg" width="75%">

- Accepts input text (*e.g.* `What are the three key pieces of advice for learning how to code?`) as prompt input using Streamlit's `st.text_area()`, then assign this to the `text` variable.
- LLM model is called via `llm()` and it is applied on the prompt input `text` to generate a response via `llm(text)`

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://langchain-quickstart.streamlit.app/)

## Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.

## Try out the app

Once the app is loaded, go ahead and enter your OpenAI API key and type a question in the text box and wait for a generated response.
