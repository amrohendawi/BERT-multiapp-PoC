# Multi-app demonstrating various BERT downstream tasks

This is a demo of various BERT tasks running over a streamlit app.

## Getting Started

### Running the app locally

First, clone the git repo, then create a virtual environment for installing dependencies.
Feel free to use conda or any other environment manager of your choice.

```
git clone https://github.com/amrohendawi/BERT-multiapp-PoC.git
cd BERT-multiapp-PoC
python -m venv venv
```

Activate the environment and install the requirements with pip

```
source venv/bin/activate
pip install -r requirements.txt
```

Run the app

```
python -m streamlit run app.py
```

## About the app



## Built With

- [Streamlit](https://streamlit.io/)
- [Transformers](https://huggingface.co/docs/transformers/index)

## Heroku DevOps

You can push the project to production by simply creating a new app on heroku and connecting it to your github repo within 60 seconds.
