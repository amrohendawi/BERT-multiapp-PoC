# Import necessary libraries
import streamlit as st

# Define the multiapp class to manage the multiple apps in our program


class Multiapp:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.apps = []

    def add_app(self, title, func) -> None:
        """Class Method to Add apps to the project
        Args:
            title ([str]): The title of app which we are adding to the list of apps 

            func: Python function to render this app in Streamlit
        """

        self.apps.append(
            {
                "title": title,
                "function": func
            }
        )

    def run(self):
        # Drodown to select the app to run
        app = st.sidebar.selectbox(
            'App Navigation',
            self.apps,
            format_func=lambda app: app['title']
        )

        # run the app function
        app['function']()
