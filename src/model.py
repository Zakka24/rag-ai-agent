import re
import subprocess
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama


class Model:
    """
    Class to initialize and manage machine learning models for embeddings and chat.
    """

    def __init__(self, embeddings_model: str, chat_model: str):
        """
        Initializes the Model class with the specified embeddings and chat models.

        Args:
            embeddings_model (str): The name of the embeddings model to use.
            chat_model (str): The name of the chat model to use.
        """
        self.embeddings = embeddings_model
        self.model = chat_model

        self._instantiate_models()

    def _instantiate_models(self):
        """
        Ensures required models for embeddings and chat are available and initializes them.

        Checks for the specified models locally using `ollama list`. If missing, downloads them
        using `ollama pull` or `ollama run`. Finally, sets up the models for embeddings and chat.
        """

        available_models = subprocess.run('ollama list',
                                          shell=True,
                                          capture_output=True,
                                          text=True)

        list_available_models = re.findall(r'^(\S+):', available_models.stdout, re.MULTILINE)

        if self.embeddings.split(':')[0] not in list_available_models:
            subprocess.run(f'ollama pull {self.embeddings}', shell=True)

        if self.model.split(':')[0] not in list_available_models:
            subprocess.run(f'ollama run {self.model}', shell=True)

        self.embeddings_model = OllamaEmbeddings(model=self.embeddings)
        self.chat_model = ChatOllama(
            model=self.model,
            temperature=0.0
        )
