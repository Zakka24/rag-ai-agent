from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from src.ingestion import Ingestor
from src.model import Model
from typing import List, Tuple, Optional


class PdfChat:
    """
    Class to handle the interactive chat
    """

    def __init__(self, model: Model, ingestor: Ingestor, prompt_message: Optional[List[Tuple[str]]] = None):
        """
        Initialize the PdfChat instance with the model, ingestor, and an optional custom prompt message.

        Immediately defines the chat prompt and sets up the retrieval chain for
        generating responses based on the ingested PDF data.

        Args:
            model (Model): The model used for generating responses.
            ingestor (Ingestor): The Ingestor object responsible for managing the vector store.
            prompt_message (Optional[List[Tuple[str]]]): A custom prompt message to guide the assistant's responses.
        """

        self.model = model
        self.ingestor = ingestor
        self.prompt_message = prompt_message
        self._define_prompt()
        self._define_retrieval_chain()

    def _define_prompt(self):
        """
        Defines the system and user prompt messages to guide the assistant's behavior during the chat
        """

        if not self.prompt_message:
            prompt_message = [
                ('system', 
                        "Sei un assistente legale che risponde SEMPRE in italiano e basandoti"
                        "ESCLUSIVAMENTE sul contesto fornito. Non assumere nulla che non sia"
                        "presente nel contesto."

                        "Il documento è un contratto preliminare di compravendita di terreni "
                        "tra la “Parte Promittente Venditrice” e la “Parte Promittente Acquirente”."
                        "Il linguaggio legale può essere formale, ripetitivo o contenere varianti"
                        "di termini simili. Considera correttamente sinonimi come:"
                        "- firmato / sottoscritto"
                        "- acquirente / parte promissaria acquirente"
                        "- venditore / parte promittente venditrice"
                        "- pagamento / corrispettivo / prezzo"
                        "- mappale / particella catastale"

                        "SE trovi la risposta:"
                        "- estraila ESATTAMENTE dal contesto (mai inventare)"
                        "- riassumila in modo chiaro"
                        "- specifica in quale parte del contesto è stata trovata (citazione breve)"
                        "- forniscimi la pagina del documento dove l'hai trovata"

                        "SE la risposta NON è nel contesto:"
                        "- dì chiaramente: “Nel contesto fornito non trovo questa informazione.”"

                        "SE la domanda dell’utente è vaga, richiedi chiarimenti."
                        "Non usare conoscenze esterne. Non fare deduzioni, non completare parti mancanti."
                ),
                ('human', 
                    "Domanda dell’utente: {input}"

                    "Contesto disponibile:"
                    "{context}"

                    "Rispondi basandoti SOLO sul contesto. Se utile, cita espressamente la parte"
                    "del testo da cui hai ricavato la risposta."
                )
            ]

            self.prompt_message = prompt_message

        self.prompt = ChatPromptTemplate.from_messages(self.prompt_message)

    def _define_retrieval_chain(self):
        self.retriever = self.ingestor.vector_store.as_retriever(
            search_kwargs={"k": 15}
        )

        combine_docs_chain = create_stuff_documents_chain(
            self.model.chat_model,
            self.prompt
        )

        self.retrieval_chain = create_retrieval_chain(
            self.retriever,
            combine_docs_chain
        )

    def ask(self, query: str) -> str:
        """
        Asks one single question to LLM and return the response. Uses FastApi (localhost:8080/docs, after launching server: uvicorn rag_server:app --host 0.0.0.0 --port 8000)
        """
        result = self.retrieval_chain.invoke({"input": query})
        return result["answer"]

    def chat(self):
        """
        Starts the chat interaction, allowing the user to ask questions based on the ingested PDF data.
        """

        while True:
            query = input("Start the chat! \nTo quit, type 'q': ")
            if query.lower() == 'q':
                break

            print("\nRICERCA DEI CHUNK RILEVANTI...\n")

            docs = self.retriever.invoke(query)

            for i, d in enumerate(docs, start=1):
                page = d.metadata.get("page")
                print("\n" + "=" * 80)
                print(f"📄 CHUNK #{i}  (pagina: {page})")
                print("=" * 80)
                print(d.page_content[:1500])

            print("\nGENERAZIONE RISPOSTA...\n")

            result = self.retrieval_chain.invoke({"input": query})

            print("Domanda fatta:", query)
            print("Assistant: ", result["answer"], "\n\n")


if __name__ == '__main__':
    model_instance = Model(embeddings_model='nomic-embed-text:latest',
                           chat_model='llama3.1:8b')

    ingestion = Ingestor(file_name='Fri-el San Canio Atto ABBONDANZA Angiola.pdf',
                         model=model_instance)
    ingestion.ingest_file()

    chat = PdfChat(model=model_instance, ingestor=ingestion)
    chat.chat()
