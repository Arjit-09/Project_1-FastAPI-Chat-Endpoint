This project is a FastAPI-based chat application that integrates OpenAI API with FAISS for vector-based retrieval.

The application exposes a /chat endpoint where users can send queries.

Incoming queries are converted into embeddings and searched against a FAISS vector store to retrieve the most relevant context.

The retrieved context is then combined with the user query and sent to the OpenAI API to generate a contextual, natural-language response.
