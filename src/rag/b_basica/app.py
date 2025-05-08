"""
Created by Datoscout at 26/03/2024
solutions@datoscout.ec
"""

# External imports
import streamlit as st
from loguru import logger

# Internal imports
from src.config.parameters import MAX_PAGES
from src.rag.b_basica.nlp_proc import compute_embeddings, response_generator
from src.rag.b_basica.utils import extract_context, retrieve_dataframe, store_dataframe

# streamlit run .\src\rag\b_basica\app.py


def main():
    st.title("Extracción de contexto de PDF y chat")
    st.markdown(
        """
        ## 📄 ¡Bienvenido!
        Por favor, <b>proporciona tu documento PDF</b>.<br>
        Será utilizado como contexto en nuestra conversación interactiva.
        <br><br>
        <span style='color: #1049A6; font-weight: bold;'>Esta es una demo:</span> solo se utilizarán las primeras <span style='background-color: #ffe066; color: #333; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>{MAX_PAGES}</span> páginas de tu archivo.
        <br><br>
        <span style='color: #888;'>¡Sube tu archivo para comenzar!</span>
        """.format(
            MAX_PAGES=MAX_PAGES
        ),
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Subir un archivo PDF", type="pdf")
    if "has_context" not in st.session_state:
        st.session_state.has_context = False

    if st.button("Enviar archivo") and not st.session_state.has_context:
        if uploaded_file is not None:
            try:
                file_name = uploaded_file.name
                embds = retrieve_dataframe(file_name)
                if embds.empty:
                    try:
                        final_text = extract_context(uploaded_file)
                        embds = compute_embeddings(final_text)
                        store_dataframe(file_name, embds)
                    except Exception as e:
                        import traceback

                        error_traceback = traceback.format_exc()
                        st.error(f"Error processing file: {str(e)}")
                        st.code(error_traceback, language="python")
                        logger.error(f"Error details: {error_traceback}")
                        return

                st.session_state.embds = embds
                st.session_state.has_context = True
                st.session_state.nb_questions = 0
            except Exception as e:
                st.error(f"An error occurred: {e}")

    if st.session_state.has_context:
        st.title("Assistant Bot")
        st.markdown(
            """
            ### ¡Gracias!
            Ahora, hablemos. Puedes hacer tus preguntas en el mismo idioma que el contexto.<br>
            <ul>
                <li>Haz hasta <b>diez preguntas</b>.</li>
                <li>Usa preguntas <b>precisas</b> sobre el contenido de tu archivo PDF.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

        logger.info(st.session_state.has_context)
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if st.session_state.nb_questions < 11:
            if prompt := st.chat_input("¿Qué es lo que quieres saber?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    stream = response_generator(prompt, st.session_state.embds)
                    response = st.write_stream(stream)
                    # response = st.write_stream(response_random_generator())
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.nb_questions += 1
        else:
            st.text("Fin de la demo.")


if __name__ == "__main__":
    main()
