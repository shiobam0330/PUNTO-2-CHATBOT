import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
from utils import get_context_from_query, custom_prompt, coseno,cosine_similarity,text_embedding
import json
path = "D:/Downloads/INTELIGENCIA ARTIFICIAL/PARCIAL FINAAL/"
df_respuestas = pd.read_excel(path + "preg_llm.xlsx")
file_name = open('credentials.json')
config_env = json.load(file_name)
df_vector_store = pd.read_pickle('df_vector_store.pkl')

def main_page():
  if "temperature" not in st.session_state:
      st.session_state.temperature = 0.0
  if "model" not in st.session_state:
      st.session_state.model = "gpt-3.5-turbo"
  if "message_history" not in st.session_state:
      st.session_state.message_history = []

  with st.sidebar:
    st.image('usta.png', use_container_width="always")
    st.header(body="Chat personalizado :robot_face:")
    st.subheader('Configuración del modelo :level_slider:')

    model_name = st.radio("**Elije un modelo**:", ("GPT-3.5", "GPT-4"))
    if model_name == "GPT-3.5":
      st.session_state.model = "gpt-3.5-turbo"
    elif model_name == "GPT-4":
      st.session_state.model = "gpt-4"
    
    st.session_state.temperature = st.slider("**Nivel de creatividad de respuesta**  \n  [Poco creativo ►►► Muy creativo]",
                                             min_value = 0.0,
                                             max_value = 1.0,
                                             step      = 0.1,
                                             value     = 0.0)
    
  if st.session_state.get('generar_pressed', False):
    for message in st.session_state.message_history:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])
  if prompt := st.chat_input("¿Cuál es tu consulta?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        Context_List = get_context_from_query(query = prompt,
                                              vector_store = df_vector_store,
                                              n_chunks = 5)
        client = OpenAI(api_key=config_env["openai_key"])
        completion = client.chat.completions.create(
          model=st.session_state.model,
          temperature = st.session_state.temperature,
          messages=[{"role": "system", "content": f"{custom_prompt.format(source = str(Context_List))}"}] + 
          st.session_state.message_history + 
          [{"role": "user", "content": prompt}]
        )
        full_response = completion.choices[0].message.content
        message_placeholder.markdown(full_response)

    st.session_state.message_history.append({"role": "user", "content": prompt})
    st.session_state.message_history.append({"role": "assistant", "content": full_response})
    st.session_state.generar_pressed = True

  if 'embedding' not in df_respuestas.columns:
    df_respuestas['embedding'] = df_respuestas['Respuesta'].apply(lambda x: np.array(text_embedding([x])))

  if st.button("Calcular similitud"):
    try:
        if st.session_state.message_history:
            respuesta_chatbot = st.session_state.message_history[-1]['content']
            embedding_generada = np.array(text_embedding([respuesta_chatbot]))
            similitudes = {}
            for idx, row in df_respuestas.iterrows():
                embedding_guardado = row['embedding']
                similitud = coseno(embedding_generada, embedding_guardado)
                similitudes[row['Respuesta']] = similitud

            mejor_respuesta = max(similitudes, key=similitudes.get)
            mejor_similitud = similitudes[mejor_respuesta]

            st.info(f"Similitud: {mejor_similitud:.4f}")
        else:
            st.error("No hay respuesta generada por el chatbot para comparar.")
    except Exception as e:
        st.error(f"Error al calcular similitudes: {e}")
if __name__ == "__main__":
    main_page()