import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from dotenv import load_dotenv
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your X-ray model
xray_model = load_model("xray_best_model.h5")



# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embedding model and retriever
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="llmapp",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Session state for memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# Gemini response generator
def generate_gemini_response(context, question):
    try:
        history_prompt = ""
        for msg in st.session_state.chat_memory:
            if isinstance(msg, HumanMessage):
                history_prompt += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_prompt += f"Bot: {msg.content}\n"

        prompt = f"{history_prompt}Context: {context}\nUser: {question}\nBot:"
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        if response and response.candidates:
            reply = response.candidates[0].content.parts[0].text
        else:
            reply = "I'm sorry, I couldn't generate a response."

        # Update memory
        st.session_state.chat_memory.append(HumanMessage(content=question))
        st.session_state.chat_memory.append(AIMessage(content=reply))

        return reply
    except Exception as e:
        print("Gemini Error:", e)
        return "There was an error generating the response."


# Streamlit Layout
st.set_page_config(page_title="MediBot", layout="wide")
st.title("💬 MediBot - Medical Chatbot")

st.subheader("📤 Upload Chest X-ray for Diagnosis")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = xray_model.predict(img_array)[0][0]
    diagnosis = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.image(img, caption="Uploaded X-ray", use_column_width=True)
    st.markdown(f"### 🩻 Diagnosis: **{diagnosis}**")
    st.markdown(f"Confidence: `{confidence:.2%}`")



# Chat form 
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask MediBot a question:", "")
    submit_button = st.form_submit_button(label="Send")

# Handle submission
if submit_button and user_input:
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context = " ".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""

    # Generate Gemini response
    response = generate_gemini_response(context=context, question=user_input)

    # Show chat
    st.markdown(f"**🧑 You:** {user_input}")
    st.markdown(f"**🤖 MediBot:** {response}")

# Sidebar chat memory (should come last so memory is up-to-date)
st.sidebar.header("🧠 Conversation Memory")
if st.session_state.chat_memory:
    for msg in st.session_state.chat_memory:
        role = "User" if isinstance(msg, HumanMessage) else "Bot"
        st.sidebar.markdown(f"**{role}:** {msg.content}")
else:
    st.sidebar.write("No memory yet.")
