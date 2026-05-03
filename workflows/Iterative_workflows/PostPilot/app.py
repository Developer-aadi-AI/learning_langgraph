import streamlit as st
from io import BytesIO
from graph import workflow

st.header("PostPilot")

# --- Graph Viewer ---
with st.expander("Show Workflow Graph"):
    graph_image = workflow.get_graph().draw_mermaid_png()
    st.image(BytesIO(graph_image), caption="PostPilot Workflow")

# --- Session state for persistence ---
if "file" not in st.session_state:
    st.session_state.file = None

# --- Chat input ---
topic = st.chat_input("Enter your topic:")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload a file (ipynb, py, pdf) supported : ")

if uploaded_file and not uploaded_file.name.endswith(".ipynb" or ".py" or ".pdf"):
    st.error("(.ipynb, .py, .pdf) files are supported.")
elif uploaded_file is not None:
    st.session_state.file = uploaded_file
    st.success(f"File uploaded: {uploaded_file.name}")

# --- Run only when topic exists ---
if topic:
    initial_state = {
        "topic": topic,
        "file": st.session_state.file
    }
    with st.spinner("Generating Your Post..."):
        try:
            final_state = workflow.invoke(initial_state)
            st.markdown(final_state['post'])
        except Exception as e:
            st.error(f"Something went wrong: {e}")