import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖"
)

st.title("🤖 AI Text Generator")
st.write("Enter a prompt and generate text using AI.")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="distilgpt2"
    )

generator = load_model()

prompt = st.text_area(
    "✍️ Enter your prompt:",
    placeholder="Artificial Intelligence is..."
)

if st.button("✨ Generate Text"):

    if prompt.strip():

        with st.spinner("Generating text..."):

            result = generator(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )

        generated_text = result[0]["generated_text"]

        st.subheader("📝 Generated Text")
        st.write(generated_text)

    else:
        st.warning("Please enter a prompt!")