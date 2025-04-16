import streamlit as st
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Example intents and questions
intents = {
    "recommendations": [
        "Can you recommend a good beach destination for a family vacation?",
        "I'm looking for suggestions on romantic getaways for couples.",
        "What are some popular outdoor adventure vacation ideas?",
    ],
    "how-to": [
        "How do I book a vacation package?",
        "What is the process for getting a travel visa?",
        "How can I find the best flight deals?",
    ],
    "locality": [
        "What are some attractions near Paris?",
        "Are there good restaurants in Rome?",
        "What's the weather like in Bali?",
    ]
}

# Prepare few-shot prompt examples
intent_examples = []
for intent, examples in intents.items():
    for q in examples:
        intent_examples.append(f"Q: {q}\nIntent: {intent}")

few_shot_examples = "\n\n".join(intent_examples)

def detect_intent(user_query):
    prompt = (
        f"You are an intent classification assistant for travel queries.\n"
        f"Classify the intent of the following question into one of these: "
        f"{', '.join(intents.keys())}.\n\n"
        f"Here are some examples:\n"
        f"{few_shot_examples}\n\n"
        f"Q: {user_query}\nIntent:"
    )
    # Use Gemini to classify intent
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    # Extract the intent from Gemini's response
    return response.text.strip().split()[0]  # Assumes intent is first word

def generate_response(user_query, detected_intent):
    prompt = (
        f"You are a helpful travel assistant. Respond to the user's question in 2-3 sentences.\n"
        f"Question: {user_query}\n"
        f"Intent: {detected_intent}\n"
        f"Respond helpfully and concisely."
    )
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Gemini-Only Semantic Router RAG POC")
user_query = st.text_input("Ask your travel question:")
response_container = st.empty()

if user_query:
    detected_intent = detect_intent(user_query)
    response = generate_response(user_query, detected_intent)
    response_container.markdown(f"""
    **Detected Intent**: `{detected_intent}`  
    **Response**: {response}
    """)
