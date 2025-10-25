from langchain_google_genai import ChatGoogleGenerativeAI

def get_model(role: str):
    if role == "generator":
        return ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.8)
    elif role in ["checker", "summarizer", "classifier"]:
        return ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.8)
    else:
        return ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.2)
