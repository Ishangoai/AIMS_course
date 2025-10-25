from langchain_google_genai import ChatGoogleGenerativeAI

def get_model(role: str):
    if role == "generator":
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    elif role in ["checker", "summarizer", "classifier"]:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    else:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
