from typing import Literal
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


load_dotenv()

groq_llm = ChatGroq(
    model_name = "llama-3.3-70b-versatile",
    temperature=0.5
)

llama_llm = ChatOllama(
    base_url="http://localhost:11434",
    model = "llama3.2",
    temperature=0.6
)


class TweetevaluationSchema(BaseModel):
    evaluation: Literal["Approved", "Needs Improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(...,description="feedback for the post.")


eval_llm = groq_llm.with_structured_output(TweetevaluationSchema)


