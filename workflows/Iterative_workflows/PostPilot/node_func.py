from langchain_core.messages import HumanMessage, SystemMessage
import nbformat
from states import PGState
from llms  import llama_llm, eval_llm, groq_llm
import io


def parse_notebook(file):    
    def read_file(file):
        # Read bytes from UploadedFile and wrap in BytesIO for nbformat
        file.seek(0)
        file_bytes = file.read()
        file_like = io.BytesIO(file_bytes)
        nb = nbformat.read(file_like, as_version=4)
        
        notebook_content = []
        for i, cell in enumerate(nb.cells):
            cell_data = {
                "cell_index": i,
                "type": cell.cell_type,
                "content": "".join(cell.source).strip()
            }
            if cell.cell_type == "code":
                outputs = []
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        outputs.append(output.get("text", ""))
                    elif output.output_type == "execute_result":
                        data = output.get("data", {})
                        if "text/plain" in data:
                            outputs.append(data["text/plain"])
                    elif output.output_type == "display_data":
                        data = output.get("data", {})
                        if "text/plain" in data:
                            outputs.append(data["text/plain"])
                    elif output.output_type == "error":
                        outputs.append("\n".join(output.get("traceback", [])))
                cell_data["outputs"] = "\n".join(outputs).strip()
            notebook_content.append(cell_data)
        return notebook_content

    def notebook_to_text(file):
        notebook_content = read_file(file)
        text_output = []
        for cell in notebook_content:
            if cell['type'] == "markdown":
                text_output.append(f"[markdown]\n{cell['content']}\n")
            elif cell['type'] == "code":
                text_output.append(f"[code]\n{cell['content']}\n")
                if cell.get("outputs"):
                    text_output.append(f"[output]\n{cell['outputs']}\n")
        return "\n".join(text_output)

    notebook_context = notebook_to_text(file)
    return notebook_context


def get_context(state: PGState):
    file = state.file
    notebook_context = parse_notebook(file)  # pass file object directly
    
    prompt = f"""
        You are an expert data analyst.

        Below is a Jupyter notebook content with markdown, code, and outputs.

        Your job is to deeply understand what is happening.

        Step 1: Explain the notebook step-by-step in simple terms.
        Step 2: Identify the goal of the notebook.
        Step 3: Extract the workflow (sequence of steps).
        Step 4: Identify key results from outputs.
        Step 5: Highlight important insights.

        Notebook:
        {notebook_context}

        Return output in this structured format:

        GOAL:
        ...

        WORKFLOW:
        - step 1
        - step 2

        KEY STEPS:
        ...

        RESULTS:
        ...

        INSIGHTS:
        ...
    """  # your prompt unchanged

    context = llama_llm.invoke(prompt).content
    return {"context": context}


def initialize(state: PGState):
    return {
        "iteration": 0,
        "max_iteration": 5
    }


def MakePost(state: PGState):

    messages = [
        SystemMessage(content="You are a clear, thoughtful LinkedIn creator who shares learning journeys   and projects in a professional, grounded, and easy-to-understand way."),
        
        HumanMessage(content=f"""
        Write a LinkedIn post about my learning journey or a project I recently built on the topic: "{state.topic}".

        Context you can use (if provided):
        {state.context or ''}

        Rules:
        - Keep the tone professional, simple, and reflective
        - Focus on what I learned, built, or explored
        - Explain the project or concept clearly in plain English
        - Highlight key features or takeaways (use bullet points if useful)
        - Avoid hype, exaggeration, or buzzwords
        - Use short paragraphs for readability
        - Do not overuse emojis (prefer none)
        - End with a brief forward-looking or reflective closing
        - Add 3-5 relevant hashtags

        Strict Output Rules:
        - Output ONLY the final post content
        - Do NOT add any introduction like "Here is your post"
        - Do NOT add any explanation before or after the post
        - Do NOT mention AI, generation, or writing process
        - Do NOT use quotes around the post

        Structure:
        1. What I started learning or building
        2. What I built or explored
        3. Key takeaways or features
        4. Reflection or next steps
        5. Hashtags
        """)
    ]

    post = llama_llm.invoke(messages).content

    return {'post': post}



def EvaluatePost(state: PGState):
    # Prompt
    messages = [
        SystemMessage(content="You are a sharp, no-nonsense LinkedIn content critic. You evaluate posts based on clarity, authenticity, structure, and professional value."),
        
        HumanMessage(content=f"""
            Evaluate the following LinkedIn post:

            Post:
            {state.post}

            Context (if available):
            {state.context or 'No context provided.'}

            ---

            Evaluate on these criteria:

            1. Clarity — Is the idea explained clearly and easy to follow?
            2. Authenticity — Does it feel genuine and grounded, not generic?
            3. Value — Does it provide meaningful insights or takeaways?
            4. Structure — Is it well-organized with good flow and readability?
            5. Engagement Potential — Would this make someone pause and read?
            6. Context Coverage — If context was provided, does the post cover 
            at least 50% of the key points from it? If no context, skip this.

            Auto-reject if ANY of these are true:
            - It is overly generic with no real insight
            - It uses excessive buzzwords or hype language
            - It lacks a clear learning, takeaway, or outcome
            - It is poorly structured or hard to read
            - It feels self-promotional without substance
            - Context was provided but fewer than half the key points are reflected in the post

            Respond ONLY in structured format:
            - evaluation: "Approved" or "Needs Improvement"
            - feedback: One paragraph covering strengths, weaknesses, and specifically 
            which context points are missing (if any)
        """)
    ]

    # llm result
    result = eval_llm.invoke(messages)
    # save it
    return {'evaluation': result.evaluation, 'feedback': result.feedback}


def OptimizePost(state: PGState):
    messages = [
        SystemMessage(content="You refine LinkedIn posts to improve clarity, authenticity, structure, and professional value based on feedback."),
        
        HumanMessage(content=f"""
            Improve the LinkedIn post based on this feedback:
            "{state.feedback}"

            Topic: "{state.topic}"

            Original Post:
            {state.post}

            Rewrite it as a clear, well-structured LinkedIn post.

            Rules:
            - Keep the tone professional, simple, and reflective
            - Improve clarity and flow
            - Make the learning or project more concrete and specific
            - Strengthen key takeaways or insights
            - Avoid buzzwords, hype, or generic statements
            - Use short paragraphs and bullet points if helpful
            - Keep it concise but complete
            - Do not overuse emojis (prefer none)
            - End with a thoughtful closing line
            - Add 3-5 relevant hashtags

            Strict Output Rules:
            - Output ONLY the final post content
            - Do NOT add any introduction like "Here is your post"
            - Do NOT add any explanation before or after the post
            - Do NOT mention AI, generation, or writing process
            - Do NOT use quotes around the post
        """)
    ]

    result = groq_llm.invoke(messages).content
    
    return {'post': result, "iteration": state.iteration+1}

