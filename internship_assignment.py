

# In[168]:
import streamlit as st
st.set_page_config(page_title="Analytical Report Chat", layout="wide")


# from dotenv import load_dotenv

# _ = load_dotenv()


# In[169]:


from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langgraph.types import Command, interrupt

# memory = SqliteSaver.from_conn_string(":memory:")
memory = InMemorySaver()


# In[170]:


class AgentState(TypedDict):
    task: str
    clarification_question: str
    clarification: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


# In[171]:

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash-preview-04-17", model_provider="google_genai", temperature=0,
                        api_key=st.secrets["GEMINI_API_KEY"])


# In[172]:


CLARIFICATION_PROMPT = """ You are an analytical assistant that helps users clarify their analysis report tasks. You 
will be given a task description, and your job is to ask specific, thoughtful follow-up questions that help the user refine and focus their original task.  

Ask the below questions to the user to clarify their task.:
- Focus area or domain (e.g., technical, historical, market analysis)
- Time period or scope
- Type of analysis (e.g., comparison, trend, causal)
- Output format preference (e.g., markdown, text, pdf)
- Preferred sources or data types

Ask atleast all of these question and ask them concisely(important). And Make sure any other questions you ask are relevant to the user's input and avoid repeating what the user has already specified.
"""


PLAN_PROMPT = """
You are an expert business writer. Your task is to create a **high-level outline** for a **5-6 page analysis report** on the topic provided by the user. 

**Important:**
- Your output should be an **outline only**, not the full report.
- The outline should be structured to guide the writing of a 5-6 page report in total length.
- Include bullet points or section-level notes where needed to indicate the intended content and depth.
- Incorporate the user's clarification into relevant sections of the outline.

Your output **must include** the following standard sections of the report:
1. Business Overview  
2. Competitive Landscape  
3. Financial Performance  
4. Valuation  
5. Board Composition & Risk  
6. Competitive Advantage  
7. Growth Prospects & Strategy  

Use the information below to guide the outline structure and content:
------------------------------------
Clarification Question: {clarification_question}  
Clarification Provided: {clarification}
"""


WRITER_PROMPT = """
You are an assistant tasked with writing a professional 5-6 page business analysis report.

Instructions:
- Ensure that the report is maximum 6 pages in length.
- Use the provided plan (outline) to structure the report section by section. Follow the section order
  and headings in the outline exactly, unless the user requests otherwise.
- Use the reference documents to support factual content.
- Respond to user feedback by improving your previous draft when a critique is provided.
- Maintain a clear, analytical tone suitable for business and investment professionals.

For generating the pdf for the report, the following function is used on the text:
--------------------------------------------------------------------------
**IMPORTANT**: Use this only to estimate the length of the report, nothing else.
def generate_pdf(draft_markdown: str) -> bytes:
    html = markdown(draft_markdown)
    return HTML(string=html).write_pdf()
--------------------------------------------------------------------------

Provided Information:
- Outline Plan: {plan}
- Reference Documents: {content}
- Critique (if any): {critique}
- Previous Draft (if any): {draft}
"""


REFLECTION_PROMPT = """
You are a teacher evaluating a business analysis report draft. You will be provided with a draft of the report.

Your task:
- Provide clear and constructive critique on the structure, clarity, depth, and quality of the draft.
- Include specific suggestions for improvement (e.g. missing points, weak arguments, incorrect tone).
- Mention if any sections are too brief or too long.
- Comment on writing style and whether it fits a business/investment audience.

Be direct and helpful. Your feedback will be used to revise and improve the draft.
"""


RESEARCH_PLAN_PROMPT = """
You are a business researcher. Your task is to create a focused list of search queries that will help gather relevant information for writing the analysis report described below.

IMPORTANT:
- Your output should be a **list of search queries** and not the full report.
- Generate a **maximum of 10 high-quality queries**.
- Your goal is to **support the creation of a well-informed 5-6 page report**.
- Use the plan (outline) provided to ensure coverage of all key sections.

Use the information given below to guide your queries:
-------------------------------------
plan = {plan}
"""

RESEARCH_CRITIQUE_PROMPT = """
You are a researcher supporting the revision of a business analysis report.

Task:
- Based on the critique provided, generate up to 10 highly relevant search queries.
- Its important that you only generate 10 queries maximum. 
- The queries should help gather information necessary to improve weak or missing parts of the report.
- Ensure that your queries are aligned with the specific gaps or revision suggestions mentioned in the critique.

Your goal is to assist the writer in making factual, well-supported improvements to the report.
"""



from langchain_core.pydantic_v1 import BaseModel

class Queries(BaseModel):
    queries: List[str]


# In[179]:


from tavily import TavilyClient
import os
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
# tavily.search(query="whats the weather in NY", max_results=2)


# In[181]:


def clarification_node(state: AgentState):
    messages = [
        SystemMessage(content=CLARIFICATION_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {
        "clarification_question": response.content
    }


# In[182]:


def plan_node(state: AgentState):
    messages = [
        SystemMessage(
            content=PLAN_PROMPT.format(
                clarification_question=state['clarification_question'],
                clarification=state['clarification']
            )
        ), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}


# In[183]:


def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(
            content=RESEARCH_PLAN_PROMPT.format(
                plan=state['plan']
            )
        ),
        HumanMessage(content=state['task'])
    ])
    # content = state['content'] or []
    content = state.get('content', [])
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


# In[184]:


def generation_node(state: AgentState):
    # content = "\n\n".join(state['content'] or [])
    draft = state.get('draft', "")
    critique = state.get('critique', "")
    user_message = HumanMessage(
        content=f"{state['task']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(plan=state['plan'] ,content=state['content'], critique=critique,
                                         draft=draft)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }


# In[185]:


def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


# In[186]:


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state.get('content', [])
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


# In[187]:


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


# In[188]:


builder = StateGraph(AgentState)


# In[189]:


builder.add_node("clarification_node", clarification_node)
builder.add_node("planner", plan_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)


# In[190]:


# builder.set_entry_point("planner")
builder.set_entry_point("clarification_node")


# In[191]:


builder.add_edge("clarification_node", "planner")
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_conditional_edges(
    "generate",
    should_continue, 
    {END: END, "reflect": "reflect"}
)
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")


# In[192]:


graph = builder.compile(checkpointer=memory)


# In[193]:


from fpdf import FPDF
import io

# def generate_pdf_from_text(text: str) -> bytes:
#     # 1. Build the PDF in memory
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.set_font("Arial", size=12)

#     for line in text.split('\n'):
#         pdf.multi_cell(0, 10, line)

#     pdf_str = pdf.output(dest='S')  
#     pdf_bytes = pdf_str.encode('latin1')  

#     return pdf_bytes


st.title("AI Analytical Report Generator")

if 'history' not in st.session_state:
    st.session_state.history = []
if 'thread' not in st.session_state:
    st.session_state.thread = {"configurable": {"thread_id": "clarify-thread"}}
if 'clar_q' not in st.session_state:
    st.session_state.clar_q = None
if 'has_clar_q' not in st.session_state:
    st.session_state.has_clar_q = False
if 'stream_steps' not in st.session_state:
    st.session_state.stream_steps = []
if 'revision_number' not in st.session_state:
    st.session_state.revision_number = 1
if 'max_revisions' not in st.session_state:
    st.session_state.max_revisions = 2

for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['text'])


if (not st.session_state.has_clar_q) and (not st.session_state.stream_steps):
    user_task = st.chat_input("Enter your analysis task…")
    if user_task:
        st.session_state.history.append({'role': 'user', 'text': user_task})
        with st.chat_message("user"):
            st.markdown(user_task)
        clarification_question = None
        for step in graph.stream(
            {"task": user_task, 
             "clarification": "", 
             "content": [],
             "max_revisions": 2,
             "revision_number": 2},
            st.session_state.thread,
            interrupt_after=["clarification_node"]
        ):
            # clar = step.get('clarification_node', {}).get('clarification_question')
            node = step.get("clarification_node", {})
            clarification_question = node.get("clarification_question")
            print(clarification_question)
            if clarification_question:
                st.session_state.clar_q = clarification_question
                st.session_state.has_clar_q = True
                st.session_state.revision_number = step.get("revision_number", 1)
                st.session_state.max_revisions = step.get("max_revisions", 2)
                break
            else:
                raise RuntimeError("Stream never yielded a clarification question!")
        st.session_state.history.append({'role': 'assistant', 'text': st.session_state.clar_q})
        st.rerun()

elif (st.session_state.has_clar_q) and (not st.session_state.stream_steps):
    clar_input = st.chat_input("Please clarify…")
    if clar_input:
        st.session_state.history.append({'role': 'user', 'text': clar_input})
        # for msg in st.session_state.history:
        with st.chat_message('user'):
            st.markdown(clar_input)
        for step in graph.stream(
            {
                "task": st.session_state.history[0]['text'],
                "clarification": clar_input,
                "clarification_question": st.session_state.clar_q,
                "content": [],
                "revision_number": 1,
                "max_revisions": 2
            },
            st.session_state.thread
        ):
            print("---------------------------------------------------------")
            print("STEP", step)
            print('draft' in step)
            print('critique' in step)
            print("---------------------------------------------------------")
            st.session_state.revision_number = step.get("revision_number", 1)
            st.session_state.max_revisions = step.get("max_revisions", 2)
            print("NUMBERS", st.session_state.revision_number, st.session_state.max_revisions)
            if('planner' in step):
                with st.chat_message('assistant'):
                    st.markdown(f"**Plan:**\n{step['planner']['plan']}", unsafe_allow_html=True)
                    st.session_state.history.append({'role': 'assistant', 'text': step['planner']['plan']})
            # if('generate' in step):
            #     with st.chat_message('assistant'):
            #         st.markdown(step['generate']['draft'], unsafe_allow_html=True)
                    # st.session_state.history.append({'role': 'assistant', 'text': step['generate']['draft']})
            if 'generate' in step:
                draft = step['generate']['draft']
                
                # Display in chat
                with st.chat_message('assistant'):
                    st.markdown(draft, unsafe_allow_html=True)

                if (draft) and (step['generate']['revision_number'] == 3):
                    
                    from markdown2 import markdown
                    from weasyprint import HTML

                    def generate_pdf(draft_markdown: str) -> bytes:
                        html = markdown(draft_markdown)
                        return HTML(string=html).write_pdf()

                    st.download_button(
                        "📄 Download Report as PDF",
                        data=generate_pdf(draft),
                        file_name="analysis_report.pdf",
                        mime="application/pdf"
                    )



            if('research_critique' in step):
                with st.chat_message('assistant'):
                    for i, content in enumerate(step['research_critique']['content']):
                        st.markdown(f"**Research Content for Critique {i+1}:**\n{content}", unsafe_allow_html=True)
                    st.session_state.history.append({'role': 'assistant', 'text': step['research_critique']['content']})
            if ('research_plan' in step):
                with st.chat_message('assistant'):
                    for i, content in enumerate(step['research_plan']['content']):
                        st.markdown(f"**Research Content {i+1}:**\n{content}", unsafe_allow_html=True)
                    st.session_state.history.append({'role': 'assistant', 'text': step['research_plan']['content']})
            if ('reflect' in step):
                with st.chat_message('assistant'):
                    st.markdown(step['reflect']['critique'], unsafe_allow_html=True)
                    st.session_state.history.append({'role': 'assistant', 'text': step['reflect']['critique']})
            if 'critique' in step:
                with st.chat_message('assistant'):
                    st.markdown(f"**Critique & Recommendations:**\n{step['critique']}", unsafe_allow_html=True)
                    st.session_state.history.append({'role': 'assistant', 'text': step['critique']})
        st.session_state.has_clar_q = False
        st.session_state.stream_steps = st.session_state.history.copy()

else:
    st.stop()