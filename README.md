**High level diagram**
A user enters a task in the Streamlit UI chat input, the Clarification agent asks questions tailored to the task, and after the human provides the input again(i.e. once clarified), the Planner agent builds an outline, and the Researcher agent then fetches sources relevant for the task(dynamic behaviour). Then the Writer agent drafts the report, and the Critic agent reviews this draft and asks for revisions, which the research_critique uses to find new sources and the cycle repeats until the max revision limit is reached, then a PDF is generated which can be downloaded.

<p align="center">
  <img  src="https://github.com/user-attachments/assets/824cc4ca-84e8-41f4-a21f-7b54028f2ceb">
</p>

