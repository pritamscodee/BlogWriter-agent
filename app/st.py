import streamlit as st
import os

from crewai import Agent, Task, Crew, Process, LLM




mistral_llm = LLM(
    model="mistral/mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
)



researcher = Agent(
    role="Researcher",
    goal="Research topic deeply {topic}",
    backstory="Researcher with deep knowledge of the topic",
    verbose=True,
   
    llm=mistral_llm
)

writer = Agent(
    role="Writer",
    goal="Write high quality blog about {topic}",
    backstory="Expert blog writer with deep research skills",
    verbose=True,
    llm=mistral_llm,
  
)



research_task = Task(
    description="Research the topic: {topic}",
    agent=researcher,
    expected_output="Key research points"
)

write_task = Task(
    description="Write a blog  ",
    agent=writer,
    context=[research_task],
    expected_output="Full blog post"
)



st.set_page_config(page_title="CrewAI Blog Writer", page_icon="🤖")

st.title("🤖 AI Blog Write ")

topic = st.text_input("Enter topic", placeholder="e.g. AI Agents in 2025")

if st.button("Run Crew"):

    if not topic.strip():
        st.warning("Please enter a topic")

    else:
        with st.spinner("Crew is working..."):

            try:
                crew = Crew(
                    agents=[researcher, writer],
                    tasks=[research_task, write_task],
                    process=Process.sequential,
                    verbose=True,
                    stream=False
                )

                result = crew.kickoff(inputs={"topic": topic})

                st.success("Done!")
                st.markdown("### Generated Blog")
                st.markdown(result)

            except Exception as e:
                st.error(f"Error: {str(e)}")