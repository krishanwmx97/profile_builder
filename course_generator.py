import json
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os
import random
import streamlit as st

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI with API key
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Available topics for training (hardcoded)
topics = ["Conflicts of interest", "Anti-Bribery and Corruption", "Data Protection", "Speaking Up"]  # remove all but 1 to just have that topic

# Load user profile from JSON file
def load_user_profile():
    try:
        with open('user_profile.json', 'r') as f:
            user_data = json.load(f)
        return user_data
    except FileNotFoundError:
        st.error("User profile not found!")
        return {}

# Function to validate user input against available options using OpenAI
def get_best_match(input_text, options):
    prompt = f"""
    You are a helpful assistant. The user has entered the following input: "{input_text}"
    There are several possible options: {', '.join(options)}.
    Return the option that best matches the input. If the input does not match any of the options well, return 'That is not one of the options, please choose again.'.
    """
    prompt_template = PromptTemplate(template=prompt, input_variables=["input_text", "options"])
    
    # Using RunnableSequence instead of LLMChain
    prompt_chain = (prompt_template | openai)
    result = prompt_chain.invoke({"input_text": input_text, "options": options})
    return result.strip()

# Function to generate a personalized welcome message after the topic is selected
def generate_welcome_message(user_name, user_profile):
    welcome_message = f"""
    Hello {user_name}, welcome to your personalized compliance training!\n
    Here are your available topics :
    - Conflicts of interest
    - Anti-Bribery and Corruption 
    - Data Protection 
    - Speaking Up 
    """
    return welcome_message.strip()

# Function to generate the personalized training introduction for the selected topic
def generate_training_intro(topic):
    prompt_text = f"""
    Generate an introduction message for the user to begin their training on {topic}. 
    The introduction should explain the importance of the topic, how it relates to professional ethics, and what the user can expect during the training.
    The introduction should be no longer than two paragraphs.
    """
    prompt_template = PromptTemplate(template=prompt_text, input_variables=["topic"])

    intro_chain = (prompt_template | openai)
    intro = intro_chain.invoke({"topic": topic})
    return intro.strip()

# Function to generate a personalized scenario for the selected topic
def generate_scenario(topic, user_profile):
    city = user_profile.get("work location", "a city")
    division = user_profile.get("department", "a department")
    seniority = user_profile.get("seniority level", "a seniority level")
    
    if topic == "Conflicts of interest":
        prompt_text = f"""
        This scenario is in the area of Conflicts of Interest.
        Create a character that is currently a {seniority} working in the {division} in {city}. The character's first name should be randomly generated, ethnically diverse and either male or female. Only use the character's first name, do not refer explicitly to their gender.
        Start with a sentence giving the character, their position and their role.
        Create a scenario in the present tense where the character faces a conflict of interest in the character's role, to do with giving a contract to a company owned by a family member.
        Describe the situation in no more than two paragraphs, highlighting the ethical dilemma the character faces.
        """
    elif topic == "Anti-Bribery and Corruption":
        prompt_text = f"""
        This scenario is in the area of Anti-Bribery and Corruption.
        Create a character that is currently a {seniority} working in the {division} in {city}. The character's first name should be randomly generated, ethnically diverse and either male or female. Only use the character's first name, do not refer explicitly to their gender.
        Start with a sentence giving the character, their position and their role.
        Create a scenario in the present tense where the character must decide whether to accept a bribe or act in compliance with company policies.
        Describe the situation in no more than two paragraphs, highlighting the ethical dilemma the character faces.
        """
    elif topic == "Data Protection":
        prompt_text = f"""
        This scenario is in the area of Data Protection.
        Create a character that is currently a {seniority} working in the {division} in {city}. The character's first name should be randomly generated, ethnically diverse and either male or female. Only use the character's first name, do not refer explicitly to their gender.
        Start with a sentence giving the character, their position and their role.
        Create a scenario in the present tense where the character must ensure compliance with data protection laws while dealing with a potential breach.
        The dilemma should involve balancing compliance and the potential consequences of inaction.
        Describe the situation in no more than two paragraphs, highlighting the ethical dilemma the character faces.
        """
    elif topic == "Speaking Up":
        prompt_text = f"""
        This scenario is in the area of Speaking Up.
        Create a character that is currently a {seniority} working in the {division} in {city}. The character's first name should be randomly generated, ethnically diverse and either male or female. Only use the character's first name, do not refer explicitly to their gender.
        Start with a sentence giving the character, their position and their role.
        Create a scenario in the present tense where the character witnesses unethical behavior in the workplace and needs to decide whether to speak up.
        The situation should involve pressure from colleagues or a supervisor to keep quiet.
        The scenario should be no longer than two paragraphs, highlighting the ethical dilemma the character faces. and the potential consequences of speaking up versus staying silent.
        """
    
    prompt_template = PromptTemplate(template=prompt_text, input_variables=["topic", "user_profile"])

    scenario_chain = (prompt_template | openai)
    scenario = scenario_chain.invoke({"topic": topic, "user_profile": user_profile})
    return scenario.strip()

# Function to generate a multiple-choice question based on the scenario
def generate_question(scenario, topic):
    prompt_text = f"""
    Based on the following scenario about {topic}, generate a multiple-choice question:
    Scenario: "{scenario}"
    The question should be about the appropriate course of action for the dilemma.
    Provide three options (A, B, C), each output on a different line, with one correct answer and two incorrect but plausible options.
    The correct answer should be a compliant action.
    Do not mention the correct answer in the question. Just provide the options.
    """
    prompt_template = PromptTemplate(template=prompt_text, input_variables=["scenario", "topic"])

    question_chain = (prompt_template | openai)
    question = question_chain.invoke({"scenario": scenario, "topic": topic})
    return question.strip()

# Start training process in Streamlit
def start_training():
    user_profile = load_user_profile()

    if not user_profile:
        st.error("Unable to load user profile. Please ensure it exists.")
        return

    user_name = user_profile.get("name", "User")
    st.write(generate_welcome_message(user_name, user_profile))

    # Topic selection
    topic = st.selectbox("Please choose the topic for your training:", topics)
    if topic:
        st.success(f"You've chosen: {topic}")

    # Training introduction
    if st.button("Start Training"):
        st.write(generate_training_intro(topic))

        # Generate a scenario and question
        scenario = generate_scenario(topic, user_profile)
        st.write(f"**Scenario:**\n\n")
        st.write(f"{scenario}")

        question = generate_question(scenario, topic)
        st.write(f"**Question: What would you advise?**\n\n")
        st.write(f"{question}")        

# Run training
start_training()