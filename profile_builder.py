from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import json

st.title('Welcome to the AI Profiler')

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

class ProfileBuilder:
    def __init__(self):
        self.openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.questions = [
            "name",
            "work location",
            "department",
            "seniority level",
        ]

    def ask_question(self):
        """Ask the user the next question and handle their input."""
        if st.session_state.step >= len(self.questions):
            return "END_OF_QUESTIONS"

        question = self.questions[st.session_state.step]
        user_name = st.session_state.user_data.get('name', '')
        city = st.session_state.user_data.get('work location', '')
        dept = st.session_state.user_data.get('department', '')
        level = st.session_state.user_data.get('seniority level', '')

        # Define prompts and labels
        if question == "name":
            prompt_text = """
            You are Hal.
            Let the user know you are looking to build a professional profile about them.
            Be friendly and engaging and ask the user for their name.
            """
            user_input_label = "Write your name here:"

        elif question == "work location":
            prompt_text = f"""
            You are Hal, a helpful assistant helping to gather some information to build a professional profile.
            You know the user's name is {user_name}. You want to ask them what city they work in.
            Compose a question to ask {user_name} what city they work in.
            Keep it short.
            Make sure to acknowledge their name and transition smoothly to the next question.
            Do not make any other assumptions or mention any other information than their name.
            Do not include any greetings, introductory phrases, or personal opinions.
            """
            user_input_label = "Write your work location here:"

        elif question == "department":
            prompt_text = f"""
            You are Hal, a helpful assistant helping to gather some information to build a professional profile.
            Provide an interesting fact about the city of {city}.
            Preface this fact by saying you heard it recently.
            You know the user's name is {user_name} and that they work in {city}.
            You also want to ask them separately what department they work in.
            Compose a question to ask {user_name} about their department.
            Do not include any greetings, introductory phrases, or personal opinions.
            """
            user_input_label = "Write your department here:"

        elif question == "seniority level":
            prompt_text = f"""
            You are Hal, a helpful assistant helping to gather some information to build a professional profile.
            The user just mentioned they work in the {dept} department.
            Provide a positive comment about working in this department.
            You also want to ask them what their seniority level is.
            Compose a question to ask {user_name} about their seniority level in {dept}.
            Do not include any greetings, introductory phrases, or personal opinions.
            """
            user_input_label = "Write your seniority level here:"

        # Generate the question using LLM
        prompt_template = PromptTemplate(template=prompt_text, input_variables=["user_name", "city", "dept", "level"])
        question_chain = LLMChain(llm=self.openai, prompt=prompt_template)
        
        try:
            # Only generate the question if it's the first time for the current step
            if 'current_question' not in st.session_state or st.session_state.current_question != question:
                response = question_chain.invoke({"user_name": user_name, "city": city, "dept": dept, "level": level}, model="gpt-3.5-turbo")
                st.session_state.current_question = question  # Save current question to state
                st.session_state.question_text = response['text'].strip().strip('"')

            # Display the question
            st.write(st.session_state.question_text)

            # Input field for the user's response
            answer = st.text_input(user_input_label, key=f"input_{st.session_state.step}")

            # Submit button
            if st.button("Submit", key=f"submit_{st.session_state.step}"):
                if answer.strip():  # Ensure non-empty input
                    # Save the response
                    st.session_state.user_data[question] = answer.strip()

                    # Clear the current question for the next step
                    del st.session_state.current_question

                    # Increment the step to move to the next question
                    st.session_state.step += 1

                    # Trigger a rerun to load the next question
                    st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")

    def save_profile_to_json(self):
        """Save the user profile to a JSON file."""
        try:
            json_data = json.dumps(st.session_state.user_data, indent=4)
            st.download_button(
                label="Download Profile",
                data=json_data,
                file_name='user_profile.json',
                mime='application/json'
            )
            st.success("Your profile has been saved successfully!")
        except Exception as e:
            st.error(f"An error occurred while saving the profile: {e}")

    def build_profile(self):
        """Guide the user through the profile creation process."""
        if st.session_state.step < len(self.questions):
            self.ask_question()
        else:
            st.write(f"Thank you, {st.session_state.user_data.get('name', 'User')}! Your profile is complete.")
            self.save_profile_to_json()
            st.json(st.session_state.user_data)

# Create an instance of ProfileBuilder
builder = ProfileBuilder()

# Start the profiling process
builder.build_profile()