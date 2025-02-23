import speech_recognition as sr
import pyttsx3
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize the TTS engine
engine = pyttsx3.init()
voice = engine.getProperty('voices')
engine.setProperty('voice', voice[0].id)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate + 50)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize speech recognition
recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, phrase_time_limit=20) # Capture voice input
            text = recognizer.recognize_google(audio) # Convert speech to text
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError:
            print("Speech Recognition service is unavailable.")
            return None

# Chatbot Template
template = """ 
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3") # Sets up which model to use
prompt = ChatPromptTemplate.from_template(template) # Sets the template we made for the model to respond to
chain = prompt | model # Chains the model and the prompt processes together

'''
Handles input from the user and the output from the model and keeps
track of conversation history so that the model can refer back to previous input.
'''
def handle_conversation():
    context = ""
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")

    while True:
        user_input = listen() # Capture voice input

        if user_input is None:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        result = chain.invoke({"context": context, "question": user_input})
        response_text = result.content if hasattr(result, 'content') else str(result)

        print("Bot: ", response_text)
        speak(response_text)

        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__": # Calls handle_conversation() as soon as the main file is run
    handle_conversation()