import os
from dotenv import load_dotenv
from openai import AzureOpenAI

api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = "2025-01-01-preview"

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

def zero_shot_prompting(prompt):
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, 
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during zero-shot prompting: {e}")
        return None
    

def run_chatbot():
    print("Welcome to the Azure OpenAI Chatbot! Type 'exit' to end.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = zero_shot_prompting(user_input)
        # response = analyze_sentiment(user_input)
        if response:
            print(f"Chatbot: {response}")
        else:
            print("Chatbot: Sorry, I couldn't generate a response.")

def analyze_sentiment(text):
    prompt = f"Categorize the following text as either 'positive,' 'negative,' or 'neutral': '{text}'"
    result = zero_shot_prompting(prompt)
    return result

if __name__ == "__main__":
    run_chatbot()