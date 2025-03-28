
import openai
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

def chat_completion(prompt):
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, 
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during chat_completion: {e}")
        return None

def few_shot_prompt(user_input):
    examples = [
        "Generate a creative name for a coffee shop with a cozy atmosphere.",
        "Output: 'The Warm Bean'",
        "Generate a creative name for a coffee shop with a modern, minimalist design.",
        "Output: 'Pixel Brew'",
        "Generate a creative name for a coffee shop that specializes in exotic coffee beans.",
        "Output: 'Global Grind'",
    ]
    prompt_lines = examples + [
        f"Generate a creative name for a coffee shop: {user_input}",
        "Output:",
    ]

    prompt = "\n".join(prompt_lines)
    return chat_completion(prompt)


def few_shot_CoT(user_input):
   
   examples = [
       "A farmer has 30 sheep. All but 10 die. how many are left?",
       "Output: 30 - 10 = 20 died and 10 are left",
       "A Boy has 20 mongo. He ate 5. How many are left?",
       "Output: 20 - 5 = 15",
   ]
   prompt = "\n".join(examples + [f"Calculate the following: {user_input}", "Output:"])
   return chat_completion(prompt)

def instruction_based(user_input):
   
   examples = [
       "A farmer has 30 sheep. All but 10 die. how many are left?",
       "Output: 30 - 10 = 20 died and 10 are left",
       "A Boy has 20 mongo. He ate 5. How many are left?",
       "Output: 20 - 5 = 15",
   ]
   prompt = "\n".join(examples + [f"Calculate the following: {user_input}", "Output:"])
   return chat_completion(prompt)


def zero_shot_CoT(user_input):
   
   prompt = f"{user_input} Let's think step by step"
   return chat_completion(prompt)
    
    



def run_chatbot():
   print("Welcome to the Azure OpenAI Chatbot! Type 'exit' to end.")

   while True:
         user_input = input("You: ")
         if user_input.lower() == "exit":
                print("Goodbye!")
                break
             
        #  response = few_shot_prompt(user_input) 
         response = zero_shot_CoT(user_input)
        #  response = few_shot_CoT(user_input)
       
         if response:
            print(f"Chatbot: {response}")
         else: 
            print("Chatbot: Sorry, I couldn't generate a response.")

if __name__ == "__main__":
    run_chatbot()