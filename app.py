from groq import Groq
import os
import hal9 as h9
import json
import openai
import requests
import replicate

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
import shutil

def story_writer(prompt):
  """
  Writes a short story about an input prompt
  """
  client = openai.AzureOpenAI(
    azure_endpoint = 'https://openai-hal9.openai.azure.com/',
    api_key = os.environ['OPENAI_AZURE'],
    api_version = '2023-05-15',
  )

  system = """
  You are a short story writer.
  """

  messages = h9.load("streamlit-messages", [{"role": "system", "content": system}])
  messages.append({"role": "user", "content": prompt})

  completion = client.chat.completions.create(model = "gpt-4", messages = messages, stream = True)
  response = h9.complete(completion, messages)

  return response

def generic_reply(prompt):
   """
   Reply to the conversation to the user with something that is not an image description or the image from the stable difussion model
     'prompt' to respond to.

   """

   messages = h9.load("messages", [])
   messages = [msg for msg in messages if ("tool_calls" not in msg and "tool_call_id" not in msg)]

   response = Groq().chat.completions.create(
     model = "llama3-70b-8192",
     messages = messages,
     temperature = 0,
     seed = 1)

   stream = Groq().chat.completions.create(model = "llama3-70b-8192", messages = messages, temperature = 0, seed = 1, stream = True)

   response = ""
   for chunk in stream:
     if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None: 
       print(chunk.choices[0].delta.content, end="")
       response += chunk.choices[0].delta.content

   return response

def flux_image(prompt, filename):
    os.environ["REPLICATE_API_TOKEN"] = "XXX"  # Make sure to replace "xxx" with your actual token
    output = replicate.run("black-forest-labs/flux-dev", input={"prompt": prompt})
    image_url = output[0]
    
    # Fetch the image content from the URL
    response = requests.get(image_url)
    response.raise_for_status()  # Check if the request was successful
    
    # Open the image and save it to the specified filename
    image = Image.open(BytesIO(response.content))
    image.save(filename)
    return image

def create_image(filename, prompt):
    """
    Creates an image or photograph for the user based on the given prompt.
    - 'prompt' is the description of the image or photograph.
    - 'filename' is a descriptive filename with the .png extension for the generated image or photograph. Always use PNG extension
    The function also stores the image in a 'storage' directory.
    """
    print("Generating image, please wait...")
    try:
        # Use ThreadPoolExecutor to download images in parallel
        with ThreadPoolExecutor() as executor:
            future = executor.submit(flux_image, prompt, filename)
            image = future.result()
        
        # Copy the saved image to the storage directory
        shutil.copy(filename, f"storage/{filename}")
        
    except Exception as e:
        print(f"An error occurred: {e}")


MODEL = "llama3-70b-8192"
def run(messages, tools):
  return Groq().chat.completions.create(
    model = MODEL,
    messages = messages,
    temperature = 0,
    seed = 1,
    tools=tools,
    tool_choice="auto")

prompt = input("")
h9.event('prompt', prompt)

messages = h9.load("messages", [{ "role": "system", "content": """This is a chatbot that generates images using the new Flux stable diffusion model. It helps the user write a better description of the image that is being generated and then generates it, even with modifications from the user. It can also hold casual generic conversations. When you generate filenames you always use the .png, for example "image.png" extension """ }])
messages.append({"role": "user", "content": prompt})
h9.save("messages", messages, hidden=True)

all_tools = [
  create_image,
  story_writer,
  generic_reply
]

tools = h9.describe(all_tools, model = "llama")
completion = run(messages, tools)

try:
  h9.complete(completion, messages = messages, tools = all_tools, show = False, model = "llama")
except Exception as e:
  one_tool = h9.describe([generic_reply], model = "llama")
  completion = run(messages, one_tool)
  h9.complete(completion, messages = messages, tools = [generic_reply], show = False, model = "llama")

h9.save("messages", messages, hidden=True)
