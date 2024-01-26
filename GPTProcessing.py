from openai import OpenAI


# Load your API key
with open('API_key.txt', 'r') as file:
    api_key = file.readline().strip()

# Set your API key
client = OpenAI(api_key= api_key)

def chat_with_gpt(prompt):

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant for correcting OCR-generated text as plain text. You should concatnate this text into plain text or a manuscript. You should be careful, this is columned newspaper, but OCR text can't recognize it"},
        {"role": "user", "content": f"{prompt}"}
    ]
    )

    return completion.choices[0].message

with open("datasets/extracted_text_page_1.txt", "r") as file:
    file_contents = file.read()

file_contents = file_contents[:10000]

response = chat_with_gpt(file_contents)
#print("Assistant:", response)

with open(f'datasets/refined_text.txt', 'w') as file:
    file.write(response.content)