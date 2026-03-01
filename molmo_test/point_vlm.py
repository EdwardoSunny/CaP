from openai import OpenAI
import base64

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = OpenAI(
    base_url="http://scai3.cs.ucla.edu:8000/v1",
)

print("client started")

# Encode local image
base64_image = encode_image("image.png")

response = client.chat.completions.create(
    model="allenai/Molmo2-8B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Can you point at where the blue cube is"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ],
        }
    ],
    max_tokens=2048,
)

print("response generated")
print(response.choices[0].message.content)
