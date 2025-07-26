import os
import mimetypes
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai
import gradio as gr

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Model config
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 4000,
}

# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Load the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Captioning prompt template
base_prompt = """You are an advanced AI model specializing in generating engaging and contextually relevant captions or post content for social media platforms. 
Given the context and an uploaded image, your task is to create a captivating caption or post content that resonates with the selected social media platform's audience.
Please analyze the provided image and the contextual description carefully. Use the following guidelines based on the social media platform specified:

1. Instagram: Focus on visually appealing, inspirational, and trendy content. Use relevant hashtags.
2. Facebook: Craft engaging and personal stories or updates. Aim for a friendly and conversational tone.
3. Twitter: Create concise, witty, and impactful tweets. Utilize popular hashtags and mentions.
4. LinkedIn: Develop professional and insightful posts. Emphasize expertise, industry relevance, and networking.
5. Pinterest: Write creative and informative descriptions. Highlight the aesthetic and practical aspects.

Output should be one after another caption/post (bulleted in case of more than 1).

The prompted message is: 
"""

# Handle image setup
def input_image_setup(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("Image file not found.")
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "image/jpeg"  # fallback
    
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    return [{"mime_type": mime_type, "data": image_bytes}]

# Main model function
def generate_gemini_response(prompt_text, image_path):
    try:
        image_parts = input_image_setup(image_path)
        prompt_parts = [prompt_text, image_parts[0]]
        response = model.generate_content(prompt_parts)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Combine everything
def process_input(image_file, context_text, platform, num_posts):
    if not image_file:
        return None, "❌ Please upload an image first."

    try:
        platform = platform or "Instagram"
        num_posts = int(num_posts or 1)

        full_prompt = base_prompt + f"\nPlatform: {platform}\nNumber of Captions/Posts: {num_posts}\nContext: {context_text}\n"
        response = generate_gemini_response(full_prompt, image_file)
        return image_file, response
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📸 Captionify - Generate AI Captions from Any Image")

    with gr.Row():
        image_input = gr.File(label="Upload an Image", file_types=["image"], type="filepath")
        image_preview = gr.Image(label="Image Preview")

    with gr.Row():
        context_text = gr.Textbox(label="Context or Description", placeholder="Describe what's happening or what you want the caption to reflect")
        platform_dropdown = gr.Dropdown(choices=["Instagram", "Facebook", "Twitter", "LinkedIn", "Pinterest"], label="Choose Platform")
        num_input = gr.Number(label="How many captions?", value=1)

    generate_button = gr.Button("✨ Generate Captions")
    output_text = gr.Textbox(label="Generated Caption/Post", lines=8)

    image_input.change(lambda f: f, inputs=image_input, outputs=image_preview)
    generate_button.click(
        fn=process_input,
        inputs=[image_input, context_text, platform_dropdown, num_input],
        outputs=[image_preview, output_text]
    )

demo.launch(share=True)

