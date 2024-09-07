# 1. Imports and API setup
import gradio as gr
from groq import Groq
import base64
import os

# Define models used in the process
llava_model = 'llava-v1.5-7b-4096-preview'
llama31_model = 'llama-3.1-70b-versatile'

# Image encoding function
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Image to text function
def image_to_text(client, model, base64_image, prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model=model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        if 'Invalid API Key' in str(e):
            return "Please enter a correct API key and try again."
        return f"Error generating text from image: {str(e)}"

# Technical review generation function
def technical_review_generation(client, image_description):
    keywords = ["econometrics", "finance", "marketing", "stock", "prediction", "chart", "graph", "time series"]
    if not any(keyword in image_description.lower() for keyword in keywords):
        return "The image is not related to the area this app covers. Please input a relevant image."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional econometrics. Write a complete review and report about the scene depicted in this image.",
                },
                {
                    "role": "user",
                    "content": image_description,
                }
            ],
            model=llama31_model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

# Main function for Gradio interface
def process_image(api_key, image, prompt="Describe this image in detail."):
    # Set the API key
    try:
        os.environ["GROQ_API_KEY"] = api_key
        client = Groq()  # Initialize the Groq client with the provided key
    except Exception as e:
        return "Please enter a correct API key and try again.", ""

    # Encode the image
    base64_image = encode_image(image)

    # Get image description from the model
    image_description = image_to_text(client, llava_model, base64_image, prompt)

    # If API key was invalid, only return the API key error message
    if "Please enter a correct API key and try again." in image_description:
        return image_description, ""

    # Generate the econometrics report based on the image description
    report = technical_review_generation(client, image_description)

    # Return both image description and the econometrics report
    return f"--- Image Description ---\n{image_description}", f"--- GroqLLaVAMA EconoMind Report ---\n{report}"

# Define CSS for centering elements and footer styling
css = """
    #title, #description {
        text-align: center;
        margin: 20px;
    }
    #footer {
        text-align: center;
        margin-top: 30px;
        padding: 10px;
        font-size: 14px;
    }
    .gradio-container {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .gradio-row {
        width: 100%;
        display: flex;
        justify-content: center;
    }
    .clear-button {
        margin-top: 10px;
    }
"""

# Gradio Interface
def gradio_interface():
    # Define the footer HTML
    footer = """
    <div id="footer">
        <a href="https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/" target="_blank">LinkedIn</a> |
        <a href="https://github.com/arad1367" target="_blank">GitHub</a> |
        <a href="https://arad1367.pythonanywhere.com/" target="_blank">Live demo of my PhD defense</a> |
        <a href="https://groq.com/introducing-llava-v1-5-7b-on-groqcloud-unlocking-the-power-of-multimodal-ai/" target="_blank">Introducing LLaVA V1.5 7B on GroqCloud</a>
        <br>
        Made with 💖 by Pejman Ebrahimi
    </div>
    """

    with gr.Blocks(theme="gradio/soft", css=css) as demo:
        gr.HTML("<h1 id='title'>GroqLLaVAMA Econometrics Agent</h1>")
        gr.HTML("<p id='description'>Upload an economic chart and get a detailed analysis using Groq + LLaVA V1.5 7B multimodal + llama-3.1-70b.</p>")
        
        with gr.Row():
            api_key_input = gr.Textbox(label="GROQ API Key", placeholder="Enter your GROQ API Key", type="password")
        with gr.Row():
            image_input = gr.Image(type="filepath", label="Upload an Image")  # Changed type to 'filepath'
        with gr.Row():
            report_button = gr.Button("Generate Report")
        with gr.Row():
            output_description = gr.Textbox(label="Image Description", lines=10, elem_id="description-box")
            output_report = gr.Textbox(label="Report", lines=10, elem_id="report-box")

        # Define the interaction between inputs and outputs
        report_button.click(
            fn=process_image,
            inputs=[api_key_input, image_input],
            outputs=[output_description, output_report]
        )

        # Add footer HTML
        gr.HTML(footer)
        
        # Add clear button
        def clear_inputs():
            return "", None, "", ""

        with gr.Row():
            clear_button = gr.Button("Clear", elem_id="clear-button")
            clear_button.click(
                fn=clear_inputs,
                inputs=[],
                outputs=[api_key_input, image_input, output_description, output_report]
            )

    # Launch the interface
    demo.launch()

# Start the Gradio interface
gradio_interface()