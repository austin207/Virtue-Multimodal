# scripts/inference/chat_demo.py

"""
Interactive chat demo using Gradio.
"""

import gradio as gr
import torch
from virtue.inference.inference_engine import InferenceEngine
from virtue.models.multimodal.virtue_mm import VirtueMultimodalForCausalLM
from transformers import AutoTokenizer
from virtue.models import VirtueConfig

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = VirtueConfig()
    model = VirtueMultimodalForCausalLM(config).to(device)
    # load fine-tuned checkpoint if available...
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    engine = InferenceEngine(model, tokenizer, device=device)

    def respond(prompt, img): 
        return engine.generate(prompt, images=img, max_length=256)

    demo = gr.Interface(
        fn=respond,
        inputs=[gr.Textbox(lines=2, placeholder="Enter prompt..."), gr.Image(type="pil")],
        outputs="text",
        title="Virtue Multimodal Chat"
    )
    demo.launch()

if __name__ == "__main__":
    main()
