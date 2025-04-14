
from diffusers.utils import make_image_grid    
def infer(url):

    print(url)
    return []
    original = Image.open(url).resize((1024, 1024), Image.LANCZOS) 
    output_image = txt2img_depth_controlnet(
        control_image_url=url,
        #prompt="Beautiful bald baby",
        controlnet_conditioning_scale=.7,
        guidance= 10.5,
        num_inference_steps=25
        ) 

    return output_image
   

import gradio as gr
demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(label="Input Image"),  # image input
    ],
    outputs=gr.Image(label="Output Image"),
    title="Stable Diffusion XL Image-to-Image Demo",
   
)

demo.launch()