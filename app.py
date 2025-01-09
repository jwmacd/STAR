import spaces
import os
import gradio as gr
from video_super_resolution.scripts.inference_sr import STAR_sr

examples = [
    ["examples/023_klingai_reedit.mp4", "The video shows a panda strumming a guitar on a rock by a tranquil lake at sunset. With its black-and-white fur, the panda sits against a backdrop of mountains and a vibrant sky painted in orange and pink hues. The serene scene highlights relaxation and whimsy, with the panda, guitar, and lake harmoniously positioned. The natural landscape's depth and perspective enhance the focus on the panda's peaceful interaction with the guitar.", 4, 24, 250],
    ["examples/017_klingai_reedit.mp4", "The video depicts a majestic lion with eagle-like wings standing on a grassy hill against rolling green hills and a clear sky. The lion’s golden mane contrasts with the warm hues of the scene, and its intense gaze draws focus. The detailed, fully spread wings add a fantastical element. A 'PremiumBeat' watermark appears in the lower right, hinting at the image's source. The style blends realism with fantasy, showcasing the lion's mythical nature.", 4, 24, 250],
    ["examples/016_video.mp4", "The video is a black-and-white silent film featuring two men in wheelchairs on a pier. The foreground man, in a suit and hat, holds a sign reading 'HELP CRIPPLE.' The background shows a building and a boat, with early 20th-century clothing and image quality suggesting a narrative of disability and assistance.", 4, 24, 300],
]

# Define a GPU-decorated function for enhancement
@spaces.GPU(duration=120)
def enhance_with_gpu(input_video, input_text):
    return star.enhance_a_video(input_video, input_text)

def star_demo(result_dir="./tmp/"):
    css = """#input_video {max-width: 1024px !important} #output_vid {max-width: 2048px; max-height:1280px}"""
    global star
    star = STAR_sr(result_dir)
    
    with gr.Blocks(analytics_enabled=False, css=css) as star_iface:
        gr.Markdown(
            "<div align='center'> <h1> STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution </span> </h1> \
                     <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2501.02976'> [ArXiv] </a>\
                     <a style='font-size:18px;color: #000000' href='https://nju-pcalab.github.io/projects/STAR'> [Project Page] </a> \
                     <a style='font-size:18px;color: #000000' href='https://github.com/NJU-PCALab/STAR'> [Github] </a> </div>"
        )
        with gr.Tab(label="STAR"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            input_video = gr.Video(label="Input Video", elem_id="input_video")
                        with gr.Row():
                            input_text = gr.Text(label="Prompts")
                        end_btn = gr.Button("Generate")
                    with gr.Row():
                        output_video = gr.Video(
                            label="Generated Video", elem_id="output_vid", autoplay=True, show_share_button=True
                        )

                gr.Examples(
                    examples=examples,
                    inputs=[input_video, input_text],
                    outputs=[output_video],
                    fn=enhance_with_gpu,  # Use the GPU-decorated function
                    cache_examples=False,
                )
            end_btn.click(
                inputs=[input_video, input_text],
                outputs=[output_video],
                fn=enhance_with_gpu,  # Use the GPU-decorated function
            )

    return star_iface


if __name__ == "__main__":
    result_dir = os.path.join("./", "results")
    star_iface = star_demo(result_dir)
    star_iface.queue(max_size=12)
    star_iface.launch(max_threads=1)