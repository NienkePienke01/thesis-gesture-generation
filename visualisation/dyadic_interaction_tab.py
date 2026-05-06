import gradio as gr
import data_processing.utils as utils
from visualisation.file_browser import create_file_browser

def validate_interaction_files(file_list):
    if file_list and len(file_list) != 2:
        gr.Warning("Please select exactly 2 files for a Dyadic Interaction.")
        return file_list[:2]
    return file_list

def render_dyadic_interaction_tab(device):
    with gr.Row(variant="panel"):
        with gr.Column():
            files_input = create_file_browser("Select 2 NPZ Files", extensions=[".npz"], file_count="multiple")
            source_dropdown = gr.Dropdown(choices=["Seamless", "BEAT"], value="Seamless", label="Source")
            render_btn = gr.Button("Generate Interaction Video", variant="primary")
        
        with gr.Column():
            output_video = gr.Video(label="Interaction Render")

    def process_interaction(file_list, source, progress=gr.Progress()):
        if not file_list or len(file_list) < 2:
            return None
        
        loader_1 = utils.SeamlessSMPLHLoader(file_list[0], device=device) if source == 'Seamless' else utils.BeatLoader(file_list[0], device=device)
        loader_2 = utils.SeamlessSMPLHLoader(file_list[1], device=device) if source == 'Seamless' else utils.BeatLoader(file_list[1], device=device)
        
        return utils.generate_interaction_video(loader_1, loader_2, progress=progress)

    render_btn.click(fn=process_interaction, inputs=[files_input, source_dropdown], outputs=output_video)