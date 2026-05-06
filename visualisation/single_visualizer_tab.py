import gradio as gr
import data_processing.utils as utils

MAX_GRID_SLOTS = 6

def render_single_visualizer_tab(device):
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload Single NPZ or MP4", file_count="single")
            with gr.Row():
                use_cache = gr.Checkbox(label="Use Cache", value=True)
                custom_name = gr.Textbox(label="Custom Label", placeholder="Experiment Note")
        
        with gr.Column(scale=1):
            source_dropdown = gr.Dropdown(choices=["Seamless", "BEAT"], value="Seamless", label="Source")
            render_btn = gr.Button("Process / Add to Grid", variant="primary")

    gr.HTML("<hr>")
    
    video_slots = []
    slot_components = [] 
    with gr.Row():
        for i in range(MAX_GRID_SLOTS):
            with gr.Column(visible=False, min_width=300) as col:
                method_md = gr.Markdown("### Method")
                vid_comp = gr.Video(label=f"Slot {i+1}", interactive=False)
                remove_btn = gr.Button("❌ Remove", size="sm", variant="stop")
                video_slots.append((col, method_md, vid_comp, remove_btn))
                slot_components.extend([col, method_md, vid_comp, remove_btn])

    slot_counter = gr.State(0)

    def add_to_grid(file_obj, source, cache_flag, label_text, current_idx, progress=gr.Progress()):
        if not file_obj:
            return [current_idx, gr.update(), gr.update()] + [gr.update() for _ in range(MAX_GRID_SLOTS * 4)]

        slot_updates = [gr.update() for _ in range(MAX_GRID_SLOTS * 4)]
        method_label = f"### {source} (Single)\n{label_text}"
        
        if file_obj.name.lower().endswith('.mp4'):
            result_path = file_obj.name
        else:
            loader = utils.SeamlessSMPLHLoader(file_obj.name, device=device) if source == 'Seamless' else utils.BeatLoader(file_obj.name, device=device)
            result_path = loader.generate_video(progress=progress, use_cache=cache_flag)

        slot_pos = current_idx * 4
        slot_updates[slot_pos:slot_pos+4] = [gr.update(visible=True), gr.update(value=method_label), gr.update(value=result_path), gr.update()]
        new_idx = (current_idx + 1) % MAX_GRID_SLOTS

        return [new_idx, gr.update(value=None), gr.update(value="")] + slot_updates

    render_btn.click(
        fn=add_to_grid,
        inputs=[file_input, source_dropdown, use_cache, custom_name, slot_counter],
        outputs=[slot_counter, file_input, custom_name] + slot_components
    )

    for i in range(MAX_GRID_SLOTS):
        col_to_hide, md_to_clear, vid_to_clear, btn = video_slots[i]
        btn.click(fn=lambda: (gr.update(visible=False), gr.update(value=""), gr.update(value=None)), outputs=[col_to_hide, md_to_clear, vid_to_clear])