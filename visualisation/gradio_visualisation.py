import gradio as gr
import torch
from visualisation.single_visualizer_tab import render_single_visualizer_tab
from visualisation.dyadic_interaction_tab import render_dyadic_interaction_tab
from visualisation.reconstruction_tab import render_reconstruction_tab
from visualisation.reconstruction_tab_whole_body import render_full_body_reconstruction_tab
from visualisation.lmbd_sample_visualisation import render_lmdb_visualiser_tab
import os
from pathlib import Path

SKIP_INTERVALS = [-30, -15, -5, 5, 15, 30]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
custom_temp_dir = PROJECT_ROOT / "cache" / "gradio_tmp"
os.makedirs(custom_temp_dir, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = str(custom_temp_dir)

def main():
    js_controls = {
        "play": "() => { document.querySelectorAll('video').forEach(v => v.play()); }",
        "pause": "() => { document.querySelectorAll('video').forEach(v => v.pause()); }",
        "reset": "() => { document.querySelectorAll('video').forEach(v => { v.currentTime = 0; v.pause(); }); }"
    }

    with gr.Blocks(title="Gesture Visualizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Gesture Visualisation System")
        
        with gr.Tabs():
            with gr.TabItem("Single Visualizer"):
                render_single_visualizer_tab(device)

            with gr.TabItem("Dyadic Interaction"):
                render_dyadic_interaction_tab(device)

            with gr.TabItem("Stage 1 Reconstruction (Hands and Upper Body)"):
                render_reconstruction_tab(device)
            
            with gr.TabItem("Stage 1 Reconstruction (Full Body)"):
                render_full_body_reconstruction_tab(device)
            
            with gr.TabItem("LMDB visualisation data check"):
                render_lmdb_visualiser_tab(device)

        # Unified Global Controls
        with gr.Row(variant="compact"):
            play_btn = gr.Button("▶ Start All")
            pause_btn = gr.Button("⏸ Pause All")
            reset_all_btn = gr.Button("⏮ Reset All to 0.0")

        # Global Time-Skip Controls
        with gr.Row():
            for seconds in SKIP_INTERVALS:
                label = f"⏪ {abs(seconds)}s" if seconds < 0 else f"{seconds}s ⏩"
                btn = gr.Button(label)
                skip_js = f"() => {{ document.querySelectorAll('video').forEach(v => v.currentTime += ({seconds})); }}"
                btn.click(fn=None, inputs=None, outputs=None, js=skip_js)

        # Event listeners for transport controls
        play_btn.click(fn=None, inputs=None, outputs=None, js=js_controls["play"])
        pause_btn.click(fn=None, inputs=None, outputs=None, js=js_controls["pause"])
        reset_all_btn.click(fn=None, inputs=None, outputs=None, js=js_controls["reset"])

    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch(share=True)