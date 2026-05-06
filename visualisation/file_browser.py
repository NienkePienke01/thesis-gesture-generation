import gradio as gr
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def create_file_browser(label, extensions=[".npz"], file_count="single"):
    """
    Creates a remote file selector
    Returns the selector component
    """
    def get_files(filter_query=""):
        choices = []
        ext_list = [ext.lower() for ext in (extensions if isinstance(extensions, list) else [extensions])]
        query = filter_query.strip().lower()
        
        for root, _, files in os.walk(PROJECT_ROOT):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                
                if any(file.lower().endswith(ext) for ext in ext_list):
                    if not query or query in rel_path.lower():
                        choices.append((rel_path, full_path))
        
        return sorted(choices, key=lambda x: x[0])

    def handle_scan(query):
        new_choices = get_files(query)
        return gr.update(choices=new_choices, value=None)

    with gr.Column(variant="panel"):
        gr.Markdown(f"**{label}**")
        with gr.Row():
            keyword_filter = gr.Textbox(
                placeholder="Filter by keyword (e.g. 'V00_S0030_I00000126_P0046')...",
                label=None,
                show_label=False,
                scale=2
            )
            refresh_btn = gr.Button("🔍 Scan for Files", scale=2, variant="primary")

        selector = gr.Dropdown(
            choices=get_files(),
            label=None,
            show_label=False,
            multiselect=(file_count == "multiple"),
            filterable=True,
            allow_custom_value=True,
            info="Select from the filtered list below"
        )
        
    refresh_btn.click(
        fn=handle_scan, 
        inputs=keyword_filter, 
        outputs=selector
    )
    
    return selector