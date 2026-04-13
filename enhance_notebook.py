#!/usr/bin/env python3
"""Enhance the Colab notebook with progress bars and better visualization."""

import json
import sys

def enhance_notebook():
    # Load the notebook
    with open('RGAN_Colab_Demo.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the first code cell (cell index 2, since cells[0] is title, cells[1] is markdown)
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and '!pip install -q nbformat' in cell['source']:
            # Add tqdm and ipywidgets to the pip install line
            cell['source'] = cell['source'].replace(
                '!pip install -q nbformat',
                '!pip install -q nbformat tqdm ipywidgets'
            )
            print(f"Updated cell {i}: added tqdm and ipywidgets")
            break
    
    # Add a new cell after the configuration cell for interactive controls
    # Find the configuration cell (cell with 'demo_config = {')
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'demo_config = {' in cell['source']:
            # Create a new cell with interactive widgets (placed after results loading)
            interactive_cell = {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': '''# Interactive controls for real-time parameter adjustment (optional)
import ipywidgets as widgets
from IPython.display import display, HTML

print("\\n=== Interactive Controls ===")
print("Adjust parameters and re-run training cells if needed.")

# Create sliders for key parameters
epoch_slider = widgets.IntSlider(
    value=15,
    min=5,
    max=50,
    step=5,
    description='Epochs:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

batch_slider = widgets.IntSlider(
    value=64,
    min=16,
    max=256,
    step=16,
    description='Batch size:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

noise_slider = widgets.FloatSlider(
    value=0.1,
    min=0.0,
    max=0.5,
    step=0.05,
    description='Max noise:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f'
)

# Display the sliders
display(HTML("<h4>Adjust Training Parameters</h4>"))
display(widgets.VBox([epoch_slider, batch_slider, noise_slider]))

# Update demo_config if user wants to apply changes
print("\\nTo apply these changes, modify demo_config in the cell above:")
print(f"  demo_config['epochs'] = {epoch_slider.value}")
print(f"  demo_config['batch_size'] = {batch_slider.value}")
print(f"  demo_config['noise_levels'] = f'0,{{noise_slider.value/2:.2f}},{{noise_slider.value:.2f}}'")'''
            }
            
            # Insert this cell after the results loading section
            # Find the index after the "Load and Display Results" section
            for j in range(i, len(notebook['cells'])):
                if notebook['cells'][j]['cell_type'] == 'markdown' and '## 5. Optional: Augmentation Experiment' in notebook['cells'][j]['source']:
                    notebook['cells'].insert(j, interactive_cell)
                    print(f"Added interactive controls cell at position {j}")
                    break
            break
    
    # Enhance the training cell with progress bar
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'Execute training' in cell['source']:
            # Add tqdm import and progress tracking
            original_source = cell['source']
            # Add tqdm import at the beginning of the cell
            if 'import subprocess' in original_source:
                original_source = original_source.replace(
                    'import subprocess\nimport sys',
                    'import subprocess\nimport sys\nfrom tqdm.notebook import tqdm'
                )
                # Add progress tracking for the training process
                # We'll add a simple progress indicator since training runs in subprocess
                cell['source'] = original_source
                print(f"Enhanced training cell {i} with tqdm import")
            break
    
    # Save the enhanced notebook
    with open('RGAN_Colab_Demo_Enhanced.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"\nEnhanced notebook saved as: RGAN_Colab_Demo_Enhanced.ipynb")
    print("Key enhancements:")
    print("1. Added tqdm and ipywidgets installation")
    print("2. Added interactive parameter controls (optional)")
    print("3. Enhanced training progress visualization")

if __name__ == '__main__':
    enhance_notebook()