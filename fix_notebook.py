import json

# Read the notebook
with open('notebooks/model_fit.ipynb', 'r') as f:
    notebook = json.load(f)

# Find and fix the corrupted cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and any('arrows_= compare_benchmark_forecasts' in line for line in cell['source']):
        print(f"Found corrupted cell at index {i}")
        
        # Find the line with the corruption
        fixed_source = []
        skip_until_ax_set = False
        
        for line in cell['source']:
            if 'arrows_= compare_benchmark_forecasts' in line:
                # Replace the corrupted line with the correct arrowprops
                fixed_source.append("                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))\n")
                fixed_source.append("    \n")
                fixed_source.append("    ax.set_xlabel('Release Date', fontsize=14)\n")
                skip_until_ax_set = True
            elif skip_until_ax_set:
                # Skip all the corrupted lines until we find ax.set_ylabel
                if 'ax.set_ylabel' in line:
                    # Fix the indentation
                    fixed_source.append("    ax.set_ylabel('Estimated Capability', fontsize=14)\n")
                    skip_until_ax_set = False
            else:
                fixed_source.append(line)
        
        cell['source'] = fixed_source
        print(f"Fixed {len(cell['source'])} lines")
        break

# Write the fixed notebook
with open('notebooks/model_fit.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook fixed successfully!")

