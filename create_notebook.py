import json

def convert_to_notebook(py_file, ipynb_file):
    with open(py_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cells = []
    
    # First cell: imports and setup
    current_cell = []
    in_phase = False
    
    for line in lines:
        if line.startswith("# ====="):
            if current_cell:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": current_cell
                })
            current_cell = [line]
        else:
            current_cell.append(line)
            
    if current_cell:
        # Avoid including the __main__ block if we want it clean, but it's fine.
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": current_cell
        })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(ipynb_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    convert_to_notebook("run_v2.py", "Traffic_Demand_Prediction.ipynb")
