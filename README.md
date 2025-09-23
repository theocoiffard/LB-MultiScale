# 🌀 LB-WBS: Lattice Boltzmann for Heterogeneous Porous Media

This project implements a **Lattice Boltzmann (LB-WBS) scheme** for simulating flows in heterogeneous porous media with pressure dependence.

## 📂 Project structure

- `streamlit run Assist_WBS.py`: Streamlit interface to **set up LB-WBS parameters** and **save them into a JSON file** describing the porous medium.  
- `streamlit run Reader_JSON.py`: Streamlit interface to **read and visualize** the content of a JSON file.  
- `LB_WBS.py`: Implementation of the LB-WBS scheme.  
- `PorousMedia.py`: Generation of heterogeneous porous matrices.  
- `extractJSON.py`: Reading and extracting parameters from `.json` files.  
- `utils.py`: Utility functions (visualization, parameter handling, etc.).  
- `media_json/`: Directory containing `.json` files describing porous layer properties.  
- `main.py`: Full example script illustrating how to use the LB-WBS scheme.  
- `_JSON_impermeable/`: Scripts to generate JSON files describing **impermeable properties**.  

## ⚙️ Requirements

Install the required dependencies (tested with **Python 3.9**):  

```bash
pip install -r requirements.txt
```

## 🚀 Example run

Run the main script:  

```bash
python main.py
```

This script will:  
1. Generate a porous matrix with multiple layers.  
2. Extract parameters from the `.json` files.  
3. Build the LB-WBS model with pressure dependence.  
4. Run the simulation until convergence.  
5. Display the **velocity magnitude field** with `matplotlib`.  

## 🖥️ Streamlit interfaces

Two graphical interfaces are provided to make the project easier to use:  

### 🔧 Parameter setup and JSON generation
```bash
streamlit run Assist_WBS.py
```
This interface allows you to:  
- Configure the LB-WBS scheme parameters.  
- Save them into a **JSON** file describing the porous medium.  

### 📑 JSON reading and visualization
```bash
streamlit run Reader_JSON.py
```
This interface allows you to:  
- Load an existing **JSON** file.  
- Visualize its parameters and associated properties.  
