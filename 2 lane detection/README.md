# GOLD Lane Detection

This repository contains the implementation of the GOLD Lane Detection pipeline, integrated with YOLOv8 for obstacle detection and distance estimation.

## Prerequisites

Before running the project, you need to install all the required Python libraries. A `requirements.txt` file is provided for your convenience.

To install the dependencies, open your terminal and run the following command from the project's root directory:

```bash
pip install -r requirements.txt
```

## How to Run the Pipeline

The entry point of the project is the `run_gold.py` script. It requires the search path to the dataset images as a command-line argument.

To execute the code, run:

```bash
python3 run_gold.py 'PandaSetSensorData/archive/044/Camera/front_camera/*.jpg'
```

**Important Note regarding the path:** 
The string `'PandaSetSensorData/archive/044/Camera/front_camera/*.jpg'` is an example of the relative path where the dataset is located. You should modify this path to point to the correct folder containing the dataset images on your local machine. Make sure to keep the `/*.jpg` at the end or wrap the path in quotes if using wildcards.

## Controls and Visualization

Once the command is launched:
- The script processes each image and displays a comprehensive OpenCV dashboard.
- The **Bird's-Eye View (BEV)** is shown with the detected lane delimiters overlaid as **green vertical lines**.
- If no lanes are detected in a given frame, the text **"No lanes found"** will be displayed in the BEV panel.
- To **advance to the next frame**, press any key. 
- To **quit** the execution, press `q` or the `ESC` key.
