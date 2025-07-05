# ![image](https://github.com/user-attachments/assets/aa56c067-7e7f-4ee3-ba19-b7f2ffa6abb8) Player Re-identification In Sports Footage (Single Feed):

This project aims to track football players, referees, and the ball from a match video using a pretrained *Ultralytics YOLOv11 model*. The goal is to ensure that the same player retains same ID, even across different camera feeds or after going out of the view.


## Features

- Detects players, referees, and the ball using a pre-trained YOLO model.

- Assigns fixed, unique IDs (e.g., Player 1–22) to players and tracks them across frames.

- Differentiates between red and blue jerseys using average color.

- Annotates frames with bounding boxes and labels:

    - Green for players

    - Yellow for referees

    - Blue for the ball

- Saves the output as a new annotated video.


## Installation

1. Clone the repo:

       git clone https://github.com/Deeksha0124/Player-Re-identification-In-Sports-Footage.git
   
       cd Player-Re-identification-In-Sports-Footage


2. Create a virtual environment (optional but recommended):

       python -m venv venv
    
       venv\Scripts\activate       # for Windows
 
3. Install dependencies:
  
       pip install -r requirements.txt



## Usage

1. Place your input video inside the Materials/ folder.

2. Put the model.pt file in the same directory as the code.

3. Run the tracking script:

       python player_reIdentification.py
   
5. The output video will be saved in the *Output* directory.


## System Architecture

[ YOLOv11 ] ──► [ Bounding Box Detection ]
			
   ▼
			
[ Center-based Tracker ]
   
   ▼
			
[ Unique ID Assignment ]

   ▼
			
[ Video Annotation & Saving ]


## Technical Details

- Language used: Python 3.10.18

- IDE used - Visual Studio Code

- Model used: YOLOv11 (pretrained)

- Tracking approach: Frame-to-frame matching using center distance and color.

- Thresholds (these values can be modified in the code as per your requirements) :

    - Maximum number of players: 22

    - Detection confidence = 0.3

    - ID matching distance = 60 pixels

- Video input: 1280×720, 15 seconds, 25 FPS


##  Performance Optimizations

- Resized input frames to speed up YOLO inference.

- Tracked player centers to avoid unnecessary ID switches.

- Only matched players within a threshold distance.

## Troubleshooting

| Problem                          | Solution                                                                 |
|----------------------------------|--------------------------------------------------------------------------|
| No output video                  | Check file paths for `model.pt` and input video                          |
| Players not detected             | Try lowering detection threshold slightly (e.g., 0.25)                   |
| IDs switching or overlapping     | Increase `DISTANCE_THRESHOLD` or improve matching logic                  |
| Bounding boxes missing           | Verify class names in `model.names` match player, referee, ball, etc.   |
| `cv2.imshow` crash in headless mode | Remove or comment out `cv2.imshow(...)` line                          |



## Output Sample

![image](https://github.com/user-attachments/assets/89744aa2-ba29-4bb3-8367-af5eb9633bd1)

- Player 1, Player 2, ... tracked across all frames

- Referee and Ball properly labeled

- Color-coded bounding boxes

Output saved at: 

        Output/result_video.mp4



## Notes:

- This project uses a pre-trained YOLOv11 model provided externally. Model fine-tuning or retraining was not part of the scope.

- Player Re-Identification is handled using center distance and jersey color, not advanced appearance embeddings or jersey number OCR.

- IDs may switch or overlap occasionally, especially when players crowd together, due to limitations of visual similarity and lack of deeper identity cues.

- The goal was to demonstrate basic consistent ID tracking of players in a football match using a single video and basic vision techniques.

- Ideal performance could be achieved by integrating better re-identification features such as:

  - Face embeddings

  - OCR for jersey numbers

  - Optical flow or trajectory matching


