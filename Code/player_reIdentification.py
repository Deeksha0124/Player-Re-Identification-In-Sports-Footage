import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO(r"E:\LIAT.ai Stealth Mode\Materials\model.pt")
video = cv2.VideoCapture(r"E:\LIAT.ai Stealth Mode\Materials\15sec_input_720p.mp4")

if not video.isOpened():
    print("Error: Couldn't open video.")
    exit()

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

output_path = r"E:\LIAT.ai Stealth Mode\Output\result_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


MAX_PLAYERS = 22
DETECTION_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 60
detect_width, detect_height = 1280, 720

detected_players = {} # here we are creating an empty dictionary 
referee_info = None
ball_info = None
vacant_ids = list(range(1, MAX_PLAYERS + 1))
frame_count = 0


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_same_object(center1, center2, threshold=DISTANCE_THRESHOLD):
    return np.linalg.norm(np.array(center1) - np.array(center2)) < threshold


while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break         # we break when all frames are over 

    frame_count += 1
    used_ids = set()

    resized_frame = cv2.resize(frame, (detect_width, detect_height))
    scale_x = frame.shape[1] / detect_width
    scale_y = frame.shape[0] / detect_height

    result = model(resized_frame, conf=DETECTION_THRESHOLD, verbose=False)[0]

    for box in result.boxes: 
        cls_id = int(box.cls.item())  # we are obtaining the bbox of 3 different classes - player, referee, ball 
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        bbox = (x1, y1, x2, y2)
        center = get_center(bbox)
        class_name = model.names[cls_id].lower()

        label = None
        color = (255, 255, 255)  

        if class_name == "player":
            crop = frame[y1:y2, x1:x2]
            avg_color = crop.mean(axis=(0, 1)) if crop.size else (0, 0, 0)
            b, g, r = avg_color
            jersey_color = "blue" if b > r else "red"

            matched_id = None
            min_dist = float('inf')

            for pid, pdata in detected_players.items():
                dist = np.linalg.norm(np.array(center) - np.array(pdata['center']))
                if pid not in used_ids and dist < min_dist and dist < DISTANCE_THRESHOLD:
                    matched_id = pid
                    min_dist = dist

            if matched_id is None:
                if vacant_ids:
                    matched_id = vacant_ids.pop(0)
                    detected_players[matched_id] = {
                        'bbox': bbox,
                        'center': center,
                        'color': jersey_color,
                        'last_seen': frame_count
                    }
                    used_ids.add(matched_id)
            else:
                detected_players[matched_id]['bbox'] = bbox
                detected_players[matched_id]['center'] = center
                detected_players[matched_id]['last_seen'] = frame_count
                used_ids.add(matched_id)

            if matched_id is not None:
                label = f"Player {matched_id}"
                color = (0, 255, 0)  # Green for players

        elif class_name == "referee":
            referee_info = {'bbox': bbox, 'center': center}
            label = "Referee"
            color = (0, 255, 255)  # Yellow for refree

        elif class_name == "ball":
            ball_info = {'bbox': bbox, 'center': center}
            label = "Ball"
            color = (255, 0, 0)  # Blue for Ball

      
        if label:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            


    cv2.imshow("Tracking", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to exit from the output window
        break

# Results
video.release()
out.release()
cv2.destroyAllWindows()

