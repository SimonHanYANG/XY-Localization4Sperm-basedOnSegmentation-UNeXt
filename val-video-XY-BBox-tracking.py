import cv2
import torch
import numpy as np
import argparse
import yaml
from albumentations import Compose, Resize
from albumentations.augmentations import transforms
from archs import UNext
from sort import Sort

# python val-video-XY-BBox-tracking.py --name XY-local-UNeXt-all299-0804 --video_path "F:\\yf\\7hh\\0__40142253__20240814_233248181.avi"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Model name')
    parser.add_argument('--video_path', required=True, help='Path to the input video')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(f'models/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = UNext(config['num_classes'], config['input_channels'], config['deep_supervision'])
    model.load_state_dict(torch.load(f'models/{args.name}/model.pth'))
    model.cuda().eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    cap = cv2.VideoCapture(args.video_path)
    frame_count = 0

    # Initialize the SORT tracker
    tracker = Sort()
    colors = {}  # To store colors associated with each track ID

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        gray_3channel = cv2.merge([gray_frame, gray_frame, gray_frame])  # Merge to make 3 channels

        augmented = val_transform(image=gray_3channel)
        img = augmented['image']
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img).unsqueeze(0).cuda()

        with torch.no_grad():
            output = torch.sigmoid(model(img)).cpu().numpy()
            output = (output >= 0.5).astype(np.uint8)

        detections = []  # Store bounding boxes in [x1, y1, x2, y2, confidence] format
        for c in range(config['num_classes']):
            output_resized = cv2.resize(output[0, c], (frame.shape[1], frame.shape[0]))
            contours, _ = cv2.findContours(output_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append([x, y, x + w+40, y + h+40, 1.0])  # Format: [x1, y1, x2, y2, score]

        # Update the tracker
        tracked_objects = tracker.update(np.array(detections))

        # Draw the bounding boxes with consistent colors for the same object ID
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track.astype(int)
            
            # Assign a unique color to each track_id
            if track_id not in colors:
                colors[track_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            
            color = colors[track_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Predictions', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
