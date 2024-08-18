import cv2
import torch
import numpy as np
import argparse
import yaml
from albumentations import Compose, Resize
from albumentations.augmentations import transforms
from archs import UNext

# python val-video-XY-local.py --name XY-local-UNeXt-all299-0804 --video_path "F:\SimonWorkspace\SpermSelectionDataset\video\\Basler acA1920-155ucMED (40214438)_20240618_195335524.mp4"
# python val-video-XY-traincking.py --name XY-local-UNeXt-all299-0804 --video_path "F:\\yf\\7hh\\0__40142253__20240814_233248181.avi"

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
    trackers = {}  # Dictionary to hold tracking info
    color_map = {}  # Map each id to a unique color

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 1 == 0:  # Process every 5th frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3channel = cv2.merge([gray_frame, gray_frame, gray_frame])

            augmented = val_transform(image=gray_3channel)
            img = augmented['image']
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)
            img = torch.tensor(img).unsqueeze(0).cuda()

            with torch.no_grad():
                output = torch.sigmoid(model(img)).cpu().numpy()
                output = (output >= 0.5).astype(np.uint8)

            new_trackers = {}
            for c in range(config['num_classes']):
                output_resized = cv2.resize(output[0, c], (frame.shape[1], frame.shape[0]))
                contours, _ = cv2.findContours(output_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Find closest existing tracker
                        min_distance = float('inf')
                        closest_id = None
                        for tracker_id, pos in trackers.items():
                            distance = np.sqrt((pos[0] - cX)**2 + (pos[1] - cY)**2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_id = tracker_id

                        if min_distance > 500000:  # If no close tracker, create new
                            closest_id = len(color_map)
                            color_map[closest_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                        new_trackers[closest_id] = (cX, cY)
                        cv2.circle(frame, (cX, cY), 5, color_map[closest_id], -1)

            trackers = new_trackers

            cv2.imshow('Predictions', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

