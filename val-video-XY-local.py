import cv2
import torch
import numpy as np
import argparse
import yaml
from albumentations import Compose, Resize
from albumentations.augmentations import transforms
from archs import UNext

# python val-video-XY-local.py --name XY-local-UNeXt-all299-0804 --video_path "F:\SimonWorkspace\SpermSelectionDataset\video\\Basler acA1920-155ucMED (40214438)_20240618_193321139.mp4"
# python val-video-XY-local.py --name XY-local-UNeXt-all299-0804 --video_path "F:\SimonWorkspace\SpermSelectionDataset\video\\Basler acA1920-155ucMED (40214438)_20240618_195335524.mp4"

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:  # Process every 5th frame
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

            for c in range(config['num_classes']):
                output_resized = cv2.resize(output[0, c], (frame.shape[1], frame.shape[0]))
                contours, _ = cv2.findContours(output_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                        angle_rad = np.pi / 7
                        length = 150
                        end_x1 = int(cX + length * np.cos(angle_rad))
                        end_y1 = int(cY - length * np.sin(angle_rad))
                        end_x2 = int(cX - length * np.cos(angle_rad))
                        end_y2 = int(cY - length * np.sin(angle_rad))
                        # cv2.line(frame, (cX, cY), (end_x1, end_y1), (0, 0, 255), 2)
                        # cv2.line(frame, (cX, cY), (end_x2, end_y2), (0, 0, 255), 2)

            cv2.imshow('Predictions', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()