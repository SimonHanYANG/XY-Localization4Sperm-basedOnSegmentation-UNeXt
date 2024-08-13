import argparse
import os
import yaml

import archs

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize

from pypylon import pylon
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Model name')
    args = parser.parse_args()
    return args

args = parse_args()

with open(f'models/{args.name}/config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print('-' * 20)
for key in config.keys():
    print(f'{key}: {config[key]}')
print('-' * 20)

cudnn.benchmark = True

print(f"=> creating model {config['arch']}")
model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'])
model = model.cuda()
model.load_state_dict(torch.load(f'models/{args.name}/model.pth'))
model.eval()

transform = Compose([
    Resize(config['input_h'], config['input_w']),
    transforms.Normalize(),
])

def calculate_focus_measure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def save_image(image, path):
    cv2.imwrite(path, image)

def draw_text(image, text, position, color, font_scale=0.7, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def process_and_visualize(image):
    # Convert image to grayscale and then replicate to three channels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.merge([gray, gray, gray])
    
    # Apply transformations
    augmented = transform(image=img)
    img = augmented['image']
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)
    input_tensor = torch.tensor(img).unsqueeze(0).cuda()

    with torch.no_grad():
        prediction = model(input_tensor)  # Assume the model outputs raw logits or scores

    # Thresholding the prediction for visualization
    mask = prediction.squeeze().cpu().numpy()
    mask = mask > 0.5  # Thresholding step

    # Find contours and draw
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    return image

def main():
    if not os.path.exists('in-focus-img'):
        os.makedirs('in-focus-img')

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    saved_frames = []
    focus_measures = []
    message = ""
    frame_count = 0

    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                image = converter.Convert(grab_result)
                img = image.GetArray()

                # Ensure image is uint8 type
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                frame_count += 1
                if frame_count % 5 == 0:  # Process every 5 frames
                    img = process_and_visualize(img)

                if message:
                    draw_text(img, message, (10, 30), (0, 0, 255))
                cv2.imshow('Basler Camera', img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    saved_frames.append(img)
                    focus_measure = calculate_focus_measure(img)
                    focus_measures.append(focus_measure)
                    message = f"Saved image with FM={focus_measure:.2f}"
                    print(f"Saved frame {len(saved_frames)} with focus measure {focus_measure}")
                elif key == ord('c') and saved_frames:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(focus_measures) + 1), focus_measures, marker='o', linestyle='-')
                    plt.title('Focus Measure Values')
                    plt.xlabel('Frame Number')
                    plt.ylabel('Focus Measure (Laplacian Variance)')
                    plt.grid(True)
                    plt.savefig(os.path.join('in-focus-img', 'focus_measures.png'))
                    plt.show()
                    best_focus_index = np.argmax(focus_measures)
                    best_image = saved_frames[best_focus_index]
                    best_image_path = os.path.join('in-focus-img', 'best_focus_image.png')
                    save_image(best_image, best_image_path)
                    print(f"Saved best focus image to {best_image_path}")
                    saved_frames.clear()
                    focus_measures.clear()
                    message = ""
                elif key == ord('q'):
                    print("Exiting program...")
                    break

            grab_result.Release()

    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()