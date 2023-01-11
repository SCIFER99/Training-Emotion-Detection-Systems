# By: Tim Tarver

# Implementing a Real-Time Emotion Detection System in Python

from torchvision.transforms import ToPILImage
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision import transforms
import EmotionNet
import torch.nn.functional as nnf
import utils
import numpy as np
import argparse
import torch
import cv2

# Initializes the argument parser and establishes the required arguments.

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--video", type=str, required=True,
                    help="Path to the Video File/Webcam")
parser.add_argument("-m", "--model", type=str, required=True,
                    help="Path to the Trained Model")
parser.add_argument("-p", "--prototxt", type=str, required=True,
                    help="Path to Deployed prototxt.txt model architecture file")
parser.add_argument('-c', '--caffemodel', type=str, required=True,
                    help='Path to Caffe model containing the weights')
parser.add_argument("-conf", "--confidence", type=int, default=0.5,
                    help="the minimum probability to filter out weak detection")
args = vars(parser.parse_args())

# Now load our serialized model from the disk

print("[INFO loading model...")
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['caffemodel'])

# Check if GPU is available or not

device = "Cuda" if torch.cuda.is_available() else "CPU"

# Dictionary Mapping fo rDifferent Outputs

emotion_dict = {"Angry": 0, "Fearful": 1, "Happy": 2, "Neutral": 3,
                "Sad": 4, "Surprised": 5}

# Load the EmotionNet weights

model = EmotionNet(num_of_channels = 1, num_of_classes = len(emotion_dict))
model_weights = torch.load(args["model"])
model.load_state_dict(model_weights)
model.to(device)
model.eval()

# Initialize a list of preprocessing steps to apply to each image during runtime

data_transform = transforms.Compose([ToPILImage(),
                                     Grayscale(num_output_channels = 1),
                                     Resize((48, 48)),
                                     ToTensor()])

# Initialize the video stream

video_stream = cv2.VideoCapture(args['video'])

# Iterate over the frames from the video file stream

while True:

    # Read the next frame from the input stream
    
    grabbed, frame = video_stream.read()

    # Check if there's any frame to be grabbed from the stream

    if not grabbed:
        break

    # Clone the current frame, convert it from BGR to RGB

    frame = utils.resize_image(frame, width = 1500, height = 1500)
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize an Empty Canvas to output the probability distributions.

    canvas = np.zeros((300, 300, 3), dtype = "uint8")

    # Get the frame dimension, resize it and conert it to a blob.

    height1, width1 = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300))

    # Infer the blog through the netwoek to get the detections and predictions

    net.setInput(blob)
    detections = net.forward()
    
    # Here is where the iteration over the detections come in.

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > args['confidence']:

            box = detections[0, 0, i, 3:7] * np.array([width1, height1, width1, height1])
            start_x, start_y, end_x, end_y = box.astype("int")

            face = frame[start_y:end_y, start_x:end_x]
            face = data_transform(face)
            face = face.unsqueeze(0)
            face = face.to(device)

            predictions = model(face)
            probability = nnf.softmax(predictions, dim = 1)
            top_probability, top_class = probability.topk(1, dim = 1)
            top_probability, top_class = top_probability.item(), top_class.item()

            emotion_probability = [p.item() for p in probability[0]]
            emotion_value = emotion_dict.values()

            # Draw the Probability Distribution on an empty canvas initialized

            for i, (emotion, prob) in enumerate(zip(emotion_value, emotion_probability)):

                probability_text = f"{emotion}: {prob * 100:.2f}%" # Regular Expression
                width = int(prob * 300)
                cv2.rectangle(canvas, (5, (i * 50) + 5), (width, (i * 50) + 50),
                              (0, 0, 255), -1)
                cv2.putText(canvas, probability_text, (5, (i * 50) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # These lines draw the Bounded Box around the face along with the associated
            # emotion and probability

            face_emotion = emotion_dict[top_class]
            face_text = f"{face_emotion}: {top_probability * 100:.2f}%" # Regular Expression
            cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.putText(output, face_text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.05, (0, 255, 0), 2)

    # Display the output to the screen

    cv2.imshow("Face", output)
    cv2.imshow("Emotion probability distribution", canvas)

    # Break the loop if the 'q' key is pressed

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):

        break

cv2.destroyAllWindows()
video_stream.release()
            
            
