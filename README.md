# -YOLO Nuisible Project
This repository contains the code and data for a YOLO-based object detection model trained to identify nuisible insects.

Project Structure
projet_nuisible.v1i.yolov8-obb.zip: The dataset used for training, validation, and testing.
runs/detect/train2: Contains the training results, including weights, plots, and logs.
weights_backup: (Optional) Backup of model weights stored on Google Drive.
Setup and Usage
Clone the repository:
 pip install ultralytics


 from ultralytics import YOLO

   # Load a model
   model = YOLO("yolo11n.yaml")  # build a new model from YAML
   model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
   model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

   # Train the model
   results = model.train(data="/content/data.yaml", epochs=100, imgsz=640)




   from ultralytics import YOLO

   # Load the best trained model
   model = YOLO("runs/detect/train2/weights/best.pt")

   # Define path to the image file or directory
   source = "/test/images"

   # Run inference on the source
   results = model(source)  # list of Results objects

   # Process and display results (example)
   for result in results:
       result.show()  # display to screen
       result.save(filename="result.jpg") # save the result
