import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

db = redis.Redis(
    host=settings.REDIS_IP,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB_ID
)

model = ResNet50(weights="imagenet", include_top=True)


def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """

    try:
        image_path = os.path.join(settings.UPLOAD_FOLDER, image_name)

        # Load image
        img = image.load_img(image_path, target_size=(224, 224))

        # Apply preprocessing (convert to numpy array, match model input dimensions (including batch) and use the resnet50 preprocessing)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get predictions using model methods and decode predictions using resnet50 decode_predictions
        predictions = model.predict(img_array)
        decoded_preds = decode_predictions(predictions, top=1)
        class_name = decoded_preds[0][0][1]
        pred_probability = decoded_preds[0][0][2]

        # Convert probabilities to float and round it
        return class_name, round(float(pred_probability), 4)
    except Exception as e:
        print(f"Error during prediction: {e}")


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Take a new job from Redis
        # returns a (list_name, element_value)
        q = db.brpop(settings.REDIS_QUEUE)[1]

        # Decode the JSON data for the given job
        q = json.loads(q)

        # Important! Get and keep the original job ID
        job_id = q["id"]

        # Run the loaded ml model (use the predict() function)
        pred_class, pred_probability = predict(q["image_name"])

        # Prepare a new JSON with the results
        output = {"prediction": pred_class, "score": pred_probability}

        # Store the job results on Redis using the original
        # job ID as the key
        db.set(job_id, json.dumps(output))

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
