import os
from typing import List

from app import db
from app import settings as config
from app import utils
from app.auth.jwt import get_current_user
from app.model.schema import PredictRequest, PredictResponse
from app.model.services import model_predict
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

router = APIRouter(tags=["Model"], prefix="/model")


@router.post("/predict")
async def predict(file: UploadFile, current_user=Depends(get_current_user)):
    print("Predicting...")
    rpse = {"success": False, "prediction": None, "score": None}
    if not utils.allowed_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File type is not supported.",
        )

    file_hash = await utils.get_file_hash(file)
    dst_filepath = os.path.join(config.UPLOAD_FOLDER, file_hash)
    if not os.path.exists(dst_filepath):
        with open(dst_filepath, "wb") as f:
            f.write(await file.read())
    print(f"File saved to {dst_filepath}")
    print("Calling model predict function")
    prediction, score = await model_predict(file_hash)
    print("Prediction complete")
    rpse["success"] = True
    rpse["prediction"] = prediction
    rpse["score"] = score
    rpse["image_file_name"] = file_hash

    return PredictResponse(**rpse)
