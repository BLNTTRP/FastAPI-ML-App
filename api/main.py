from app.auth import router as auth_router
from app.feedback import router as feedback_router
from app.model import router as model_router
from app.user import router as user_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Image Prediction API", version="0.0.1")

# Allow calls from the Streamlit app
origins = [
    "http://localhost:9090/",
    "http://127.0.0.1:9090/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # o [""] en desarrollo
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)
app.include_router(auth_router.router)
app.include_router(model_router.router)
app.include_router(user_router.router)
app.include_router(feedback_router.router)
