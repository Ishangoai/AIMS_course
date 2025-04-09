import os
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    user = os.environ.get("GITHUB_USER", "default")
    return {"message": f"Hello form {user}!"}
