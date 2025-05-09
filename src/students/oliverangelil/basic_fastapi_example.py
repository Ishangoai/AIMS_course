from fastapi import FastAPI

# Create an instance of the FastAPI class
# This instance will be the main point of interaction to create all your API.
app = FastAPI()


# Define a path operation decorator
# This tells FastAPI that the function below is in charge of handling requests that go to:
# - the path "/"
# - using a GET operation
@app.get("/")
async def read_root():
    """
    This is the root endpoint of the API.
    It returns a simple JSON response.
    """
    return {"Hello": "World"}
