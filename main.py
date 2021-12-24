from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi import FastAPI, Form

from src.predict import *

app = FastAPI()

templates = Jinja2Templates(directory="./templates/")




@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})
    
@app.post("/predict/")
async def create_item(request: Request, sentence: str = Form("sentence"), aspect : str = Form("aspect")):
    print(sentence)
    label=pred(sentence, aspect)[0]
    label_dict={
        0:"Negative",
        1:"Neutral",
        2:"Positive"
    }
    label=label_dict[label]
    return templates.TemplateResponse("show.html", {"request":request, "sentence":sentence, "aspect":aspect, "label":label})

