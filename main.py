from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import json
import uvicorn

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates/static/"), name="static")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(s):
    return s * (1 - s)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def load_from_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        self.weights1 = np.array(data["weights1"])
        self.bias1 = np.array(data["bias1"])
        self.weights2 = np.array(data["weights2"])
        self.bias2 = np.array(data["bias2"])
        self.mean = np.array(data["mean"])
        self.std = np.array(data["std"])
        self.classes = data.get("classes", ["Класс 0", "Класс 1", "Класс 2"])


    def predict(self, x):
        x_normalized = (x - self.mean) / self.std
        hidden = sigmoid(x_normalized @ self.weights1 + self.bias1)
        output = sigmoid(hidden @ self.weights2 + self.bias2)
        return output


class RegressionNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def load_from_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        self.weights1 = np.array(data["weights1"])
        self.bias1 = np.array(data["bias1"])
        self.weights2 = np.array(data["weights2"])
        self.bias2 = np.array(data["bias2"])
        self.mean = np.array(data["mean"])
        self.std = np.array(data["std"])
        self.target_mean = data["target_mean"]
        self.target_std = data["target_std"]

    def predict(self, x):
        x_normalized = (x - self.mean) / self.std
        hidden = sigmoid(x_normalized @ self.weights1 + self.bias1)
        output = hidden @ self.weights2 + self.bias2
        real_prediction = output[0] * self.target_std + self.target_mean
        return real_prediction


wine_model = NeuralNetwork(13, 1, 3)
wine_model.load_from_json("wine_weights.json")

housing_model = RegressionNetwork(8, 8, 1)
housing_model.load_from_json("housing_regression_weights.json")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/classification")
async def classification_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={
            "prediction": None,
            "features": [
                "Алкоголь", "Яблочная кислота", "Зола",
                "Щелочность золы", "Магний", "Общие фенолы",
                "Флаваноиды", "Нефлаваноидные фенолы",
                "Проантоцианины", "Интенсивность цвета",
                "Оттенок", "OD280/OD315 разбавленных вин", "Пролин"
            ]
        }
    )


@app.post("/classification")
async def classify_wine(
    request: Request,
    alcohol: float = Form(...),
    malic_acid: float = Form(...),
    ash: float = Form(...),
    alcalinity_of_ash: float = Form(...),
    magnesium: float = Form(...),
    total_phenols: float = Form(...),
    flavanoids: float = Form(...),
    nonflavanoid_phenols: float = Form(...),
    proanthocyanins: float = Form(...),
    color_intensity: float = Form(...),
    hue: float = Form(...),
    od280_od315: float = Form(...),
    proline: float = Form(...)
):
    features = np.array([
        alcohol, malic_acid, ash, alcalinity_of_ash,
        magnesium, total_phenols, flavanoids,
        nonflavanoid_phenols, proanthocyanins,
        color_intensity, hue, od280_od315, proline
    ])
    
    output = wine_model.predict(features)
    predicted_class_idx = np.argmax(output)
    predicted_class_name = wine_model.classes[predicted_class_idx]
    probabilities = output.tolist()
    
    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={
            "prediction": predicted_class_name,
            "probabilities": probabilities,
            "features": [
                "Алкоголь", "Яблочная кислота", "Зола",
                "Щелочность золы", "Магний", "Общие фенолы",
                "Флаваноиды", "Нефлаваноидные фенолы",
                "Проантоцианины", "Интенсивность цвета",
                "Оттенок", "OD280/OD315 разбавленных вин", "Пролин"
            ],
            "input_values": {
                "alcohol": alcohol,
                "malic_acid": malic_acid,
                "ash": ash,
                "alcalinity_of_ash": alcalinity_of_ash,
                "magnesium": magnesium,
                "total_phenols": total_phenols,
                "flavanoids": flavanoids,
                "nonflavanoid_phenols": nonflavanoid_phenols,
                "proanthocyanins": proanthocyanins,
                "color_intensity": color_intensity,
                "hue": hue,
                "od280_od315": od280_od315,
                "proline": proline
            }
        }
    )


@app.get("/regression")
async def regression_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={
            "prediction": None,
            "features": [
                "Доход в районе (в десятках тысяч $)",
                "Средний возраст домов (лет)",
                "Среднее количество комнат",
                "Среднее количество спален",
                "Население района",
                "Среднее количество членов семьи",
                "Широта",
                "Долгота"
            ]
        }
    )


@app.post("/regression")
async def predict_house_price(
    request: Request,
    MedInc: float = Form(...),
    HouseAge: float = Form(...),
    AveRooms: float = Form(...),
    AveBedrms: float = Form(...),
    Population: float = Form(...),
    AveOccup: float = Form(...),
    Latitude: float = Form(...),
    Longitude: float = Form(...)
):
    features = np.array([
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ])
    
    predicted_price = housing_model.predict(features)
    predicted_price_dollars = predicted_price * 100000
    
    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={
            "prediction": predicted_price_dollars,
            "features": [
                "Доход в районе (в десятках тысяч $)",
                "Средний возраст домов (лет)",
                "Среднее количество комнат",
                "Среднее количество спален",
                "Население района",
                "Среднее количество членов семьи",
                "Широта",
                "Долгота"
            ],
            "input_values": {
                "MedInc": MedInc,
                "HouseAge": HouseAge,
                "AveRooms": AveRooms,
                "AveBedrms": AveBedrms,
                "Population": Population,
                "AveOccup": AveOccup,
                "Latitude": Latitude,
                "Longitude": Longitude
            }
        }
    )


if __name__ == "__main__":
    print("http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)