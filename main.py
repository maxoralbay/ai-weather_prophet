import os
import pickle
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta

# Соңғы енгізілген деректерді сақтау файлы
PICKLE_FILE = "last_input.pkl"

# Соңғы енгізілген деректерді сақтау
def save_last_input(data):
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(data, f)

# Соңғы енгізілген деректерді жүктеу
def load_last_input():
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

# WeatherAPI-дан ауа райы деректерін алу
def fetch_weather_data(api_key, city, start_date, end_date):
    try:
        base_url = f"http://api.weatherapi.com/v1/history.json"
        weather_data = []
        current_date = start_date

        while current_date <= end_date:
            formatted_date = current_date.strftime("%Y-%m-%d")
            response = requests.get(
                f"{base_url}?key={api_key}&q={city}&dt={formatted_date}"
            )
            if response.status_code == 200:
                data = response.json()
                if 'forecast' in data and 'forecastday' in data['forecast']:
                    day_data = data['forecast']['forecastday'][0]['day']
                    weather_data.append({
                        'Күні': current_date.strftime("%Y-%m-%d"),
                        'Температура': day_data['avgtemp_c'],
                    })
                else:
                    raise ValueError("Болжамдық деректер жоқ")
            else:
                raise ValueError(f"API қатесі: {response.status_code} - {response.text}")
            current_date += timedelta(days=1)

        return pd.DataFrame(weather_data)
    except Exception as e:
        messagebox.showerror("Қате", f"Деректерді алу мүмкін болмады: {e}")
        return pd.DataFrame()

# Prophet арқылы ауа райын болжау
def predict_weather():
    try:
        # Деректерді UI-дан алу
        city = city_entry.get()
        start_date = datetime.strptime(start_date_entry.get(), "%d-%m-%Y")
        end_date = datetime.strptime(end_date_entry.get(), "%d-%m-%Y")
        forecast_days = int(forecast_days_entry.get())
        api_key = api_key_entry.get()

        # Енгізілген деректерді сақтау
        save_last_input({
            "city": city,
            "start_date": start_date_entry.get(),
            "end_date": end_date_entry.get(),
            "forecast_days": forecast_days_entry.get(),
            "api_key": api_key
        })

        # Дата диапазонын тексеру
        if start_date > end_date:
            messagebox.showerror("Қате", "Басталу күні аяқталу күнінен бұрын болуы керек!")
            return

        # Ауа райы деректерін алу
        weather_df = fetch_weather_data(api_key, city, start_date, end_date)
        if weather_df.empty:
            return

        # Prophet үшін деректерді дайындау
        weather_df['Күні'] = pd.to_datetime(weather_df['Күні'])
        weather_df = weather_df.rename(columns={'Күні': 'ds', 'Температура': 'y'})

        # Модельді оқыту
        model = Prophet()
        model.fit(weather_df)

        # Болашақ үшін болжам жасау
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # График салу
        plt.figure(figsize=(10, 6))
        plt.plot(weather_df['ds'], weather_df['y'], label="Тарихи деректер", marker='o')

        plt.plot(
            forecast['ds'],
            forecast['yhat'],
            label="Болжам",
            linestyle='--',
            color='orange'
        )
        plt.axvline(x=weather_df['ds'].iloc[-1], color='red', linestyle=':', label='Болжам басталуы')
        plt.xlabel('Күні')
        plt.ylabel('Температура (°C)')
        plt.title(f"{city} қаласы үшін температура болжамы")
        plt.legend()
        plt.grid()

        # X осіндегі даталарды форматтау
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d-%m-%Y"))

        plt.show()

    except ValueError as e:
        messagebox.showerror("Қате", str(e))
    except Exception as e:
        messagebox.showerror("Қате", f"Бірдеңе дұрыс болмады: {e}")

# Tkinter UI
root = tk.Tk()
root.title("ИИ (Prophet) арқылы ауа райы болжамы")

# Соңғы енгізілген деректерді жүктеу
last_input = load_last_input()

# Енгізу өрістері
tk.Label(root, text="Қала:").grid(row=0, column=0, padx=10, pady=5)
city_entry = tk.Entry(root)
city_entry.insert(0, last_input.get("city", ""))
city_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Басталу күні (КК-АА-ЖЖЖЖ):").grid(row=1, column=0, padx=10, pady=5)
start_date_entry = tk.Entry(root)
start_date_entry.insert(0, last_input.get("start_date", ""))
start_date_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Аяқталу күні (КК-АА-ЖЖЖЖ):").grid(row=2, column=0, padx=10, pady=5)
end_date_entry = tk.Entry(root)
end_date_entry.insert(0, last_input.get("end_date", ""))
end_date_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Болжам күндері:").grid(row=3, column=0, padx=10, pady=5)
forecast_days_entry = tk.Entry(root)
forecast_days_entry.insert(0, last_input.get("forecast_days", ""))
forecast_days_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="WeatherAPI API кілті:").grid(row=4, column=0, padx=10, pady=5)
api_key_entry = tk.Entry(root)
api_key_entry.insert(0, last_input.get("api_key", ""))
api_key_entry.grid(row=4, column=1, padx=10, pady=5)

# Болжам жасау батырмасы
predict_button = ttk.Button(root, text="Болжам жасау", command=predict_weather)
predict_button.grid(row=5, column=0, columnspan=2, pady=10)

# Қолданбаны іске қосу
root.mainloop()
