import os
import settings
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import geofast
from geopy.geocoders import Bing
from PIL import Image

st.title('São Paulo Rent Prices Prediction App')

img = Image.open('housesbanner.png')
st.image(img)

st.text(" ")
st.text(" ")

# CREATE ADDRESS
st.sidebar.header('User Input Features')
address = st.sidebar.text_input("Address", 'Av. Paulista')
address = address + ' São Paulo'

# CREATE LOCATION
geolocator = Bing(api_key="AjTXqeqA-_HoBIIZVhCHWAf9Z1-OVtMK-_wMzsk6ai8jyEpw1GS39hqJvCFKpHfG")
location = geolocator.geocode(address)
lat = location.latitude
lon = location.longitude

map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data)

head = [{'latitude': lat, 'longitude': lon}]
df = pd.DataFrame(head)


# Modelo
def load_model():

    joblib_file = "blended1.pkl"
    gbr_reg = joblib.load(joblib_file)
    predict1 = np.floor(np.expm1(gbr_reg.predict(df.values)))

    return predict1


# Funções de distância
def distance(dataframe):
    # Calcula a distância mais próxima
    data = ['dist_subway.csv', 'dist_bus.csv', 'dist_school.csv']
    rows = dataframe.index
    for dados in data:
        stops = pd.read_csv(os.path.join(settings.dir_sp, dados))
        dataframe[dados] = np.nan
        for n in rows:
            h = dataframe.loc[n, ["latitude", "longitude"]]
            dataframe.loc[n, dados] = geofast.distance(h.latitude, h.longitude,
                                                       stops.latitude, stops.longitude).min()

    # Calcula os pontos em uma localidade de 1km
    data = ['culture.csv', 'crime.csv', 'restaurant.csv']

    for dados in data:
        region = pd.read_csv(os.path.join(settings.dir_sp, dados))
        dataframe[dados] = np.nan
        for n in rows:
            h = dataframe.loc[n, ["longitude", "latitude"]]
            h500 = geofast.distance(region.latitude, region.longitude, h.latitude, h.longitude)
            dataframe[dados][n] = sum(h500 < 700)

    dataframe['crime.csv'] = dataframe['crime.csv'] * 100 / 12377
    return dataframe


df1 = distance(df)

# Variáveis
type = st.sidebar.selectbox('Tipo', list(['Apartamento', 'Casa']))
floor_area = st.sidebar.number_input("Area Total (m²)", 20, value=60)
bedrooms = st.sidebar.slider("Quantos", 0, 15, 1)
bathrooms = st.sidebar.slider("Banheiro", 0, 15, 1)
vacancies = st.sidebar.slider("Estacionamento", 0, 15, 1)
suite = st.sidebar.selectbox('Suíte', list(['Não', 'Sim']))
duplex = st.sidebar.selectbox('Duplex', list(['Não', 'Sim']))
office = st.sidebar.selectbox('Escritório', list(['Não', 'Sim']))
furnished = st.sidebar.selectbox('Mobiliado', list(['Não', 'Sim']))
lounge = st.sidebar.selectbox('Salão de Festas', list(['Não', 'Sim']))
balcony = st.sidebar.selectbox('Varanda', list(['Não', 'Sim']))
renovated = st.sidebar.selectbox('Reformado', list(['Não', 'Sim']))
townhouse = st.sidebar.selectbox('Sobrado', list(['Não', 'Sim']))
air = st.sidebar.selectbox('Ar Condionado', list(['Não', 'Sim']))
barbecue = st.sidebar.selectbox('Churrasqueira', list(['Não', 'Sim']))
btn_predict = st.sidebar.button("MAKE PREDICTION")

if btn_predict:

    head = [{'Type': model, 'Total Area': floor_area, 'Bathrooms': bedrooms, 'Bedrooms': bathrooms,
             'Vacancies': vacancies, 'suíte': suite, 'mobiliado': furnished, 'churrasqueira': barbecue,
             'salão': lounge, 'varanda': balcony, 'duplex': duplex, 'reformado': renovated, 'sobrado': townhouse,
             'condicionado': air, 'escritorio': office}]

    df2 = pd.DataFrame(head)
    df = pd.concat([df1, df2], axis=1, join="inner")


    def rename(dataframe):

        dataframe = dataframe.rename(columns={'Type': 'Type_house', 'dist_bus.csv': 'dist_bus',
                                              'dist_subway.csv': 'dist_subway', 'dist_school.csv': 'dist_school',
                                              'culture.csv': 'ncult', 'crime.csv': 'crime%', 'restaurant.csv': 'food'})

        dataframe = dataframe[['Total Area', 'Bathrooms', 'Bedrooms', 'Vacancies', 'dist_subway', 'dist_bus',
                               'dist_school', 'ncult', 'food', 'crime%', 'latitude', 'longitude', 'suíte', 'mobiliado',
                               'churrasqueira', 'salão', 'varanda', 'duplex', 'reformado', 'sobrado', 'condicionado',
                               'escritorio', 'Type_house']]

        return dataframe


    df = rename(df)
    print(df)

    for i in range(12, 22):
        df.iloc[:, i] = df.iloc[:, i].map({'Yes': 1, 'No': 0})

    df['Type_house'] = df['Type_house'].map({'Casa': 1, 'Apartamento': 0})

    # Predict
    st.header('O valor do aluguel é: **R$%s**' % ("{:,}".format(int(load_model()))))
