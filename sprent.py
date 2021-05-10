import os
import settings
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import geofast
from geopy.geocoders import Bing
#from geopy.geocoders import Nominatim
#from geopy.extra.rate_limiter import RateLimiter

st.title('São Paulo Rent Prices Prediction App')


from PIL import Image
img = Image.open('housesbanner.png')
st.image(img)

st.text(" ")
st.text(" ")

## CREATE ADDRESS
st.sidebar.header('User Input Features')
address = st.sidebar.text_input("Address", 'Av. Paulista')
address = address + ' São Paulo'

## CREATE LOCATION
geolocator = Bing(api_key="AjTXqeqA-_HoBIIZVhCHWAf9Z1-OVtMK-_wMzsk6ai8jyEpw1GS39hqJvCFKpHfG")
#geocode = RateLimiter(geolocator.geocode, timeout=10)
location = geolocator.geocode(address)
lat = location.latitude
lon = location.longitude

map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data)

head = [{'latitude': lat, 'longitude': lon}]
df = pd.DataFrame(head)

# Model
#@st.cache(allow_output_mutation=True)
def load_model():
    joblib_file = "blended1.pkl"
    gbr_reg = joblib.load(joblib_file)
    predict1 = np.floor(np.expm1(gbr_reg.predict(df.values)))
    return predict1

# Funções de distância
def distance(df):
    #Calcula a distância mais próxima
    data = ['dist_subway.csv', 'dist_bus.csv', 'dist_school.csv']
    rows = df.index
    for i in data:
        stops = pd.read_csv(os.path.join(settings.dir_sp, i))
        df[i] = np.nan
        for n in rows:
            h = df.loc[n, ["latitude", "longitude"]]
            df.loc[n, i] = geofast.distance(h.latitude, h.longitude, stops.latitude, stops.longitude).min()

    # Calcula os pontos em uma localidade de 1km
    data = ['culture.csv', 'crime.csv', 'restaurant.csv']
    for i in data:
        region = pd.read_csv(os.path.join(settings.dir_sp, i))
        df[i] = np.nan
        for n in rows:
            h = df.loc[n, ["longitude", "latitude"]]
            h500 = geofast.distance(region.latitude, region.longitude, h.latitude, h.longitude)
            df[i][n] = sum(h500 < 700)
    df['crime.csv'] = df['crime.csv'] * 100 / 12377
    return df

df1 = distance(df)

## VARIABLES
model = st.sidebar.selectbox('Type', list(['Apartament', 'House']))
floor_area = st.sidebar.slider("Total Area (m²)", 15, 500, 80)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 1)
bathrooms = st.sidebar.slider("Bathrooms", 0, 10, 1)
vacancies = st.sidebar.slider("Vacancies", 0, 10, 1)
suite = st.sidebar.selectbox('suite', list(['No', 'Yes']))
furnished = st.sidebar.selectbox('furnished', list(['No', 'Yes']))
barbecue = st.sidebar.selectbox('barbecue', list(['No', 'Yes']))
lounge = st.sidebar.selectbox('lounge', list(['No', 'Yes']))
balcony = st.sidebar.selectbox('balcony', list(['No', 'Yes']))
duplex = st.sidebar.selectbox('duplex', list(['No', 'Yes']))
renovated = st.sidebar.selectbox('renovated', list(['No', 'Yes']))
townhouse = st.sidebar.selectbox('townhouse', list(['No', 'Yes']))
air = st.sidebar.selectbox('air conditioning', list(['No', 'Yes']))
office = st.sidebar.selectbox('office', list(['No', 'Yes']))

head = [{'Type': model, 'Total Area': floor_area, 'Bathrooms': bedrooms, 'Bedrooms': bathrooms,
         'Vacancies': vacancies, 'suíte': suite, 'mobiliado': furnished, 'churrasqueira': barbecue,
         'salão': lounge, 'varanda': balcony, 'duplex': duplex, 'reformado': renovated, 'sobrado': townhouse,
         'condicionado': air, 'escritorio': office}]

df2 = pd.DataFrame(head)
df = pd.concat([df1, df2], axis=1, join="inner")

def rename(df):

    df = df.rename(columns={'Type': 'Type_house', 'dist_bus.csv': 'dist_bus', 'dist_subway.csv': 'dist_subway',
                            'dist_school.csv': 'dist_school', 'culture.csv': 'ncult',
                            'crime.csv': 'crime%', 'restaurant.csv': 'food'})

    df = df[['Total Area', 'Bathrooms', 'Bedrooms', 'Vacancies', 'dist_subway', 'dist_bus', 'dist_school',
             'ncult', 'food', 'crime%', 'latitude', 'longitude', 'suíte', 'mobiliado', 'churrasqueira', 'salão',
             'varanda', 'duplex', 'reformado', 'sobrado', 'condicionado', 'escritorio', 'Type_house']]
    return df
df = rename(df)
for i in range(12, 22):
    df.iloc[:, i] = df.iloc[:, i].map({'Yes': 1, 'No': 0})

df['Type_house'] = df['Type_house'].map({'House': 1, 'Apartament': 0})

## PREDICT
st.header('Predicted Rent Price is **R$%s**' % ("{:,}".format(int(load_model()))))