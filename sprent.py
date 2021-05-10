import os
import settings
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import geofast
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.title('Interactive App to Visualize and Predict São Paulo Rent Prices')

st.text(" ")
st.text(" ")
st.text(" ")


## CREATE USER INPUT SIDBAR
st.sidebar.header('User Input Features')
address = st.sidebar.text_input("Address", 'Av. Paulista')
address = address + ' São Paulo'
model = st.sidebar.selectbox('Type', list(['House', 'Apartament']))
floor_area = st.sidebar.slider("Total Area (m²)", 15,600,93) # floor area
bedrooms = st.sidebar.slider("Bedrooms", 0,12,1)
bathrooms = st.sidebar.slider("Bathrooms", 0,10,1)
vacancies = st.sidebar.slider("Vacancies", 0,10,1)
living_room = st.sidebar.selectbox('living room', list(['No', 'Yes']))
suite = st.sidebar.selectbox('suite', list(['No', 'Yes']))
furnished = st.sidebar.selectbox('furnished', list(['No', 'Yes']))
swimming_pool = st.sidebar.selectbox('swimming pool', list(['No', 'Yes']))
barbecue = st.sidebar.selectbox('barbecue', list(['No', 'Yes']))
lounge = st.sidebar.selectbox('lounge', list(['No', 'Yes']))
balcony = st.sidebar.selectbox('balcony', list(['No', 'Yes']))
gym = st.sidebar.selectbox('gym', list(['No', 'Yes']))
duplex = st.sidebar.selectbox('duplex', list(['No', 'Yes']))
yard = st.sidebar.selectbox('yard', list(['No', 'Yes']))
closet = st.sidebar.selectbox('closet', list(['No', 'Yes']))
renovated = st.sidebar.selectbox('renovated', list(['No', 'Yes']))
townhouse = st.sidebar.selectbox('townhouse', list(['No', 'Yes']))
air = st.sidebar.selectbox('air conditioning', list(['No', 'Yes']))
condominium = st.sidebar.selectbox('condominium', list(['No', 'Yes']))
office = st.sidebar.selectbox('office', list(['No', 'Yes']))

## CREATE LOCATION
geolocator = Nominatim(user_agent="br")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
location = geolocator.geocode(address)
lat = location.latitude
lon = location.longitude

map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data)

head = [{'Type':model, 'Total Area':floor_area, 'Bathrooms':bedrooms,'Bedrooms':bathrooms, 'Vacancies':vacancies,
        'sala':living_room, 'suíte':suite,'mobiliado':furnished, 'piscina':swimming_pool, 'churrasqueira':barbecue,
        'salão':lounge, 'varanda':balcony, 'academia':gym,'duplex':duplex, 'quintal':yard, 'armário':closet,
        'reformado':renovated, 'sobrado':townhouse, 'condicionado':air,'condominio':condominium, 'escritorio':office,
       'latitude':lat, 'longitude':lon}]

df = pd.DataFrame(head)

for i in range(5, 21):
    df.iloc[:,i] = df.iloc[:,i].map({'Yes':1 ,'No':0})

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

df = distance(df)

def rename(df):

    df = df.rename(columns={'Type':'Type_house','dist_bus.csv': 'dist_bus', 'dist_subway.csv': 'dist_subway',
                            'dist_school.csv': 'dist_school', 'culture.csv': 'ncult',
                            'crime.csv': 'crime%', 'restaurant.csv': 'food'})

    df = df[['Total Area', 'Bathrooms',
             'Bedrooms', 'Vacancies', 'dist_subway', 'dist_bus', 'dist_school',
             'ncult', 'food', 'crime%', 'latitude', 'longitude', 'sala', 'suíte',
             'mobiliado', 'piscina', 'churrasqueira', 'salão', 'varanda', 'academia',
             'duplex', 'quintal', 'armário', 'reformado', 'sobrado', 'condicionado',
             'condominio', 'escritorio','Type_house']]
    return df

df = rename(df)
df['Type_house'] = df['Type_house'].map({'House':1 ,'Apartament':0})

## PREDICT
joblib_file = "predict_model.pkl"
gbr_reg = joblib.load(joblib_file)
predict1 = np.floor(np.expm1(gbr_reg.predict(df.values)))
st.header('Predicted Rent Price is **R$%s**' % ("{:,}".format(int(predict1))))