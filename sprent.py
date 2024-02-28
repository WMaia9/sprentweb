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
st.sidebar.header('Property Characteristics')
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


# Model
def load_model():

    joblib_file = "blended1.pkl"
    gbr_reg = joblib.load(joblib_file)
    predict1 = np.floor(np.expm1(gbr_reg.predict(df.values)))

    return predict1


# Distance Functions
def distance(dataframe):
    # Calculate closest distance
    data = ['dist_subway.csv', 'dist_bus.csv', 'dist_school.csv']
    rows = dataframe.index
    for data_file in data:
        stops = pd.read_csv(os.path.join(settings.dir_sp, data_file))
        dataframe[data_file] = np.nan
        for n in rows:
            h = dataframe.loc[n, ["latitude", "longitude"]]
            dataframe.loc[n, data_file] = geofast.distance(h.latitude, h.longitude,
                                                           stops.latitude, stops.longitude).min()

    # Calculate points within 1km area
    data = ['culture.csv', 'crime.csv', 'restaurant.csv']

    for data_file in data:
        region = pd.read_csv(os.path.join(settings.dir_sp, data_file))
        dataframe[data_file] = np.nan
        for n in rows:
            h = dataframe.loc[:, ["longitude", "latitude"]].to_numpy()  # Convert to NumPy array
            h500 = geofast.distance(region.latitude.to_numpy(), region.longitude.to_numpy(), h[:, 0], h[:, 1]) 
            dataframe.loc[n, data_file] = sum(h500 < 700)

    dataframe['crime.csv'] = dataframe['crime.csv'] * 100 / 12377
    return dataframe


df1 = distance(df)

# Variables
model = st.sidebar.selectbox('Type', list(['Apartment', 'House']))
floor_area = st.sidebar.number_input("Total Area (m²)", 20, value=60)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 1)
bathrooms = st.sidebar.slider("Bathrooms", 0, 10, 1)
vacancies = st.sidebar.slider("Parking", 0, 10, 1)
suite = st.sidebar.selectbox('Suite', list(['No', 'Yes']))
duplex = st.sidebar.selectbox('Duplex', list(['No', 'Yes']))
office = st.sidebar.selectbox('Office', list(['No', 'Yes']))
furnished = st.sidebar.selectbox('Furnished', list(['No', 'Yes']))
lounge = st.sidebar.selectbox('Party Room', list(['No', 'Yes']))
balcony = st.sidebar.selectbox('Balcony', list(['No', 'Yes']))
renovated = st.sidebar.selectbox('Renovated', list(['No', 'Yes']))
townhouse = st.sidebar.selectbox('Townhouse', list(['No', 'Yes']))
air = st.sidebar.selectbox('Air Conditioning', list(['No', 'Yes']))
barbecue = st.sidebar.selectbox('Barbecue', list(['No', 'Yes']))
btn_predict = st.sidebar.button("CALCULATE RENT")

if btn_predict:

    head = [{'Type': model, 'Total Area': floor_area, 'Bathrooms': bathrooms, 'Bedrooms': bedrooms,
             'Vacancies': vacancies, 'suite': suite, 'furnished': furnished, 'barbecue': barbecue,
             'lounge': lounge, 'balcony': balcony, 'duplex': duplex, 'renovated': renovated, 'townhouse': townhouse,
             'air': air, 'office': office}]

    df2 = pd.DataFrame(head)
    df = pd.concat([df1, df2], axis=1, join="inner")


    def rename(dataframe):

        dataframe = dataframe.rename(columns={'Type': 'Type_house', 'dist_bus.csv': 'dist_bus',
                                              'dist_subway.csv': 'dist_subway', 'dist_school.csv': 'dist_school',
                                              'culture.csv': 'culture', 'crime.csv': 'crime%', 'restaurant.csv': 'food'})

        dataframe = dataframe[['Total Area', 'Bathrooms', 'Bedrooms', 'Vacancies', 'dist_subway', 'dist_bus',
                               'dist_school', 'culture', 'food', 'crime%', 'latitude', 'longitude', 'suite', 'furnished',
                               'barbecue', 'lounge', 'balcony', 'duplex', 'renovated', 'townhouse', 'air',
                               'office', 'Type_house']]

        return dataframe


    df = rename(df)
    print(df)

    for i in range(12, 22):
        df.iloc[:, i] = df.iloc[:, i].map({'Yes': 1, 'No': 0})

    df['Type_house'] = df['Type_house'].map({'House': 1, 'Apartment': 0})

    # Predict
    prediction = load_model()[0]
    st.header('The Rent Value is: **R$%s**' % ("{:,}".format(int(prediction))))
