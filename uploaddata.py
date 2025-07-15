pip install pymongo


from pymongo.mongo_client import MongoClient
import pandas as pd
import json

url="mongodb+srv://akshay:akshay1234@cluster0.jrmb9uy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


client=MongoClient(url)


Database_name="sensor"
Collection_name="waferfault"

df=pd.read_csv("C:\Users\sharm\Downloads\sensorproject\notebook\waferfault.csv")

df=df.drop("Unnamed: 0",axis=1)

json_record=list(json.loads(df.T.to_json()).values())


client[Database_name][Collection_name].insert_many(json_record)

