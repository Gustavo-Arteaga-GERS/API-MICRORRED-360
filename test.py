from flask import Blueprint, render_template, request, flash, redirect, url_for, Flask,jsonify,redirect, make_response
import json
import numpy as np
import pandas as pd
import random
import datetime
from pulp import *
import gurobipy as gp
import pymongo
from pymongo import MongoClient

app = Flask(__name__)

# database mongo
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DATABASE = "MR360_DB"
MONGO_COLLECTION ="profiles"
client = pymongo.MongoClient(MONGO_URI)
dataBase_ = client[MONGO_DATABASE]
collection_ = dataBase_[MONGO_COLLECTION]

# line_of_code_with_preset_order_of-data:
myOrder = []
Dem_tmp = []

for ind in collection_.find():
    temp = ind["Demand_6"]
    Dem_tmp.append(temp)
Dem = np.array(Dem_tmp)

for ind in collection_.find():
    temp = ind["Order"]
    myOrder.append(temp)
#myOrder = sum(myOrder, [])
print(myOrder)
print(type(myOrder))
print(len(myOrder))