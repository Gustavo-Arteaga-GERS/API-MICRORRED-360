from flask import Blueprint, render_template, request, flash, redirect, url_for, Flask,jsonify,redirect, make_response

import json


app = Flask(__name__)


#metodos de prueba 
# @app.route("/")

# def hello_world():
#     return "<p>Hello, wolcome to app flask microrred 360 !</p>"

# @app.route("/foo")
# def neplanRequest():
#     return "Welcome to NEPLAN"

@app.route("/calculator" , methods = ['POST'])
def neplanRequest():
    print("**********")
    #lectura de datos
    jsonInput = request.get_json()
    base = jsonInput['base']
    altura = jsonInput['altura']



    #Logica aquí
    area = base*altura

    #estructura de la respuesta
    response = {
        "area rectangulo": area
    }


    res = make_response(jsonify(response), 200)

    return res
    
   


if __name__ == '__main__':
    app.run(debug = True, port = 5000)


