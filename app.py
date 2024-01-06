from flask import Flask, render_template, request
import pandas as pd
from model_RForest import predecir_cliente,reloj_default

def crear_app():

    app = Flask(__name__, template_folder="templates")
    # modelo_forest = joblib.load('modelo_forest.joblib')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predecir', methods=['POST'])
    def predecir():
        if request.method == 'POST':
            datos = {
                'Estado_cuenta': int(request.form['Estado_cuenta']),
                'Historial_credito': int(request.form['Historial_credito']),
                'Proposito': int(request.form['Proposito']),
                'Ahorros': int(request.form['Ahorros']),
                'Empleo_actual': int(request.form['Empleo_actual']),
                'Cuota_como_porcentaje_ingreso': int(request.form['Cuota_como_porcentaje_ingreso']),
                'Otros_deudores': int(request.form['Otros_deudores']),
                'Residencia_actual_desde': int(request.form['Residencia_actual_desde']),
                'Propiedad': int(request.form['Propiedad']),
                'Otros_planes_cuotas': int(request.form['Otros_planes_cuotas']),
                'Vivienda': int(request.form['Vivienda']),
                'Creditos_en_banco': int(request.form['Creditos_en_banco']),
                'Trabajo': int(request.form['Trabajo']),
                'Personas_a_cargo': int(request.form['Personas_a_cargo']),
                'Telefono': int(request.form['Telefono']),
                'Trabajador_extranjero': int(request.form['Trabajador_extranjero']),
                'Rango_edad': int(request.form['Rango_edad']),
                'Rango_valor_credito': int(request.form['Rango_valor_credito']),
                'Rango_plazos_credito': int(request.form['Rango_plazos_credito']),
                'Sexo': int(request.form['Sexo']),
                'Estado_civil': int(request.form['Estado_civil']),
            }
            df = pd.DataFrame.from_dict(datos, orient='index').T
            
            resultado = predecir_cliente(df)
            grafica= reloj_default(resultado[1],resultado[2])

            return render_template('resultado.html', resultado=(resultado[0],resultado[1][0],resultado[2]), Grafica=grafica,acierto=resultado[3])
            # return render_template('resultado.html', resultado=resultado[0], grafica=grafica)
    return app

if __name__ == '__main__':
    app=crear_app()
    app.run(debug=True)
