from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

def fig_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def predecir_cliente(df):
   df_bank = pd.read_csv('df_credit_scoring.csv')

   X_train, X_test, y_train, y_test = train_test_split(df_bank.drop('Incumplimiento', axis=1), df_bank['Incumplimiento'], test_size=0.2, random_state=42)

   # Inicializar y entrenar el modelo
   forest_model = RandomForestClassifier(random_state=42,max_features='sqrt', n_estimators=250)
   forest_model.fit(X_train, y_train)

   probabilidades_predichas = forest_model.predict_proba(df)

   # Seleccionar las probabilidades de la clase 1 (default)
   probabilidades_default = probabilidades_predichas[:, 1]

   # Establecer un umbral para determinar la clase
   umbral = 0.3
   prediccion_binaria = (probabilidades_default > umbral).astype(int)

   # Calcular el porcentaje de acierto para la clase:
   if prediccion_binaria[0] == 1:
      porcentaje_acierto_default = probabilidades_default[prediccion_binaria == 1].mean()
   else:
      porcentaje_acierto_default = probabilidades_default[prediccion_binaria == 0].mean()

   # Imprimir la respuesta con el porcentaje de acierto
   if prediccion_binaria[0] == 1:
      result= f'El Cliente NO es apto. Porcentaje de acierto: {porcentaje_acierto_default * 100:.2f}%'
      porc_acierto=round(porcentaje_acierto_default * 100,2)
   else:
      result=f'El Cliente SI es apto. Porcentaje de acierto: {(1 - porcentaje_acierto_default) * 100:.2f}%'
      porc_acierto=round((1-porcentaje_acierto_default) * 100,2)
   return result,prediccion_binaria,porcentaje_acierto_default,porc_acierto

def reloj_default(prediccion_binaria,porcentaje_acierto_default):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    nuevos_valores = [0.25, 0.25,0.25, 0.25,0.25, 0.25,0.25, 0.25]
    if prediccion_binaria[0] == 1:
        success=int(porcentaje_acierto_default*100)
        # success=70
        arrow_props = dict(arrowstyle='->', linewidth=6, color='orange', headwidth=15, headlength=15, connectionstyle="arc3,rad=0.1")

      #   ax.annotate(f'{success}%', xy=(-1.5,.7),fontsize=20,color='darkorange',fontweight='bold')
        ax.annotate('Cliente NO apto',xy=(-.65, 1.2),color='red', fontsize=28,fontweight='bold')
        if success>0 and success<30:
          ax.annotate("", xy=(0, 0), xytext=(-.15, .6), arrowprops=dict(arrowstyle='<-', linewidth=6, color='gold'))
        elif success>=30 and success<51:
          ax.annotate("", xy=(0, 0), xytext=(-.3, .5), arrowprops=dict(arrowstyle='<-', linewidth=6, color='gold'))
        elif success>=51 and success<65:
          ax.annotate("", xy=(0, 0), xytext=(-.5, .4), arrowprops=dict(arrowstyle='<-', linewidth=6, color='orange'))
        elif success>=65 and success<75:
          ax.annotate("", xy=(0, 0), xytext=(-.6, .2), arrowprops=dict(arrowstyle='<-', linewidth=6, color='orange'))
        elif success<100:
          ax.annotate("", xy=(0, 0), xytext=(-.6, 0.05), arrowprops=dict(arrowstyle='<-', linewidth=6, color='orange'))
    else:
        success=int((1-porcentaje_acierto_default)*100)
        # success=55
        ax.annotate('Cliente SI es apto',xy=(-.78, 1.2),color='forestgreen', fontsize=28,fontweight='bold')
      #   ax.annotate(f'{success}%', xy=(1,.7),fontsize=20,color='forestgreen',fontweight='bold')
        if success>0 and success<30:
          ax.annotate("", xy=(0, 0), xytext=(.15, .6), arrowprops=dict(arrowstyle='<-', linewidth=6, color='limegreen'))
        elif success>=30 and success<51:
          ax.annotate("", xy=(0, 0), xytext=(.3, .5), arrowprops=dict(arrowstyle='<-', linewidth=6, color='limegreen'))
        elif success>=51 and success<65:
          ax.annotate("", xy=(0, 0), xytext=(.48, .38), arrowprops=dict(arrowstyle='<-', linewidth=6, color='forestgreen'))
        elif success>=65 and success<75:
          ax.annotate("", xy=(0, 0), xytext=(.6, .2), arrowprops=dict(arrowstyle='<-', linewidth=6, color='green'))
        elif success<100:
          ax.annotate("", xy=(0, 0), xytext=(.6, 0.05), arrowprops=dict(arrowstyle='<-', linewidth=6, color='green'))

    etiquetas=['Super', 'Bueno','Bajo','Malo','', '','','']
    colores = ['green', 'limegreen', 'yellow', 'orange', 'none', 'none', 'none', 'none']
    ax.pie(nuevos_valores,labels=etiquetas, colors=colores,startangle=0,textprops={'fontsize': 20,'color':'gray','fontweight':'bold'}, wedgeprops=dict(width=0.4))

    # Coordenadas polares para el semicírculo
    theta = np.linspace(0, np.pi, 200)
    x = np.cos(theta)
    y = np.sin(theta)

    theta_1 = np.linspace(0, np.pi*2, 100)
    x_1 = np.cos(theta_1)
    y_1 = np.sin(theta_1)

    # Graficar el semicírculo
    ax.plot(x, y, color='silver', linewidth=12)
    ax.plot(x_1/10, y_1/10, color='silver', linewidth=2)
    ax.plot(x_1/34, y_1/34, color='black', linewidth=6)
   #  ax.plot(x_1/40, y_1/40, color='black', linewidth=3)

    # Graficar la línea de la primera porción de pastel
    ax.plot(x, y, color='black', linewidth=1)
    ax.plot(x/2.2, y/2.2, color='black', linewidth=1)
    # plt.title('CREDIT SCORES',pad=5,color='gray', fontsize=20,fontweight='bold')
   # Hacer el fondo transparente
    fig.set_facecolor('none')
   #  plt.ylim(-.12,max(y)*1.05)
   #  graf = fig_to_base64(fig)
   #  plt.ylim(-.12,max(y)*1.05)
    graf = fig_to_base64(fig)
    return graf