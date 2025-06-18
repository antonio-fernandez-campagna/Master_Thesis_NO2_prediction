""" 

*Contexto de esta web:*

- Permite visualizar los datos de los sensores de NO2 y tráfico de la ciudad de Madrid.
- Permite analizar los datos de los sensores de NO2 y tráfico de la ciudad de Madrid.
- Permite entrenar modelos de machine learning para predecir los datos de los sensores de NO2 y tráfico de la ciudad de Madrid.
- Permite visualizar los modelos de machine learning entrenados.
- Permite descargar los datos de los sensores de NO2 y tráfico de la ciudad de Madrid.
- Permite descargar los modelos de machine learning entrenados.
- Permite descargar los datos de los sensores de NO2 y tráfico de la ciudad de Madrid.


Datos descargados:


- Datos de los sobre la calidad del Aire: https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=f3c0f7d512273410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default
    - Granularidad: horaria
- Ubicaciones de las estaciones de control del aire: https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=9e42c176313eb410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default
- Datos de los sensores de tráfico de la ciudad de Madrid desde 2013 hasta 2024: https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=33cb30c367e78410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default
-  Granularidad: 15 minutos
- Ubicacion de los sensores de trafico: https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=ee941ce6ba6d3410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD
- Datos de la meteorología de la ciudad de Madrid: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
    - Granularidad: horaria



*Tratamiento de los datos:*

Trafico:

- Dado que los datos proporcionados son a nivel de 15 minutos, para cada sensor de tráfico se ha calculado:
    - Se calcula la **intensidad** como promedio de la intensidad en los 4 intervalos de 15 minutos.
    - Se calcula la **carga** como un promedio ponderado de los valores de carga usando intensidad como peso.
    - Se calcula la **ocupación** como un promedio ponderado usando la intensidad como peso.
    - Se calcula la **velocidad media (vmed)** como un promedio ponderado usando la intensidad como peso.

- Limpiado: Solo se han considerado los datos de los sensores de tráfico que no tienen errores.

Calidad del aire:

- Limpiado: Solo se han considerado los datos de los sensores de tráfico que no tienen errores.

Metereologicos:
- Suponemos que todos los valores son correctos ya que vienen de un dataset oficial.


Qué sensores de NO2 se han considerado?
- Posterior al analisis, se ha decidido eliminar todos los datos anteriores a 2018 por existir una gran ausencia de datos y por una gran dificultad en el mapeo de los sensores de NO2. (sensores con id's repetidos, cambios de posicion, etc)
- Los que al menos tengan un sensor de trafico cercano, a menos de 200.
    - Si hay mas de un sensor, se ha considerado el que tenga más datos.


*Variables a analizar:*
- Variables temporales:Se ha creado el sin y cos de la hora, dia de la semana y mes porque... <explicar>
- Datos de tráfico:
    - Carga
    - Intensidad
    - Ocupacion
    - Velocidad media
- Datos de meteorología: <explicar que es cada cosa y por que se han utilizado en el modelo, por quépuede ayudar?>
    - d2m
    - t2m
    - ssr
    - ssrd
    - u10
    - sp
    - tp
- Valor No2: variable a predecir



Hipotesis:

1. La lluvia reduce los niveles de NO₂
Hipótesis: La precipitación arrastra contaminantes atmosféricos, incluyendo NO₂, y contribuye a su deposición húmeda.

📚 Referencia:

Zhang, Y., et al. (2004). “Simulation of summertime ozone and related pollutants over North America using CMAQ.” Atmospheric Environment.
→ Se observa que episodios de lluvia reducen significativamente las concentraciones de NO₂ y PM.

📊 Gráfica recomendada:

Scatterplot o boxplot: tp (precipitación) vs NO₂

Line plot superpuesto: días con y sin lluvia

🌬️ 2. El viento dispersa el NO₂ y reduce su concentración
Hipótesis: A mayor velocidad del viento, mayor dispersión de los contaminantes, lo que lleva a menores niveles de NO₂.

📚 Referencia:

Kukkonen, J., et al. (2003). “Evaluation of the dispersion model CAR-FMI against measurements near a major road.” Atmospheric Environment.
→ Relación negativa entre velocidad del viento y concentración de NO₂ cerca de carreteras.

📊 Gráfica recomendada:

Scatterplot: u10 (velocidad viento) vs NO₂

Heatmap: NO₂ en función de dirección y velocidad del viento

☁️ 3. Altas temperaturas favorecen la formación de ozono y pueden reducir el NO₂ por fotólisis
Hipótesis: El NO₂ se descompone con luz solar en presencia de temperaturas altas, lo que favorece formación de ozono y reduce el NO₂.

📚 Referencia:

Finlayson-Pitts, B. J., & Pitts Jr, J. N. (2000). Chemistry of the upper and lower atmosphere.
→ Describe el mecanismo de fotólisis del NO₂ bajo radiación solar intensa.

📊 Gráfica recomendada:

Scatterplot: t2m (temperatura) vs NO₂

Boxplot mensual: para ver cómo cambia el NO₂ con estaciones cálidas/frías

☀️ 4. Mayor radiación solar reduce NO₂ por fotólisis
Hipótesis: La radiación solar descompone NO₂ → NO + O, lo que puede disminuir su concentración a lo largo del día.

📚 Referencia:

Seinfeld, J.H., & Pandis, S.N. (2016). Atmospheric Chemistry and Physics: From Air Pollution to Climate Change.
→ Relación entre irradiancia solar y fotólisis de NO₂.

📊 Gráfica recomendada:

Scatterplot: ssr o ssrd vs NO₂

Boxplot por hora del día para ver evolución de NO₂ vs luz solar

🚗 5. Mayor intensidad de tráfico incrementa el NO₂
Hipótesis: El tráfico rodado es la principal fuente de NO₂ en entornos urbanos.

📚 Referencia:

Cyrys, J., et al. (2003). “Relationship between different traffic-related particles in the city of Munich.” Science of the Total Environment.
→ Relación directa entre intensidad de tráfico y concentración de NO₂.

📊 Gráfica recomendada:

Scatterplot: intensidad, ocupación, carga vs NO₂

Boxplots por hora del día o día de la semana

🕒 6. NO₂ presenta un ciclo diario y semanal por la actividad humana
Hipótesis: NO₂ aumenta en horas punta (mañana y tarde) y disminuye por la noche. También baja los fines de semana.

📚 Referencia:

Vardoulakis, S., et al. (2003). “Modelling air quality in street canyons: a review.” Atmospheric Environment.
→ Muestra patrones de concentración de NO₂ relacionados con horarios de tráfico.

📊 Gráfica recomendada:

Boxplot de NO₂ por hora del día

Boxplot de NO₂ por día de la semana

Heatmap hora vs día con media de NO₂

💨 7. La presión atmosférica puede estar relacionada con acumulación de contaminantes
Hipótesis: Altas presiones tienden a producir condiciones de estancamiento del aire → acumulación de contaminantes como el NO₂.

📚 Referencia:

Jacob, D. J., & Winner, D. A. (2009). “Effect of climate change on air quality.” Atmospheric Environment.
→ Explica cómo ciertos sistemas de presión pueden atrapar contaminantes a nivel del suelo.

📊 Gráfica recomendada:

Scatterplot: sp (presión) vs NO₂

🧊 8. Humedad relativa y punto de rocío pueden influir en la química del NO₂
Hipótesis: Aunque no es lineal, la humedad puede modular la deposición húmeda o la formación de aerosoles secundarios que afectan el NO₂.

📚 Referencia:

Beig, G., et al. (2007). “Role of meteorology in modulating air quality during a winter haze episode in Delhi, India.”
→ Analiza la influencia de la humedad y el punto de rocío en contaminantes como NO₂.

📊 Gráfica recomendada:

Scatterplot: d2m (punto de rocío) vs NO₂

Comparación día húmedo vs seco





Analisis de los datos:

- Analisis temporal:
    - Agregacion: 
        - Horaria
        - Diaria
        - Semanal
        - Mensual
- Correlaciones
    - Agregacion: 
        - Horaria
        - Diaria
        - Semanal
        - Mensual





*Modelo GAM para entender el NO2:*
<Explicar aqui por que se ha decidido entrenar un GAM, por qué ayuda a entender el analisis.>
Metodologia:
- Seleccionar el sensor de NO2.
- Seleccionar peroido de entrenamiento
- Seleccionar tipo de filtrado: 
    - Zscore <explicar>
    - Quantiles <explicar>
    - IQR: <explicar>
    - Ninguno.
- Seleccionar las variables a usar para entrenar el modelo:
    - Variables de tráfico:
        - Intensidad
        - Carga
        - Ocupacion
        - Velocidad media
    - Variables de meteorología:
        - Temperatura
        - Humedad


Resultados:



*Modelo de Machine Learning:*
<Explicar aqui por que se ha decidido entrenar un modelo de Machine Learning, por qué un XGBoost para este tipo de problema.>
Metodologia:
- Seleccionar el sensor de NO2.
- Seleccionar peroido de entrenamiento
- Seleccionar tipo de filtrado de outliers: 
    - Zscore <explicar>
    - Quantiles <explicar>
    - IQR: <explicar>
    - Ninguno.
- Preprocesamiento temporal:
    - Sin y cos
    - nada 
- Seleccionar las variables a usar para entrenar el modelo:
    

Resultados:


*Simulaciones*




        
"""