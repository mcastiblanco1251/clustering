import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import datasets


#im = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/arroz.png')
im2 = Image.open('r1.jpg')
st.set_page_config(page_title='Cluster-App', layout="wide", page_icon=im2)
st.set_option('deprecation.showPyplotGlobalUse', False)

row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    image = Image.open('r1.jpg')
    st.image(image, use_column_width=True)
    st.markdown('Web App by [Manuel Castiblanco](https://github.com/mcastiblanco1251)')
with row1_2:
    st.write("""
    # Agrupación - Clustering App
    Esta App utiliza algoritmos de Machine Learning para hacer Agrupar segmentos de un mercado   !
    """)
    with st.expander("Contáctanos 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name = st.text_input('Name')
            mail = st.text_input('Email')
            q = st.text_area("Query")

            submit_button = st.form_submit_button(label='Send')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com", 587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n' + name + '\n' + mail + '\n' + q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()

st.header('Aplicación')
st.markdown('____________________________________________________________________')
app_des = st.expander('Descripción App')
with app_des:
    st.write("""Esta aplicación muestra como se puede segmentar un mercado, el dataset es provisto por Kaggle, en el
    cual se quiere segmentar el mercado
    """)

st.sidebar.header('Parámetros de Entrada Usario')

# st.sidebar.markdown("""
# [Example CSV input file](penguins_example.csv)
# """)

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Cargue sus parámetros desde un archivo CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv('Mall_Customers.csv')
    #df.rename(index=str, columns={'Annual Income (k$)': 'Ingreso',
    #                          'Spending Score (1-100)': 'Puntaje'}, inplace=True)
    st.subheader ('Agrupación')
    st.write('**Tabla de Datos**')
    st.table(df.head())
    var1=st.multiselect('Variables Desechadas', df.columns, df.columns[0])
    #var1
    X = df.drop(var1, axis=1)
    st.table(X.head(5))
    var2=st.multiselect('Variables a analizar', df.columns, df.columns[1])
    #var2
    g=sns.pairplot(X, hue=var2[0], aspect=1.5)
    st.write('**Correlación**')
    st.pyplot(g)

    st.subheader ('Selección Número Ideal de Agrupaciones o Clústers')
    clusters = []
    X = X.drop(var2, axis=1)
    #X
    nc=st.number_input('Numero de Cluters a revisar',min_value=0,  value= 11)
    for i in range(1, nc):
        km = KMeans(n_clusters=i).fit(X)
        clusters.append(km.inertia_)

    if st.checkbox("Ver ejemplo para el Data set Demo para ubicar el No. de Cluster Ideal por Método del Codo",value=False):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
        ax.set_title('Búsqueda por Codo')
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Inercia')

        # Annotate arrow
        ax.annotate('Posible punto de Codo', xy=(3, 140000), xytext=(3, 50000), xycoords='data',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

        ax.annotate('Posible punto de Codo', xy=(5, 80000), xytext=(5, 150000), xycoords='data',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

        st.pyplot(plt.show())

    st.subheader('Cluster Seleccionados')
    row1_3, row1_4 = st.columns((2, 2))

    with row1_3:
        n=st.number_input('Cluster 1', value=1)
        kmn = KMeans(n_clusters=n).fit(X)

        X['Labels'] = kmn.labels_

        var3=st.multiselect('Variable Eje X', df.columns, df.columns[3])
        var4=st.multiselect('Variable Eje Y', df.columns, df.columns[4])
        plt.figure(figsize=(12, 8))
        sns.scatterplot(X[var3[0]], X[var4[0]], hue=X['Labels'],
                        palette=sns.color_palette('hls', n))
        #sns.scatterplot(kmn.cluster_centers_[:, 0], kmn.cluster_centers_[:, 1],
        #                hue=range(n), s=200, ec='black', legend=False)
        plt.title(f'KMeans con {n} Clusters')
        st.pyplot(plt.show())
    with row1_4:
        n=int(st.number_input('Cluster 2', value=1))
        kmn = KMeans(n_clusters=n).fit(X)
        var5=st.multiselect('Variable Eje X2', df.columns, df.columns[3])
        var6=st.multiselect('Variable Eje Y2', df.columns, df.columns[4])
        X['Labels'] = kmn.labels_
        plt.figure(figsize=(12, 8))
        sns.scatterplot(X[var5[0]], X[var6[0]], hue=X['Labels'],
                        palette=sns.color_palette('hls', n))
        #sns.scatterplot(kmn.cluster_centers_[:, 0], kmn.cluster_centers_[:, 1],
        #                hue=range(n), s=200, ec='black', legend=False)
        plt.title(f'KMeans con {n} Clusters')
        st.pyplot(plt.show())
    app_desc = st.expander('Conclusión App')
    with app_desc:
        st.write("""
                Para este Análisis, podríamos decir que el grupo 5 parece mejor que el grupo 3. Como este es un problema no supervisado,
                no podemos saber con certeza cuál es el mejor en la vida real, pero al observar los datos, es seguro decir que 5 sería nuestra elección.

                Podemos analizar nuestros 5 clústeres en detalle ahora:

                - La etiqueta 0 es de bajos ingresos y bajos gastos.
                - La etiqueta 1 es de altos ingresos y altos gastos
                - La etiqueta 2 es ingreso medio y gasto medio
                - La etiqueta 3 es ingresos altos y gastos bajos
                - La etiqueta 4 es de bajos ingresos y alto gasto.

                *Esto le sirve de guía para otros análisis de DataSet*
        """)
    if st.checkbox("Complementar Visual de Cluster",value=False):
        fig = plt.figure(figsize=(20,8))
        #x1=st.multiselect('Variable Eje x1', df.columns, df.columns[4])
        y1=st.multiselect('Variable Eje y1', df.columns, df.columns[3])
        y2=st.multiselect('Variable Eje y2', df.columns, df.columns[4])
        ax = fig.add_subplot(121)

        sns.swarmplot(x='Labels', y=y1[0], data=X, ax=ax)
        ax.set_title(f'Etiquetas de acuerdo a {y1[0]}')
        ax = fig.add_subplot(122)
        sns.swarmplot(x='Labels', y=y2[0], data=X, ax=ax)
        ax.set_title(f'Etiquetas de {y2[0]}')

        st.pyplot(plt.show())

with st.expander("Contáctanos👉"):
    st.subheader('Quieres conocer mas de IA, ML o DL 👉[contactanos!!](http://ia.smartecorganic.com.co/index.php/contact/)')
