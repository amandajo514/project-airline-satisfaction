import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

#supressing warning that comes when confusion matrix is shown
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page title and icon
st.set_page_config(page_title = "Airline Satisfaction", page_icon = ":small_airplane:")

#Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["🏠 Home","🗄️ Data Overview","📊 Exploritory Data Analysis","📉 Modeling"])

#read in the data
df_train = pd.read_csv('data/cleaned_train.csv')
df_test = pd.read_csv('data/cleaned_test.csv')

df_train.drop(columns = ["Unnamed: 0"], inplace = True) 
df_test.drop(columns = ["Unnamed: 0"], inplace = True) 
# ^^ We did do this in Jupyter and checked and the unnamed column was gone. Also checked the cleaned data here and it shows the column gone. However, was still showing when preview the data in the app for some reason (couldn't figure out why) so pasting the code here too to have it not show.

#build homepage
if page == "🏠 Home":
    st.title(":small_airplane: Predicting Airline Customer Satisfaction")
    st.subheader("This app is designed to help airlines predict customer satisfaction based on a set of existing data!")
    st.image('data:image/jpeg;base64,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')
    st.write(":rewind: Please use the sidebar to the left to drop down to the data set, the visualizations from the exploritory analysis, and then see your predictions")

#build data overview page
if page == "🗄️ Data Overview":
    st.title("🗄️ Data Overview")
    st.subheader("About the Data")
    st.write("The dataset used in this project is called: Airline Passenger Satisfaction, which can be downloaded from Kaggle. The dataset contains information about airline passengers, including features such as flight distance, seat comfort, inflight entertainment, and more.")
    st.image('data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAJMBDQMBIgACEQEDEQH/xAAcAAACAwEBAQEAAAAAAAAAAAADBAACBQEGCAf/xABFEAABAwEDBgwEBAUBCQEAAAACAAEDBAUREhMhIjJSkRQVMUJRVGFxgZOh0TNBYrEGI0PBRFNykvCCNDZjc4Oi0+HxJf/EABoBAAMBAQEBAAAAAAAAAAAAAAACAwEEBQb/xAAnEQACAQMDBAEFAQAAAAAAAAAAAQIDERITQVEEITFhQgUUImKhFf/aAAwDAQACEQMRAD8AxxZXEUUI0cIE90jpsBEEUYkwMKZihRkbYUjgxIzUy0gpERqdLmNiZwU6OMCeCBMxUaxzNxM2OlTkFFi5q0ApohVilw6MY+KVzbNsB4EI62ilZpMOiOl2pgxlLWJCyCF7BmfNlZNYiw9qAUC1ngQygVlJE3EySgQThWwUCEcCdSJOJjPChHEtcoUA4U6kI0ZRRoBRLUOJBKNNcUzCiVHjWgcaG8a25gk8ark044LjgmTMYnk1VwTbgqOCdCMVcVVwTLiquKdCMVcVxwTDguOCYRirgquCacFVwWk2KuCo4JtwVHFbcw9THEmQjWnFZ8RaqMVniPOXg6qPdxM4Ik3DTkjxwYU7DFoLHMZIHHEjhS4keKIUYXLmqeRoAKUYtbciOxc0dHtRGBGEFuQCORVmgT2SXWiW5AIZBceBaGSXHiW5GGaUKG8S0niQyjT5GGaUSCUK0yjQiiTKQrRlnEgSQrVOJAOJOpk3EyDiS0kS2JIUscKdTRNxMk4kJ4lqHESrwfF7JsxcTKeJUKNab06o9OqKohXBmY8X0qhRLYaAUI6dOpiuBkvGqvGtAokIolVSJNCbgqECcKJDcE1xGKOCq4JohVHZbcRoWcFRwTNyq7IuK0e9Ay0cOIR6L8yajL6kCME7FFiXzGaPorBIxRxFDBkwEZEjMLFgEUZo12OJHGLaRmKwQxooxoogiCC3IVsE0a7k0yILuBMmJcVyao8adyaq4JrhcSKNDKJPPGhEK3I24iUaE8SfIFR40ZAIPEqvS4tVOlGqO4jzkZgZ50hbKXOk2lonMl5JiRmwsZssAjtIDCOPCI6S0TkEtbRS4j+cKZVDMReQR2UA2+lNmOI9IlR4VWM0K4iDsOyqHhTkkKAcavGaJOLFHEUvMcQ1nBtLKZFpey5+Ru/OniBZ1f8A74VEA/o0wx/2sLLZVbSSW4QpZRk3sjhChECdKNDKNdKmjkcRJwQ3BOlGhvGmzQjiJuCo4pxwQ3BGYuJ+gRw/SnRjEQ0dZQDHBhXQXxqrH0TiWCLaTcbYQwihAmAJPGqI0XASRmZUEkUXV4zRJlxZFEFQEcFeDRKTIIK2BEBlfCuqMLoi5CzihkyZMUA1OSsPF3AkqOiGgkpOdiqQIyQTdGNBJ1J1SiiANkIhRydDJ1N1xsRcgQiiL/MyOToRpfuDcBc4Pq/dBKL+rcmXbCqi+LFilLCLORP0My2NfJ2RuAuwj/8AWQnjUG2YhPRo48Pyx53u7XdFG1qEviU2H+g3b913qE0Ta9C5CgkC0MtZUn6s0e52+zLhU9GXw7Qj/wCoDt9r0yzWxjSEaeHKVMMZapGzP3O7XrygVPCfx5XFtCf3b2XtHjGkCSpy8Mgxg7tkzve+67k5fmvzn8N5Wp/GZZMSIpAN7gZ3d7mZ78yVybkm9itOCwfs9eYoJCn6iIaYMVXPT04/N5Zha7wvv9Fj1NvWFB/GSVRdFNE7t/cVzK6qs59C/gKQobjzR1uhIt+IpZzw2ZYuLNex1Ejvm5L7mub1dCrT/EM8OlONOObFFSPgd2d87vdy3dF63XsZ9ryzQniyAYqso6cf+KbA/gzve/gyXy9O+o9RK21FTvd/3uP2XKKzxpAL80piJ8V53Pc/S3z9UxgWa/sXRS8K57UZ0YZl5hrUw6JCXhnVxtjZgqC7mb3Xy2jUPcdNHqwmRhnXk47d2qWoEu5vdNBbOzBN6e6XCqhHSR6gJ0YJl52C0Sk/S3mzPuTQ1o6OyXzbOzeKZVKkSMqB6COVNxEvPQ1Y7X3T8NWuql1TT7nHUos2xVkrTz4kxiFe7RrRlG5wyi0zkiUkJFnmwrOmn2SXD1PUK/YvSg2XORBKVKTVBDteDJM62T+VJuu+68+VaTO2FE0JJgHDiLDie5u1+hCORZFTMVXTSQFHhxNy42Zxf5OztyOz3O3csmWstyLRKpoSza70xNf2v+Z9mSrKW9i6onqClQilXljr7cHnURdrQF/5FwK+2C5tP35Emb1NZoyfmSKKl6PSnN9SCc/1Ly9dW/iEfgy0AjhvzRXv358XZ0JDh9vkAkVp0seUFia6kid2Z2/5b51SPSNq+aMaa+J6+Sf6kCoqJxo6jg44iwPys7tmdnuzdy8VXW1+JqLRltOoEs9zBGMd7X5nzC2Z2z+KUobctm1MWR/EddiG53Bqg2e7NnuZ2+d7eC6odFhaTkTyb7Jf02gt3V4TSiPbe7M29k6FbBPDl4YimD55Jxd272d2f0Xm6orXq8VNUWnNNHdiZ5Tc736M97s+dIDSWjSaUMol0h8ibodnZr2XenDkxwlwb9Va1LpZGcYeyenN3Z+9nuWNV2xOOraNP/ogP93QaoaktKOmkxdnI3Ze/wAknPSWjg0oh7nMfdblwzMHwBqLdtHAUY1OiXK4Nc7t0cvIk4bTrBMSGpKMh/UB7iHxZMhYlZOf5mTjH5vmd/Bm5d6ff8P0OiJSzEI8rXs1/oscjVBscsaxoLUhG0JaySbE7sTXXEJNys7vf6Zlt0FmUtIGlFHJJid2kNr3u+TZ+R+5JUksFBTDTU8WTiG+5r3fO/K7u/K6IVoLnlqNu3gsoRt3ZplkssMmH8wRcWf53X33d2ZVKVZb14qj1w/Us05iuMTTKRcyiyyrfpJCeu/qW6cxcYmhDVy6IjKRF05R703HlSMSKsLxmd23XLKc6OTSHEPZeihHzhEsPoiUEUUjcEJ5f44R7mz77kxThXU2rUxzD0HesMQnLVEhLszI0UNolpCMmHxXNOn7Q6ZqjVTxH/s0fbga/wDdOx25hDCMAj2PmSNFRWjJrDhHpPMtOGzZy+Jh+65ZqBr9nY7Yn5uHw5GWvQWnLg/M0kpT2VEOlJhxdDJ+KkEdUvRSdtiU8HsaMdZlMOsKK9SXokxjHaJGFxHnEmjKXJyOEeAc9SRLLqJCWqUseygG8RayxorDtsYx1k46spLrWoQh+YOIvstE6eItXD4syUlocWrh3MnVtyyaewodpCWjku9KzlBLzSThWeQ/Dw4u5J1NJaBav7NcuynGjpu7/IomLmcQ7QjuQirBLREVWSitET0iwj05nZVOCeMPjli7I0mnHkopATkxaPre6FLLoaRCP7dyrJHWFo5UfEHZJVNHWEGjh8MyvGkuRZT9GPbEA1NSRDVSQ5ribOTP2tna7uSFmNBZBkUYlMZMwuZ5rm7Gb91rvY88ulIQj4qkdgl+oRF2BdnXbDBKzdzjnk3dKx0baiHmlvbN6Kz2pAWliLdnVXsuIf4aQvG/7Kh0Mo6tH9vdUUKbEc6iKlaMRHiypEPyZ2ZXGuGUNHJ+LZ0Eo5Y/4MhLsZkNhqdkh7LmVlSgyLqzQ1wmX9Mo93/pRpp8f5kseHsa90JqOctaXD3uzKp00o/q4v8AWyZQiY6k+GGy4kesXiDN63rmUiH4kvgzsyVMJf5Q72S5YR1ovumVJCOu+B0qwR+HFi7XN3QjtPDo4REexJvktklxx2dIe1OqMST6ib8DL2gJf4y49bElQp5S1YhxI7WfVfMI2Q4QQKdVm7TWeRaokPa6cgpKkedo3pQbV0F0bSLaXkOM2eunE2YXGLW1k+FsjEGEV5nhOLWJEilH+pQnRv5KxZ6cLbJMx22WBecCZXA1zSpIfFM9PHahc5FG0l5sJkYJlCVMMEej4xXCtFYLTrrzpcDNNGwdcWPEuHaJayxzn0EB6hVVMMEbMlpkhPapDzlhyVCXlqlaFG5uKPQPbZChlbxCvMSVSWOpXVDpUxJSSPWt+ISLWQyt/EvGnVEh8L+pXXRRIyrWPacbxFrCP2Q5Z4J9GOXCXbyLxxVhYEPhhDzky6JbE31J6aaGUvhzji7XzLPlGsHRKIi7Q+ay2r5dpHitWUecqKjKPjuLrRYQpqnZkEuh70A6ip5uJPx2xi+IIkijaEBaWEUfkviasX4kYklRU87FhQWqS/y9ekesgLmjuS5jSy80RVY1OUJOjfwzFeolk0VVpZRPSWk8en9Kto84cSrkiWD5M/DLKekWH1RGpZdrF4Jt5dkfRUap2RRm9g047gGiEdEYCKTuTQWdKIfFjEuhmvuS51xJc68tpH5vwFqa8mg1JPzpx3IZU8t+epH1WeVeRc5Ceq+pChLcx1ILwdCVGCVJC6NGag0Wix8DTMUqzRkRAkUpQLwkaoVCaCoWMEiYCZc06dzpjI1xnRxnWME6MMyg6RRM1WnXSqFlZZceoS6RpqcIQDnSDzoB1CtGkK2NyzpU6hLnOlzmXTCmSlMZOZLnKlykQTkXVGJyzmFklQiNBIlTKKyRyykMNIuEaAxYlx3TEmxlpFMogM6uDYk3YQLlMKjTqwQogQRFrLOwyuUGoV2qSVmpYl1qeJZ2HWR0aokVpyVGjEVVzFZY27QcJ1x5UAT2VZ1mKNUmDkbElShItVPi4q2LCtvYVxuZc9PLFrDuzpfEtkjxaypkIn5orcuRHTu+xlBIO0O9kYZB2h3svqni+h6nT+UPspxdQ9Tp/KH2UbGLqPR8tNKO0O9FCQdod6+oOL6PqdP5Q+ynF9F1On8ofZY4jrq7bHzG0o7Q70RpR2h3r6X4vo+p0/lD7LvAKPqdP5Y+yR0ii662x81jOI84d7IjVQ7Q719IcX0fU6fyx9lOL6PqdP5Y+yXQQ6+o/qfODVA7Q71OEDtDvX0fwCj6nT+WPspwCj6nT+WPsjQQf6P6/wBPmw6kdod6AdQO0O9fTXF9H1On8sfZTi+i6nT+UPsmVNIx/UL/ABPl8qkdod7IRVA7Q72X1LxfQ9Tp/KH2U4voup0/lD7J1GxN9ZfY+VnqB2h3shvMJc4d6+nLTKz6DJjxfFJJJcws0YMzvezXZ7s9zu7N2PyJGa2LKGAZILNGbRInHBGLizAZ573+h29U5N9RfY+bSk+od6sI4tlfSr2lZDZT/wDNJ8njxPwcGa4XYXdnd2Z2vdmvbv5M65xnZQnk5LPETypRCzRxliJjIW+ea/A73vmb5u2a/chNZcHzYzYecO9HEMXOHey+kaWrsyamKbgMeIagoMLRCz4md8+e7Nc1978vyvva8IWtYxHhGz/mDO+RC4cQuTX59lnfsuu5cyMjNVcHzpgEed6oraOyvpaqGjpwEms+EheYIivjYbsRM17Xtna925FmlX0wwjJxTSkWdijB9MSa5mbODM95ELX33Z3fOzOtyM1FwfgLTCPOHeo8g7Q719AwV9m1M35FBTlHfE2UcGG9jxNezOzPdeN3bffycsjrKYohIrLpRfInITO+YSAmZx1L3fSbOzXX5mv5UZG6p8/MY7XqrDKPOIV9D2VJTV8k0clm08JxCzuNwlc7kQu3I11zg/e1z9jafAKPqdP5beyMg1fR8zHKO0Ko7jtDvX05wCj6nT+W3spwCj6nT+W3ssyN1vR8wM2HnDvRGk+od6+m+L6PqdP5beynF9H1On8tvZGQKsuD5kchHnDvVMsO0O9fT3F9H1Sn8tvZTi+i6pT+UPsjINb0fMTzDtD6LjzjtDvX09xfRdUp/KH2U4vouqU/lD7IuGt6GlFFEpAiiiiAIooogCKKKIAiiiiAIooogCKKKIAGQAT43AXJuR3bkXMhD/KD+1uhRRAEaCHP+UGly6LZ/wDLmUKCJ3zxA+JnZ9Fs6iiAI8Mb8sYv4dqjQxs73Ri2e/k/zpfeoogDpxhLhygCVz3te3I65kYrsGTHC7XXXZrlFEAR4YjuxxiWG669uRVGnga66ENHM2i2Zr11RAF2ABd8Is2J73ubldXUUQBFFFEARRRRAEUUUQBFFFEAf//Z')
    st.link_button("Click here to learn more", "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data", help = "Kaggle: Airline Passenger Satisfaction Dataset", type = 'primary')
    st.subheader("Quick Glance at the Data")
    # Display dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df_train)
    # Column list
    if st.checkbox("Column List"):
        st.code(f"Columns: {df_train.columns.tolist()}")
        if st.toggle('Further breakdown of columns'):
            num_cols = df_train.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df_train.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols} \nObject Columns: {obj_cols}")
    # Shape
    if st.checkbox("Shape"):
        # st.write(f"The shape is {df.shape}") -- could write it out like this or do the next instead:
        st.write(f"There are {df_train.shape[0]} rows and {df_train.shape[1]} columns.")

#build EDA page
if page == "📊 Exploritory Data Analysis":
    st.title("📊 Exploritory Data Analysis")
    num_cols = df_train.select_dtypes(include = 'number').columns.tolist()
    obj_cols = df_train.select_dtypes(include = 'object').columns.tolist()
    eda_type = st.multiselect("What type of EDA are you interested in explorinig?", ["Histogram", "Box Plot", "Scatterplot"])
    #HISTOGRAM
    if "Histogram" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for your Histogram:", num_cols, index = None)
        if h_selected_col:
            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction Hue on Histogrom:"):
                st.plotly_chart(px.histogram(df_train, x = h_selected_col, title = chart_title, color = 'satisfaction', barmode = 'overlay'))
            else:
                st.plotly_chart(px.histogram(df_train, x = h_selected_col, title = chart_title))
    #BOXPLOT
    if "Box Plot" in eda_type:
        st.subheader("Boxplots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for your Boxplot:", num_cols, index = None)
        if b_selected_col:
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Satisfaction Hue On Box Plot"):
                st.plotly_chart(px.box(df_train, x = b_selected_col, y = 'satisfaction', title = chart_title, color = 'satisfaction'))
            else:
                st.plotly_chart(px.box(df_train, x = b_selected_col, title = chart_title))
    #SCATTERPLOT
    if "Scatterplot" in eda_type:
        st.subheader("Scatterplots = Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index = None)
        if selected_col_x and selected_col_y:
            chart_title = f"Relationship of {selected_col_x} vs {selected_col_y}"
            if st.toggle("Satisfaction Hue On Scatterplot"):
                st.plotly_chart(px.scatter(df_train, x = selected_col_x, y = selected_col_y, title = chart_title, color = 'satisfaction'))
            else:
               st.plotly_chart(px.scatter(df_train, x = selected_col_x, y = selected_col_y, title = chart_title))

    # Build Modeling Page      
if page == "📉 Modeling":
    st.title("📉 Modeling")
    st.markdown("On this page, you can see how well different **machine learning** models make predictions.")
    #Set up X and y
    X = df_train.drop(columns = 'satisfaction')
    y = df_train['satisfaction']
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    #Model selection
    model_option = st.selectbox("Select a Model:", ['KNN', 'Logistic Regression', 'Random Forest'], index = None)
    if model_option:
        if model_option == "KNN":
            k_value = st.slider("Select the number of neighbors (k):", 1, 29, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        #create a button & fit your model
        if st.button("Let's see the performance!"):
            model.fit(X_train, y_train)
            #Display results
            st.subheader(f"{model} Evaluation")
            st.text(f"Training Accuracy: {round(model.score(X_train,y_train)*100,2)}%")
            st.text(f"Testing Accuracy: {round(model.score(X_test,y_test)*100,2)}%")
            #Confusion Matrix
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model,X_test,y_test, cmap = 'Blues')
            st.pyplot()