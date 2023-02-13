import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from description import description
from visualization import visualization_missingvalue, visualization_comparison
from prediction import predict_water_potability

st.title('Water Potability Prediction')
menu = st.sidebar.selectbox(
    '**Choose Menu**', ['Description', 'Visualization', 'Prediction'])
#display = st.sidebar.checkbox('display code')
df = pd.read_csv('water_potability.csv')

# Function for Description Part
if menu == 'Description':
    display = st.sidebar.checkbox('display code')
    st.subheader('**Data Frame**')
    st.dataframe(df)
    st.markdown('-----')
    st.write(description(df))
    if display:
        st.code("""
        import streamlit as st
import io


def description(df):
    st.markdown('# Dataframe Description Part')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Each Columns Description')
        st.markdown('''
            - ph: pH of 1. water (0 to 14).
            - Hardness: Capacity of water to precipitate soap in mg/L.
            - Solids: Total dissolved solids in ppm.
            - Chloramines: Amount of Chloramines in ppm.
            - Sulfate: Amount of Sulfates dissolved in mg/L.
            - Conductivity: Electrical conductivity of water in μS/cm.
            - Organic_carbon: Amount of organic carbon in ppm.
            - Trihalomethanes: Amount of Trihalomethanes in μg/L.
            - Turbidity: Measure of light emiting property of water in NTU.
            - Potability: Indicates if water is safe for human consumption. Potable - 1 and Not potable - 0
                ''')
    with col2:
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.subheader('Dataframe Info')
        st.text(s)
    st.subheader('Statistical Description of Each Columns')
    st.write(df.describe()) 
        """, language='python')

# Function for Visualization Part
if menu == 'Visualization':
    sub_menu_visualization = st.sidebar.radio(
        label='Sub Menu', options=['Missing Value', 'Potability Composition'])
    if sub_menu_visualization == 'Missing Value':
        st.write(visualization_missingvalue(df))
        display = st.sidebar.checkbox('display code')
        if display:
            st.code("""
            import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from datetime import date
    import plotly.express as px


    def visualization_missingvalue(df):
        colors = ['#DFF6FF', '#47B5FF', '#256D85', "#06283D"]
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        nan_data_numerical = round(
            100*(df.isna().sum())/(len(df.index)), 2).sort_values(ascending=False).to_frame()
        fig, axs = plt.subplots(1, 1, figsize=(10, 7.5))
        plt1 = sns.heatmap(nan_data_numerical, annot=True, cmap=colors, ax=axs)
        #axs.set_title('Numerical Columns\n',fontweight = 'bold',fontsize=15)
        plt.suptitle('MISSING VALUES \n PER COLUMN\n',
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        st.write('\n')
        st.pyplot(fig)
            """)
    else:
        st.write(visualization_comparison(df))
        display = st.sidebar.checkbox('display code')
        if display:
            st.code("""
           import streamlit as st
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            import pandas as pd
            from datetime import date
            import plotly.express as px

            def visualization_comparison(df):
                df.columns = df.columns.str.lower().str.replace(' ', '_')
                colors = ['#DFF6FF', '#47B5FF', '#256D85', "#06283D"]
                nan_data_numerical = round(
                    100*(df.isna().sum())/(len(df.index)), 2).sort_values(ascending=False).to_frame()
                data = df.copy()
                list_nan_features = list(
                    nan_data_numerical[nan_data_numerical[0] > 0].index)
                for col in list_nan_features:
                    data[col] = data[col].replace(np.nan, data[col].median())
                fig = plt.figure(figsize=(10, 6))
                ax = sns.countplot(data['potability'],
                                order=data['potability'].value_counts().index)

                # Create annotate
                for i in ax.patches:
                    ax.text(x=i.get_x()+i.get_width()/2, y=i.get_height()/7, s=f"{np.round(i.get_height()/len(data)*100,0)}%",
                            ha='center', size=50, weight='bold', rotation=90, color='white')
                for p in ax.patches:
                    ax.annotate(format(p.get_height(), '.0f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center',
                                va='center',
                                xytext=(0, 10),
                                textcoords='offset points')

                plt.title("Comparison of Potable and Not Potable Samples \n",
                        size=15, weight='bold')
                plt.annotate(text="Not safe water for human consumption", xytext=(0.5, 1790), xy=(0.2, 1250),
                            arrowprops=dict(arrowstyle="->", color='blue', connectionstyle="angle3,angleA=0,angleB=90"), color='black')
                plt.annotate(text="Safe water for human consumption", xytext=(0.8, 1600), xy=(1.2, 1000),
                            arrowprops=dict(arrowstyle="->", color='blue',  connectionstyle="angle3,angleA=0,angleB=90"), color='black')

                # Setting Plot
                sns.despine(right=True, top=True, left=True)
                ax.axes.yaxis.set_visible(False)
                st.pyplot(fig)
           """)
        # else:
        #    st.write(visualization_distribution(df))
        #    if display:
        #        st.code("""

        #        """)

if menu == 'Prediction':
    display = st.sidebar.checkbox('display code')
    st.sidebar.markdown('**Parameters**')
    ph = st.sidebar.slider('pH', 0.0, 14.0)
    hardness = st.sidebar.slider('hardness', 47.0, 324.0)
    solids = st.sidebar.slider('solids', 320.0, 61226.0)
    chloramines = st.sidebar.slider('chloramines', 0.0, 14.0)
    sulfate = st.sidebar.slider('sulfate', 129.0, 480.0)
    conductivity = st.sidebar.slider('conductivity', 181.0, 752.0)
    organic_carbon = st.sidebar.slider('organic_carbon', 2.0, 27.0)
    trihalomethanes = st.sidebar.slider('trihalomethanes', 0.0, 124.0)
    turbidity = st.sidebar.slider('turbidity', 1.0, 6.8)

    params = dict()
    params['ph'] = ph
    params['hardness'] = hardness
    params['solids'] = solids
    params['chloramines'] = chloramines
    params['sulfate'] = sulfate
    params['conductivity'] = conductivity
    params['organic_carbon'] = organic_carbon
    params['trihalomethanes'] = trihalomethanes
    params['turbidity'] = turbidity

    data = pd.DataFrame(params, index=[0])
    st.subheader('Data Input')
    st.write(data)
    st.subheader('Prediction:')
    st.write(predict_water_potability(params))

    if display:
        st.code("""
        import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import pandas as pd

with open('water_potability.bin', 'rb') as f_in:
    dv, rf = pickle.load(f_in)


def predict_single(potability_parameters):
    X = dv.transform([potability_parameters])
    y_pred = rf.predict_proba(X)[:, 1]
    return y_pred[0]


def predict_water_potability(potability_parameters):
    prediction = predict_single(potability_parameters)
    potability = prediction >= 0.5
    result = {
        'Potability Probability': float(prediction),
        'Is Potable?': 'Potable' if potability else 'Not Potable'}
    return pd.DataFrame(result, index=[0])

        """)
