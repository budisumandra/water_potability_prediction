
import streamlit as st
import io


def description(df):
    st.markdown('# Dataframe Description Part')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Each Columns Description')
        st.markdown("""
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
                """)
    with col2:
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.subheader('Dataframe Info')
        st.text(s)
    st.subheader('Statistical Description of Each Columns')
    st.write(df.describe())
