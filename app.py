
import streamlit as st
from offer_chain import get_offers

st.set_page_config(page_title="Personalized Offer Assistant")

st.title("🛒 Personalized Offer Assistant")

username = st.text_input("Enter Customer Name")

if st.button("Generate Offers"):
    if username:
        response = get_offers(username)
        st.write(response)
    else:
        st.warning("Please enter a customer name.")
