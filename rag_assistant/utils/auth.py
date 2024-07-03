import hmac
import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""

    if st.session_state.get("password_correct", False):
        return True

    try:

        if "password" not in st.secrets:
            # no password required
            st.session_state["password_correct"] = True
            return True

    except FileNotFoundError:
        # no secrets.toml so no password required
        # no password required
        st.session_state["password_correct"] = True
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

        # Return True if the password is validated.

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False
