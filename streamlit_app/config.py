"""

Config file for Streamlit App

"""

from member import Member


TITLE = "PenPyText"

TEAM_MEMBERS = [
    Member(
        name="Samy Ait-Ameur",
        #linkedin_url="",
        github_url="https://github.com/SamyAitAmeur",
    ),
    #Member("Samy Ait-Ameur"),
    Member(
        name="Sophie Amédro",
        linkedin_url="https://www.linkedin.com/in/sophie-amedro-072775201/",
        github_url="https://github.com/SophieAmedro",
    ),
    #Member("Sophie Amedro"),
    Member(
        name="Stéphane Tchatat",
        linkedin_url="https://www.linkedin.com/in/st%C3%A9phane-tchatat-53787a45/",
        github_url="https://github.com/tnstephane",
    ),
    #Member("Stéphane Tchatat"),
]

PROMOTION = "Promotion Data Scientist - April 2021"
