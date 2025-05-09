import pandas as pd
from langchain.tools import Tool

# Load the dataset
df = pd.read_csv("data/listings.csv")

def get_available_properties_by_neighbourhood(neighbourhood: str) -> str:
    filtered = df[
        (df["neighbourhood"].str.lower() == neighbourhood.lower()) &
        (df["availability_365"] > 0)
    ]
    
    if filtered.empty:
        return f"No available properties found in {neighbourhood}."

    result = filtered[["name", "neighbourhood", "room_type", "price"]].head(5)
    return result.to_string(index=False)

# Define as a LangChain tool
check_properties_tool = Tool.from_function(
    name="GetAvailableProperties",
    description="Use this to find Airbnb listings available in a specific neighbourhood.",
    func=get_available_properties_by_neighbourhood
)
