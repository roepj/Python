import os
import json
import pandas as pd
import plotly.express as px
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("API_KEY"),
)

api_key = os.environ.get("API_KEY")
if api_key:
    print("API key set successfully.")
else:
    print("Error: API key not found in environment variables.")

# Read previous day followups and save to variable
with open("02152024 followups.txt", "r") as file:
    previous_followups = file.read()

with open('02162024_Workday.json') as f:
    transcript = json.load(f)

# Extract the text from the JSON data
full_text = transcript['text']

# Define the maximum token limit for each segment
max_tokens_per_segment = 15000  # Adjust as needed

# Split the full text into smaller segments based on the token limit
segments = []
current_segment = ''
current_tokens = 0

for segment in transcript['segments']:
    segment_text = segment['text']
    segment_tokens = len(segment_text.split())  # Count tokens by splitting the text

    if current_tokens + segment_tokens > max_tokens_per_segment:
        segments.append(current_segment)
        current_segment = ''
        current_tokens = 0

    current_segment += segment_text
    current_tokens += segment_tokens

# Add the last segment
segments.append(current_segment)

# Print the segments
for i, segment in enumerate(segments, 1):
    print(f"Segment {i}:\n{segment}\n--- Segment End ---\n")

system_msg_test = """You are a helpful assistant who understands, 
data science, Python, SQL, Microsoft SQL Server, Pycharm, and Microsoft Office. You are familiar with 
the Electronic Health Record named 'Epic'.  You are able to evaluate a transcript from audio recordings 
and summarize them. You can  identify specific follow-ups and items that need more attention.  Please
attempt to identify and call out record IDs, phone numbers, dates, within each follow up. Please format in a manner that
the user could follow up on these items the next day.  Please indicate priority level, at your discretion.
"""

user_msg_test = f'''\n Please give 2-4 specific follow ups for the following:\n{current_segment}\n\n   
Attempt to  identify specific information, like any kind of record identifier. Keep in mind the user's job 
is as an Epic EHR analyst.  There are many emails, tickets, build requests, meetings to track.
\n\n 
Please try to limit the follow ups to 4 items and have each one be specific\nAfter the followups, please describe some
success/progress made by the user in the transcript.  Limit this to 3 sentences and make it specific to tasks
accomplished.
\n
As further reference, the follow-ups you provided to me yesterday are:
\n{previous_followups}
'''

# Create a message list from the system and user messages. Assign to msgs_test.
msgs_test = [
    {"role": "system", "content": system_msg_test},
    {"role": "user", "content": user_msg_test}
]

# Store response from API
rsps_test = client.chat.completions.create(model="gpt-4-0125-preview",
                                           messages=msgs_test)

# Print content of API response to console
print(rsps_test.choices[0].message.content)

# Write new followups file - Initialize with API response, no user input
with open("Prev_Day_Followups.txt", "w") as file:
    file.write(rsps_test.choices[0].message.content)

# Initialize conversation history with the initial system and user messages
conversation_history = [
    {"role": "system", "content": system_msg_test},
    {"role": "user", "content": user_msg_test}
]
initial_prompt_response = rsps_test.choices[0].message.content

followups = conversation_history

with open("Prev_Day_Followups.txt", "a") as file:
    for _ in range(5):  # Adjust the number of iterations as needed
        # User input from the terminal
        user_input = input("User:  ")

        # Add user message to conversation history
        followups.append({"role": "user", "content": user_input})

        # Add AI's response to conversation history
        # conversation_history.append({"role": "assistant", "content": initial_prompt_response})

        # Send messages to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=conversation_history
        )

        # Extract and print AI's response
        ai_reply = response.choices[0].message.content
        print(f"AI: {ai_reply}")

        # Add AI's response to conversation history
        followups.append({"role": "assistant", "content": ai_reply})

        # Update initial prompt response after the first iteration
        if initial_prompt_response is None:
            initial_prompt_response = ai_reply

        # Write the user's input and AI's response to the file
        file.write(f"User: {user_input}\n")
        file.write(f"AI: {ai_reply}\n")
        file.write("\n")  # Add a separator between interactions
