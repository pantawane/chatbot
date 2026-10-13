
from backend import chat

response = chat(
    "Search the web for the latest gold price per kilogram. "
    "Summarize the results and include the source URLs.",
    thread_id="search-test"
)

print(response)






























# from ddgs import DDGS

# with DDGS(timeout=10) as ddgs:
#     print(
#         list(
#             ddgs.text(
#                 "gold price per kilogram today",
#                 max_results=3,
#             )
#         )
#     )


