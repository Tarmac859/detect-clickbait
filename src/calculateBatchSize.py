# Given data
total_titles = 32000
input_cost_per_token = 0.0015
output_cost_per_token = 0.002
max_billing_amount = 5.0

# Calculate total number of tokens for all article titles
average_title_length = 10  # Assume an average title length of 10 tokens
total_tokens = total_titles * average_title_length

# Determine maximum batch size for input and output tokens
input_batch_size = min(total_tokens, 90000)  # Maximum tpm is 90,000
output_batch_size = min(total_tokens, 3500)  # Maximum rpm is 3500

# Calculate the total number of API calls and total cost for input and output tokens
num_api_calls = total_tokens // output_batch_size
total_input_cost = input_batch_size * input_cost_per_token
total_output_cost = total_tokens * output_cost_per_token

# Check if the total cost exceeds the maximum billing amount
total_cost = total_input_cost + total_output_cost
if total_cost <= max_billing_amount:
    print("Optimal Batch Size:", output_batch_size)
    print("Total Cost: $", total_cost)
else:
    print("Total Cost exceeds Maximum Billing Amount.")
