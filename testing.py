import torch
from bleurt import score

# print("CUDA Available: ", torch.cuda.is_available())
# print("Number of GPUs: ", torch.cuda.device_count())

checkpoint = "/home/vitor/Documents/phd/bleurt/BLEURT-20"
references = ['Workable, recruit people by creating beautiful job posts through an API. Creating a candidate', 'Workable, recruit people by creating beautiful job posts through an API. Creating a candidate']
candidates = ["Create a candidate profile for John Doe for the job with shortcode DEV123.", "Create a candidate for the job with shortcode 55667 using the API key 'workable-api'."]

scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references=references, candidates=candidates)

print(scores)