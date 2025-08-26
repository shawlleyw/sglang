import pickle
import matplotlib.pyplot as plt

# self.iteration_bsz = []
# self.req_start_iteration: Dict[str, int] = {}
# self.req_end_iteration: Dict[str, int] = {}

with open('scheduler_stats.pkl', 'rb') as f:
    iteration_bsz, req_start_iteration, req_end_iteration = pickle.load(f)
    
niters = len(iteration_bsz)

req_to_id = {}

req_cnt = 0 
for req in req_start_iteration.keys():
    req_cnt += 1
    req_to_id[req] = req_cnt
    
req_id_start = {}
for req, start_iter in req_start_iteration.items():
    req_id_start[req_to_id[req]] = start_iter
    
req_id_end = {}
for req, end_iter in req_end_iteration.items():
    req_id_end[req_to_id[req]] = end_iter
    
print("Total iterations:", niters)
print("len(req_id_start):", len(req_id_start))
print("len(req_id_end):", len(req_id_end))

# Plot
plt.figure(figsize=(12, 5))

# 1. Batch size over iterations
plt.subplot(1, 2, 1)
plt.plot(range(niters), iteration_bsz, linestyle='-')
plt.title("Batch Size over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Batch Size")
plt.yticks(range(0, max(iteration_bsz)+1, max(1, max(iteration_bsz)//10)))
plt.grid(True)

# 2. Request lifespan over iterations
plt.subplot(1, 2, 2)
for req_id in req_id_start:
    if req_id not in req_id_end or req_id not in req_id_start:
        # print(f"Warning: req_id {req_id} missing start or end iteration.")
        continue
    start = req_id_start[req_id]
    end = req_id_end[req_id]
    plt.hlines(y=req_id, xmin=start, xmax=end, linewidth=1)

plt.title("Request Lifespan over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Request ID")
# plt.grid(True)

plt.tight_layout()
plt.savefig("scheduler_stats.png", dpi=300)