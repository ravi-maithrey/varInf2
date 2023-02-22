from scipy.sparse import lil_matrix
from matrix import *
import scipy.sparse
import time
import matplotlib.pyplot as plt

print("Please enter choice of dataset: \n")
print("1. 500-500-3\n")
print("2. 100-k\n")
choice = int(input("Please enter your choice: 1 (or) 2 [To Debug, type 0] - "))

name = "40-40-2"
data = "hw2data/data-40-20-2.txt"
rank = [1]
rank1 = 1

if choice == 1:
    name = "500-500-3"
    data = "hw2data/data-500-500-3.txt"
    rank = [1, 2, 3, 5, 10, 20]
    rank1 = 3

if choice == 2:
    name = "ml-100k"
    data = "hw2data/ml-100k.txt"
    rank = [1, 2, 3, 5, 20, 100]
    rank1 = 5

# reading the dataset and storing in a sparse matrix

with open(data, encoding="utf-8") as d:
    test = []
    instance = d.readline()
    # splitting the instance at space
    users, items, scores = map(int, instance.split())
    # storing users and items into a list of lists (sparse) matrix
    m = scipy.sparse.lil_matrix((users, items))
    # doing a half split
    split = int(scores / 2)
    for _ in range(split):
        inst = d.readline().split()
        x = int(inst[0])
        y = int(inst[1])
        # m will be our training data
        m[x, y] = float(inst[2])
    for _ in range(scores - split):
        inst = d.readline().split()
        x = int(inst[0])
        y = int(inst[1])
        test.append((x, y, float(inst[2])))

# making sure m is a sparse matrix
m = m.tocsr()

# baseline
a, b = [], []
c = scipy.sparse.find(m)[2].mean()
for i in range(0, m.shape[0]):
    if len(scipy.sparse.find(m[i])[0]) == 0:
        a.append(c)
    else:
        a.append(m[i].mean())
for j in range(0, m.shape[1]):
    if len(scipy.sparse.find(m.T[j])[0]) == 0:
        b.append(c)
    else:
        b.append(m.T[j].mean())


# defning an RMSE function (since we have to use it many times)
def rmse(true, pred):
    # convert them to np.array just in case
    true = np.array(true)
    pred = np.array(pred)
    difference_2 = (true - pred) ** 2
    error = np.sqrt(difference_2.mean())
    return error


# calculating the baseline score and the baseline rmse
true_scores, base_preds = [], []
for instance in test:
    # unpacking
    user, item, score = instance
    true_scores.append(score)
    base_preds.append((a[user] + b[item]) / 2)
    base_rmse = rmse(true_scores, base_preds)

print("Baseline rmse is: ", base_rmse)

# part 1 (given dataset/rank has already been chosen)
part1_rmse, times = [], []
start = time.time()
time_taken = 0
# since we used a generator, we can step through each u, v and see how the rmse
# goes down per iteration (till convergence or end of 100 iterations)
for u, v in matrix_factorization(m, rank1):
    part1_preds = []
    end = time.time()
    time_taken += end - start
    times.append(time_taken)
    for instance in test:
        user, item, score = instance
        pred = u[user].dot(v[item])  # making the predictions
        part1_preds.append(pred)
    part1_rmse.append(rmse(true_scores, part1_preds))  # rmse per iteration
    start = time.time()  # resetting time for next iteration

print("For Part 1: \n")
print("Total Iterations: ", len(part1_preds), "\n")
if len(part1_preds) < 100:
    print("We had early stopping due to difference between iterations < 0.01\n")
print("RMSE: ", part1_rmse)
print("Time taken: ", times)
print("\n")

# part 2
# part 2 (given dataset has already been chosen)
part2_rmse, times2 = [], []
start = time.time()
time_taken = 0
for r in rank:
    # we do this to get the last item from our generator
    end = time.time()
    time_taken += end - start
    times2.append(time_taken)
    output = list(matrix_factorization(m, r))
    final = output[-1]
    u, v = final
    part2_preds = []
    for instance in test:
        user, item, score = instance
        pred = u[user].dot(v[item])
        part2_preds.append(pred)
    part2_rmse.append(rmse(true_scores, part2_preds))
    start = time.time()

print("For Part 2: ")
print("For the given ranks: ", rank)
print("The RMSE changes as follows: ", part2_rmse)

plt.plot(times, part1_rmse)
plt.title(name + " Part 1")
plt.xlabel("Time taken ")
plt.ylabel("RMSE")
plt.savefig("Part 1 " + name + ".png")

plt.clf()

plt.plot(rank, part2_rmse)
plt.xticks(rank)
plt.title(name + " Part 2")
plt.xlabel("Rank")
plt.ylabel("Chnage in RMSE")
plt.savefig("Part 2 " + name + ".png")

plt.clf()

plt.plot(times2, part2_rmse)
plt.title(name + " Time taken")
plt.xlabel("Time")
plt.ylabel("Change in RMSE")
plt.savefig(name + "_time_rank.png")

print("Please check your current directory for generated plots")
