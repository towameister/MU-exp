import numpy as np
from del_eff_kmeans import Kmeans, QKmeans, DCKmeans
import time
import pickle

def online_deletion_stream(num_dels, model):
    t0 = time.time()
    c = 1
    for _ in range(num_dels):
        dr = np.random.choice(model.n, size=1)[0]
        print(f'processing deletion request # {c}...')
        model.delete(dr)
        c += 1
    t = time.time()
    print(f'Total time to process {c - 1} deletions is {t - t0}')


def prepare_mnist():
    data_file = 'kmeans_data_deletion_NeurIPS19_datasets_scaled.p'
    with open(data_file, mode='rb') as f:
        data = pickle.load(f)
    _mnist = data['mnist'][0]
    _mnist_labels = data['mnist'][1]
    _n = _mnist.shape[0]
    _k = data['mnist'][2]
    return _mnist, _mnist_labels, _n, _k

def dckfinallosscalc(dck):
    # helper method that calculates the loss over the final partition using the final centroids
    final_centroids = dck.dc_tree[0][0].centroids

    loss=0
    data = dck.dc_tree[1][0].data
    assignments = dck.dc_tree[1][0].assignments
    for i in range(1, len(dck.dc_tree[1])):
        data = np.append(data, dck.dc_tree[1][i].data, axis=0)
        assignments = np.append(assignments, dck.dc_tree[1][i].assignments, axis=0)
    for x in range(data.shape[0]):
        d = np.linalg.norm(data[x,:] - final_centroids, axis=1)
        loss += np.min(d)**2
    loss = loss / data.shape[0]
    return loss


def comparison():
    print('______________________COMPARISON__________________________')
    kmeans = Kmeans(k)
    centers, assignments, loss = kmeans.run(mnist.copy())
    print(f'Kmeans Training Clustering loss is {loss}')
 

    qkmeans = QKmeans(k, 0.05)
    q_centers, q_assignments, q_loss = qkmeans.run(mnist.copy())
    print(f'QKmeans Training Clustering loss is {q_loss}')
   

    dckmeans = DCKmeans([k, k], [1, 16])
    dc_centers, dc_assignments, dc_loss = dckmeans.run(mnist.copy(), assignments=True)
    print(f'DCKmeans Training Clustering loss is {dc_loss}')
    

    print('Simulation deletion stream for Kmeans')
    online_deletion_stream(100, kmeans)
    print(f'Kmeans Clustering loss after unlearning is {kmeans.loss}')
   

    print('Simulation deletion stream for Qkmeans')
    online_deletion_stream(100, qkmeans)
    print(f'QKmeans Clustering loss after unlearning is {qkmeans.minloss}')
   

    print('Simulation deletion stream for DCkmeans')
    online_deletion_stream(100, dckmeans)
    print(f'DCKmeans Clustering loss after unlearning is {dckmeans.loss}')
    


def qkmeansvaryeps():
    print('______________________QKMEANS WITH VARYING EPSILON__________________________')
    qkmeans1 = QKmeans(k, 0.05)
    q_centers1, q_assignments1, q_loss1 = qkmeans1.run(mnist.copy())
    print(f'QKmeans 0.05 Training Clustering loss is {q_loss1}')
    qkmeans2 = QKmeans(k, 0.005)
    q_centers2, q_assignments2, q_loss2 = qkmeans2.run(mnist.copy())
    print(f'QKmeans 0.005 Training Clustering loss is {q_loss2}')
    qkmeans3 = QKmeans(k, 0.5)
    q_centers3, q_assignments3, q_loss3 = qkmeans3.run(mnist.copy())
    print(f'QKmeans 0.5 Training Clustering loss is {q_loss3}')

    print('Simulation deletion stream for Qkmeans 0.05')
    online_deletion_stream(100, qkmeans1)
    print(f'QKmeans Clustering loss after unlearning is {qkmeans1.minloss}')

    print('Simulation deletion stream for Qkmeans 0.005')
    online_deletion_stream(100, qkmeans2)
    print(f'QKmeans Clustering loss after unlearning is {qkmeans2.minloss}')

    print('Simulation deletion stream for Qkmeans 0.5')
    online_deletion_stream(100, qkmeans3)
    print(f'QKmeans Clustering loss after unlearning is {qkmeans3.minloss}')


def dckmeansvaryw():
    print('______________________DCKMEANS WITH VARYING W__________________________')
    dckmeans1 = DCKmeans([k, k], [1, 16])
    dc_centers, dc_assignments, dc_loss = dckmeans1.run(mnist.copy(), assignments=True)
    print(f'DCKmeans 16 Training Clustering loss is {dc_loss}')

    dckmeans2 = DCKmeans([k, k], [1, 160])
    dc_centers, dc_assignments, dc_loss = dckmeans2.run(mnist.copy(), assignments=True)
    print(f'DCKmeans 160 Training Clustering loss is {dc_loss}')

    dckmeans3 = DCKmeans([k, k], [1, 1600])
    dc_centers, dc_assignments, dc_loss = dckmeans3.run(mnist.copy(), assignments=True)
    print(f'DCKmeans 1600 Training Clustering loss is {dc_loss}')

    print('Simulation deletion stream for DCkmeans 16')
    online_deletion_stream(100, dckmeans1)
    d1loss = dckfinallosscalc(dckmeans1)
    print(f'DCKmeans Clustering loss after unlearning is {d1loss}')

    print('Simulation deletion stream for DCkmeans 160')
    online_deletion_stream(100, dckmeans2)
    d2loss = dckfinallosscalc(dckmeans2)
    print(f'DCKmeans Clustering loss after unlearning is {d2loss}')


    print('Simulation deletion stream for DCkmeans 1600')
    online_deletion_stream(100, dckmeans3)
    d3loss = dckfinallosscalc(dckmeans3)
    print(f'DCKmeans Clustering loss after unlearning is {d3loss}')

if __name__ == '__main__':
    mnist, mnist_labels, n, k = prepare_mnist()
    comparison()
    qkmeansvaryeps()
    dckmeansvaryw()
