import numpy as np
from numpy.linalg import svd, norm, inv
import math, cmath


# get model M (homogeneous coordinates)
model = []
for y in range(0,6):
    for x in range(0, 8):
        model.append([x*25,y*25, 1])

# get 9 images (homogeneous coordinates)
all_images = []
little_m = []
count = 0
with open("imgpoints.txt", 'r') as f:
    for line in f.readlines():
        x_y = line.split(' ')
        x_y = [float(num) for num in x_y]
        x_y.append(1.0)
        little_m.append(x_y)
        count += 1
        if count == 48:
            all_images.append(little_m)
            little_m = []
            count = 0


# Part 1: solve for homogeny

# build M using formula from appendix A
# right now only for all_images[0]
M = []
for i, point in enumerate(model):
    point_matrix = [0,0]
    neg_point = [p*-1 for p in point]
    point_matrix[0] = neg_point + [0]*len(point)
    neg_u_M = [p*all_images[0][i][0] for p in point]
    point_matrix[0].extend(neg_u_M)

    point_matrix[1] = [0]*len(point) + neg_point
    neg_v_M = [p*all_images[0][i][1] for p in point]
    point_matrix[1].extend(neg_v_M)

    M.extend(point_matrix)

# solve for H using SVD
u, s, v_t = svd(M)
h = v_t[-1] # second to last row
H = [ [0]*3 for i in range(3)]
count = 0
for i in range(3):
    for j in range(3):
        H[i][j] = h[count]
        count += 1
H = np.asarray(H)
print("Homography: \n", H)

# get error for each image
errors = []
for image in all_images:
    error = 0
    for point in range(len(model)):
        u = model[point]
        p = image[point]
        H_p = np.dot(H,p)
        normalized = H_p/H_p[2]
        euclidean = math.dist(u, H_p)
        error += euclidean
    errors.append(error)

print("Total image errors: \n", errors)

# Part 2: solve for A, the intrinsic matrix

# to contruct L (or V, in the paper), need v_1_2, v_1_1, v_2_2
# vij = [hi1hj1, hi1hj2 + hi2hj1, hi2hj2,
# hi3hj1 + hi1hj3, hi3hj2 + hi2hj3, hi3hj3] T

i, j = 1, 2
v_1_2 = np.asarray([H[0][i]*H[0][j], H[0][i]*H[1][j] + H[1][i]*H[0][j], H[1][i]*H[1][j], H[2][i]*H[0][j] + H[0][i]*H[2][j], H[2][i]*H[1][j] + H[1][i]*H[2][j], H[2][i]*H[2][j]])
i, j = 1, 1
v_1_1 = np.asarray([H[0][i]*H[0][j], H[0][i]*H[1][j] + H[1][i]*H[0][j], H[1][i]*H[1][j], H[2][i]*H[0][j] + H[0][i]*H[2][j], H[2][i]*H[1][j] + H[1][i]*H[2][j], H[2][i]*H[2][j]])
i, j = 2, 2
v_2_2 = np.asarray([H[0][i]*H[0][j], H[0][i]*H[1][j] + H[1][i]*H[0][j], H[1][i]*H[1][j], H[2][i]*H[0][j] + H[0][i]*H[2][j], H[2][i]*H[1][j] + H[1][i]*H[2][j], H[2][i]*H[2][j]])

L = [v_1_2.T, (v_1_1 - v_2_2).T]

u, s, v_t = svd(L)
print("v_t: ", v_t)
b = v_t[-1] # second to last row
print("b: ", b)

# by definition
B_0 = b[0]
B_1 = b[1]
B_2 = b[2]
B_3 = b[3]
B_4 = b[4]
B_5 = b[5]

# get intrinsic params
w = B_0*B_2*B_5 - (B_1**2)*B_5 - B_0*(B_4**2) + 2*B_1*B_3*B_4 - B_2*(B_3**2)
d = B_0*B_2 - (B_1**2)

# note: added absolute values just to get numbers for part 2 and 3 to make sure they work in theory
# even though if my homogeny was right then the numbers would not be negative
alph = math.sqrt(abs(w/(d*B_0)))
beta = math.sqrt(abs(w/(d**2*B_0)))
gamma = B_1 * math.sqrt(abs(w/(d**2*B_0)))
u_c = (B_1*B_4 - B_2*B_3)/d
v_c = (B_1*B_3 - B_0*B_4)/d

print("alpha: ", alph)
print("beta: ", beta)
print("gamma: ", gamma)
print("u_c: ", u_c)
print("v_c: ", v_c)

A = np.asarray([[alph, gamma , u_c], [0, beta, v_c], [0, 0, 1]])
print("A: ", A)

# Part 3: calculate extrinsic matrix

h_1 = np.asarray(H[0])
h_2 = np.asarray(H[1])
h_3 = np.asarray(H[2])
lamda = 1/norm(np.dot(inv(A),h_1))
r_1 = lamda*np.dot(inv(A),h_1)
r_2 = lamda*np.dot(inv(A),h_2)
r_3 =r_1*r_2

t = lamda*np.dot(inv(A), h_3)
R = np.asarray([r_1, r_2, r_3])
print("R: ", R)
print("t: ", t)









